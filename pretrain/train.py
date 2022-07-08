import os
import numpy as np
# from sklearn import datasets
import torch
import torchvision
import argparse
from modules import transform, resnet, network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
from readdata import NCT_CRC_Data, dataloader
from cluster import inference
from evaluation.evaluation import evaluate
from refinement import Refinement, clusterToList, mergeClusters
from torch.utils.tensorboard import SummaryWriter
import tempfile
def NCT_CRC_train(args):
    loss_epoch = 0
    for step, (x_i, x_j, _) in enumerate(dataloader_train):
        x_i = x_i.to(args.device)
        x_j = x_j.to(args.device)
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster
        loss.backward()
        if (step + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        if step % (args.accumulation_steps * 50) == 0:
            print(
                f"Step [{step}/{len(dataloader_train)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
        loss_epoch += loss.item()
    return loss_epoch

def NCT_CRC_test():
    print("start validate:")
    predict_vector, labels_vector, feature_vector = inference(dataloader_tset, model, args.device)
    nmi, ari, f, acc = evaluate(labels_vector, predict_vector)
    print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
    print("start refine")
    a = Refinement(feature_vector, predict_vector, labels_vector, 9)
    return acc
def refine():
    print("start refine")
    predict_vector, labels_vector, feature_vector = inference(dataloader_tset, model, args.device)
    print(f"labelvector{labels_vector}")
    feature_vector = feature_vector/100
    a = Refinement(feature_vector, predict_vector, labels_vector, 9)
    a = mergeClusters(a)
    cluster_list, label_list = clusterToList(a)
    nmi, ari, f, acc = evaluate(label_list, cluster_list) 
    print('after refinement NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))


if __name__ == "__main__":
    # loading args...
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data

    class_num = 9
    dataset_train = NCT_CRC_Data()
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )
    dataset_test = NCT_CRC_Data(train=False)
    dataloader_tset = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )
    # initialize model
    # res = resnet.get_resnet(args.resnet)
    model = network.Network(args.vit_type, args.feature_dim, class_num)
    if args.resume is not None:
       print(f"resume from:{args.resume}")

       checkpoint = torch.load(args.resume, map_location='cpu')
       print(checkpoint.keys())
       model.load_state_dict(checkpoint['net'], strict=False)
    
    model = model.to(args.device)
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # if args.reload:
    #     print("reload")
    #     model_fp = '/data_sda/lf/tissuse_project/tissue_segmentation/save/checkpoint_30.tar' #os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
    #     checkpoint = torch.load(model_fp)
    #     model.load_state_dict(checkpoint['net'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     args.start_epoch = checkpoint['epoch'] + 1
    loss_device = torch.device(args.device)
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)
    # train
    writer = SummaryWriter()
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        acc = NCT_CRC_test()
        loss_epoch = NCT_CRC_train(args)
        if epoch % 10 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t train_Loss: {loss_epoch / len(dataloader_train)}")

        # writer.add_scalar("loss", epoch, loss_epoch)
        # writer.add_scalar("acc", epoch, acc)
        # refine()
    save_model(args, model, optimizer, args.epochs)
    writer.close()
