import torch
import argparse
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch import distributed, optim
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
from sklearn.cluster import SpectralClustering

def train_one_epoch():
   for step, (x_i, x_j, _) in enumerate(train_loader):
      loss_epoch = 0
      x_i = x_i.cuda(args.local_rank, non_blocking=True)
      x_j = x_j.cuda(args.local_rank, non_blocking=True)
      z_i, z_j, c_i, c_j = model(x_i, x_j)
      loss_instance = criterion_instance(z_i, z_j)
      loss_cluster = criterion_cluster(c_i, c_j)
      loss = loss_instance + loss_cluster
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if step % 50 == 0:
         print(
               f"Step [{step}/{len(train_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
         loss_epoch += loss.item()
   return loss_epoch

# def NCT_CRC_test():
#     print("start validate:")
#     predict_vector, labels_vector, feature_vector = inference(dataloader_tset, model, args.device)
#     nmi, ari, f, acc = evaluate(labels_vector, predict_vector)
#     print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
 



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
   dataset_test = NCT_CRC_Data(train=False)
   # dataloader_train = torch.utils.data.DataLoader(
   #    dataset_train,
   #    batch_size=args.batch_size,
   #    shuffle=False,
   #    drop_last=True,
   #    num_workers=args.workers,
   # )
   # dataset_test = NCT_CRC_Data(train=False)
   # dataloader_tset = torch.utils.data.DataLoader(
   #    dataset_test,
   #    batch_size=args.batch_size,
   #    shuffle=False,
   #    drop_last=True,
   #    num_workers=args.workers,
   # )


   dist.init_process_group(backend='nccl')
   torch.cuda.set_device(args.local_rank)

   train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)

   train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), pin_memory=True)
   
   model = network.Network(args.vit_type, args.feature_dim, class_num)
   optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
   model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[args.local_rank])
   optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
   cudnn.benchmark = True
   
   loss_device = 'useless'
   criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).cuda(args.local_rank)
   criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).cuda(args.local_rank)
      # train
   for epoch in range(100):
      train_sampler.set_epoch(epoch)
      loss_epoch = train_one_epoch()
      if epoch % 10 == 0:
         save_model(args, model, optimizer, epoch)
         print(f"Epoch [{epoch}/{args.epochs}]\t train_Loss: {loss_epoch / len(train_loader)}")