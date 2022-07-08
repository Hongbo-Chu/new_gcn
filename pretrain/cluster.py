import os
import argparse
import torch
import torchvision
import numpy as np
from utils import yaml_config_hook
from modules import resnet, network, transform
from evaluation import evaluation
from torch.utils import data
import copy


def inference(loader, model, device):
    model.eval()
    predict_vector = []
    feature_vector = []
    labels_vector = []
    for step, (x, y, label) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c = model.forward_cluster(x)
            i = model.forward_instance(x)
        c = c.detach()
        i = i.detach()
        predict_vector.extend(c.cpu().detach().numpy())
        feature_vector.extend(i.cpu().detach().numpy())
        labels_vector.extend(label.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    predict_vector = np.array(predict_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return predict_vector, labels_vector, feature_vector


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     config = yaml_config_hook("./config/config.yaml")
#     for k, v in config.items():
#         parser.add_argument(f"--{k}", default=v, type=type(v))
#     args = parser.parse_args()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     if args.dataset == "CIFAR-10":
#         train_dataset = torchvision.datasets.CIFAR10(
#             root=args.dataset_dir,
#             train=True,
#             download=True,
#             transform=transform.Transforms(size=args.image_size).test_transform,
#         )
#         test_dataset = torchvision.datasets.CIFAR10(
#             root=args.dataset_dir,
#             train=False,
#             download=True,
#             transform=transform.Transforms(size=args.image_size).test_transform,
#         )
#         dataset = data.ConcatDataset([train_dataset, test_dataset])
#         class_num = 10
#     elif args.dataset == "CIFAR-100":
#         train_dataset = torchvision.datasets.CIFAR100(
#             root=args.dataset_dir,
#             download=True,
#             train=True,
#             transform=transform.Transforms(size=args.image_size).test_transform,
#         )
#         test_dataset = torchvision.datasets.CIFAR100(
#             root=args.dataset_dir,
#             download=True,
#             train=False,
#             transform=transform.Transforms(size=args.image_size).test_transform,
#         )
#         dataset = data.ConcatDataset([train_dataset, test_dataset])
#         class_num = 20
#     elif args.dataset == "STL-10":
#         train_dataset = torchvision.datasets.STL10(
#             root=args.dataset_dir,
#             split="train",
#             download=True,
#             transform=transform.Transforms(size=args.image_size).test_transform,
#         )
#         test_dataset = torchvision.datasets.STL10(
#             root=args.dataset_dir,
#             split="test",
#             download=True,
#             transform=transform.Transforms(size=args.image_size).test_transform,
#         )
#         dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
#         class_num = 10
#     elif args.dataset == "ImageNet-10":
#         dataset = torchvision.datasets.ImageFolder(
#             root='datasets/imagenet-10',
#             transform=transform.Transforms(size=args.image_size).test_transform,
#         )
#         class_num = 10
#     elif args.dataset == "ImageNet-dogs":
#         dataset = torchvision.datasets.ImageFolder(
#             root='datasets/imagenet-dogs',
#             transform=transform.Transforms(size=args.image_size).test_transform,
#         )
#         class_num = 15
#     elif args.dataset == "tiny-ImageNet":
#         dataset = torchvision.datasets.ImageFolder(
#             root='datasets/tiny-imagenet-200/train',
#             transform=transform.Transforms(size=args.image_size).test_transform,
#         )
#         class_num = 200
#     else:
#         raise NotImplementedError
#     data_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=500,
#         shuffle=False,
#         drop_last=False,
#         num_workers=args.workers,
#     )

#     res = resnet.get_resnet(args.resnet)
#     model = network.Network(res, args.feature_dim, class_num)
#     model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
#     model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
#     model.to(device)

#     print("### Creating features from model ###")
#     X, Y = inference(data_loader, model, device)
#     nmi, ari, f, acc = evaluation.evaluate(Y, X)
#     print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
