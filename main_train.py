import torch
import argparse
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

from maintrain.models.gcn import GCN
from train_engine import train_one_epoch
from reduce_backbone import build_model


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch implementation')
    # 
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    # 
    parser.add_argument('--backbone', type=str, default='vit',
                    help='backbonetype')
    parser.add_argument('--lr', type=float, default=1e-5,
                    help='backbonetype')
    parser.add_argument('--decay', type=float, default=0.1,
                    help='backbonetype')
    
    parser.add_argument('--cluster_num', type=int, default=6,
                help='backbonetype')
    
    parser.add_argument('--mask_decay_rate', type=float, default=0.1,
                help='backbonetype')
    
    parser.add_argument('--pos_choose', type=int, default=1,
                help='用于表示找物理特征的时候看周围几圈点')
    
    parser.add_argument('--mask_num', type=int, default=10,
                help='一个wsi加几轮mask')
    parser.add_argument('--mask_rate_high', type=float, default=0.1,
                help='一个wsi加几轮mask')
    parser.add_argument('--mask_rate_mid', type=float, default=0.1,
                help='一个wsi加几轮mask')
    parser.add_argument('--mask_rate_low', type=float, default=0.1,
                help='一个wsi加几轮mask')
    parser.add_argument('--embeding_dim', type=int, default=768,
                help='一个wsi加几轮mask')

    return parser
        

def main(args):
    #设置各种参数
    backboneModel = build_model(args.backbone).to(args.device)
    graph_model = GCN(in_dim=args.embeding_dim, num_hidden=512, out_dim=args.embeding_dim, num_layers=3, dropout=0,activation="prelu", residual=True,norm=nn.LayerNorm).to(args.device)
    optimizer = optim.Adam(backboneModel.parameters(), lr=args.lr, weight_decay=args.decay)
    train_one_epoch(backboneModel, graph_model, optimizer, args)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)