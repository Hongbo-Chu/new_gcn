import torch
import argparse
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from yaml import parse


from maintrain.construct_graph import new_graph
from maintrain.utils.utils import chooseNodeMask, compute_pys_feature, fea2pos, chooseEdgeMask
from maintrain.models.gcn import GCN
from train_engine import train_one_epoch
from reduce_backbone import build_model


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch implementation')
    # 
    parser.add_argument('--device', type=str, default="cpu",
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



# def train(backbone, optimizer, args): 
#     #首先加载数据
#     input_image = torch.randn(30,3,224,224).to(args.device)
#     backbone.train()
#     node_fea = backbone(input_image)
#     node_fea_de = node_fea.detach()
#     save_dict_path = 'C:/Users/86136/Desktop/new_gcn/wsi_dict.npy'
#     wsi_dict = np.load(save_dict_path, allow_pickle='TRUE')
#     wsi = wsi_dict.item()['43'][:30]
#     print(f"wsi的大小{node_fea.size()}")
#     wsi = [[w, node_fea_de[i-1]] for i, w in enumerate(wsi) ]
#     wsi = wsi[0:30]
#     # a, b = prepoess_file_list(wsi, 6)
#     # print(a[0])
#     #TODO将node_fea和wsi解耦合
#     g, node_fea, clu_label, wsi_dic_new, u_v_pair,edge_fea = new_graph(wsi, args.cluster_num).init_graph()
#     #下面该挑mask了
#     mask_idx, fea_center, fea_edge, sort_idx_rst = chooseNodeMask(node_fea, args.cluster_num, [0.01, 0.05, 0.1], wsi_dic_new)#TODO 检查数量
#     mask_edge_pair = chooseEdgeMask(u_v_pair, clu_label,[], {"inter":0.1} )
#     print(f"mask nodes{mask_idx}")
#     #添加mask
#     node_fea[mask_idx] = 0
#     #预测点
#     grapg_model = GCN(in_dim=128, num_hidden=256, out_dim=128, num_layers=3, dropout=0,activation="prelu", residual=True,norm=nn.LayerNorm)
#     predict_nodes = grapg_model(g, node_fea, edge_fea)
#     center_fea = [torch.randn(1,128) for i in range(args.cluster_num)]
#     loss = inner_cluster_loss(predict_nodes, clu_label, center_fea, mask_idx, 0.5)
#     pys_center, pys_edge = compute_pys_feature(wsi_dic_new, 1)    
#     fea2pos(fea_center, fea_edge, pys_center, pys_edge)
#     loss.backward()
#     print(loss)
#     # # print(predict_nodes.size())
        


def main(args):
    #设置各种参数
    backboneModel = build_model(args.backbone).to(args.device)
    graph_model = grapg_model = GCN(in_dim=args.embeding_dim, num_hidden=512, out_dim=args.embeding_dim, num_layers=3, dropout=0,activation="prelu", residual=True,norm=nn.LayerNorm).to(args.device)
    optimizer = optim.Adam(backboneModel.parameters(), lr=args.lr, weight_decay=args.decay)
    train_one_epoch(backboneModel, graph_model, optimizer, args)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)