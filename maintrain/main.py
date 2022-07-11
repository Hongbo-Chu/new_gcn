
import torch
import argparse
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



from construct_graph import new_graph
from utils import chooseNodeMask, compute_pys_feature, fea2pos
from models.gcn import GCN
from models.loss import inner_cluster_loss



def train(): #TODO 还未添加backbone
    #首先加载数据
    #假装通过backbone来的
    cluster_num = 6
    
    save_dict_path = 'C:/Users/86136/Desktop/new_gcn/wsi_dict.npy'
    wsi_dict = np.load(save_dict_path, allow_pickle='TRUE')
    wsi = wsi_dict.item()['43']
    wsi = [[w, torch.randn(1, 128)] for w in wsi ]
    wsi = wsi[0:30]
    # a, b = prepoess_file_list(wsi, 6)
    # print(a[0])
    g, node_fea, clu_label, wsi_dic_new = new_graph(wsi, cluster_num).init_graph()
    #下面该挑mask了
    mask_idx, fea_center, fea_edge = chooseNodeMask(node_fea, cluster_num, [0.01, 0.05, 0.1])#TODO 检查数量
    print(f"mask nodes{mask_idx}")
    #添加mask
    node_fea[mask_idx] = 0
    #预测点
    grapg_model = GCN(in_dim=128, num_hidden=3, out_dim=128, num_layers=3, dropout=0,activation="prelu", residual=True,norm=nn.LayerNorm)
    predict_nodes = grapg_model(g, node_fea)
    center_fea = [torch.randn(1,128) for i in range(cluster_num)]
    loss = inner_cluster_loss(node_fea, clu_label, center_fea, mask_idx, 0.5)
    pys_center, pys_edge = compute_pys_feature(wsi_dic_new, 1)    
    fea2pos(fea_center, fea_edge, pys_center, pys_edge)
    #TODO mask比例递减
    # loss.backward()
    # print(loss)
    # # print(predict_nodes.size())
    
    
    
def main():
    # parser = argparse.ArgumentParser(description='PyTorch implementation')
    # parser.add_argument('--device', type=int, default=0,
    #                     help='which gpu to use if any (default: 0)')
    # parser.add_argument('--batch_size', type=int, default=32,
    #                     help='input batch size for training (default: 32)')
    train()

if __name__ == '__main__':
    main()