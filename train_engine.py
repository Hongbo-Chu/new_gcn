import torch
import argparse
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


from maintrain.construct_graph import new_graph
from maintrain.utils.utils import chooseNodeMask, compute_pys_feature, fea2pos, chooseEdgeMask
from maintrain.models.gcn import GCN
from maintrain.models.loss import my_loss
"""
目前图神经和backbone训练比1:10
对一张wsi加n次mask
"""

def train_one_epoch(backbone: torch.nn.Module, gcn: torch.nn.Module, 
                    optimizer:torch.optim.Optimizer, 
                    args=None):
    backbone.train()
    gcn.train()
    optimizer.zero_grad()
    
    #for data in dataloader
    #一张wsi
    input_image = torch.randn(30,3,224,224).to(args.device)
    node_fea = backbone(input_image)
    node_fea_detach = node_fea.detach()
    #这一步仅仅适用于建图用的，不参与传播，所以要detach
    save_dict_path = 'C:/Users/86136/Desktop/new_gcn/wsi_dict.npy'
    wsi_dict = np.load(save_dict_path, allow_pickle='TRUE')
    wsi = wsi_dict.item()['43'][:30]
    
    wsi = [[w, node_fea_detach[i]] for i, w in enumerate(wsi) ]
    #建图
    g, node_fea, clu_label, wsi_dic_new, u_v_pair, edge_fea = new_graph(wsi, args.cluster_num, args).init_graph()
    g = g.to(args.device)
    #下面该挑mask了
    no_mask_lsit = []#用于存储不要mask的列表
    mask_rates = [args.mask_rate_high, args.mask_rate_mid, args.mask_rate_low]#各个被mask的比例
    for i in range(args.mask_num):
        mask_idx, fea_center, fea_edge, sort_idx_rst, cluster_center_fea = chooseNodeMask(node_fea, args.cluster_num, mask_rates, wsi_dic_new, args.device)#TODO 检查数量
        mask_edge_idx = chooseEdgeMask(u_v_pair, clu_label,[], {"inter":0.1} )
        print(f"mask nodes{mask_idx}")
        #添加mask
        node_fea[mask_idx] = 0
        edge_fea[mask_edge_idx] = 0
        #预测点
        predict_nodes = gcn(g, node_fea, edge_fea)
        loss = my_loss(predict_nodes, clu_label, cluster_center_fea, mask_idx, 0.5, sort_idx_rst)
        pys_center, pys_edge = compute_pys_feature(wsi_dic_new, args.pos_choose) #计算物理特征
        fea2pos(fea_center, fea_edge, pys_center, pys_edge)#统计对齐信息并打印
        loss.backward()
        optimizer.step()
        #mask比例下降
        for rate in mask_rates:
           rate = rate *  (1 - args.mask_decay_rate)
        #在下一轮的choosenodoemask的过程中会自动更新聚类中心，所以就不用再算一遍了
        
        

    
    
    #TODO loader
    #TODO 连GCn的时候将backbone的梯度先冻上