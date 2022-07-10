import torch
import sys
sys.path.append('../maintrain')
from utils import Cluster


def inner_cluster_loss(node_fea, clu_label, center_fea, mask_nodes, mask_weight):
    """用于计算类内loss
        对于更新后的node_fea(N, dim)，生成对应的中心向量矩阵。
    Args:
        node_fea (tensor): 更新后的node_fea，require_grade=True
        clu_label (_type_): 每个点的聚类标签
        center_fea (_type_): 几个聚类中心的向量
        mask_nodes:加了mask的点
        mask_weight:对于mask点的聚类权重
    """
    
    optim_matrix = []#由各种中心向量组成，是优化的目标
    for i in range(len(clu_label)):
        if i in mask_nodes:
            optim_matrix.append((1+mask_weight) * center_fea[clu_label[i]])
        optim_matrix.append(center_fea[clu_label[i]])
    optim_matrix = torch.cat(optim_matrix, dim = 0)
    loss = node_fea @ optim_matrix.transpose(1, 0)
    
    return loss

def inter_cluster_loss():
    pass

def loss():
    pass

if __name__ == '__main__':
    node_fea = torch.randn(3000, 128)
    clu_label = Cluster(node_fea, 6).predict()
    center_fea = [torch.randn(1,128) for i in range(6)]
    loss = inner_cluster_loss(node_fea, clu_label, center_fea,1,1)