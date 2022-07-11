import torch
import sys
sys.path.append('../maintrain')
from maintrain.utils.utils import Cluster
import torch.nn.functional as F

def inner_cluster_loss(node_fea, clu_label, center_fea, mask_nodes, mask_weight):
    """用于计算类内loss
        对于更新后的node_fea(N, dim)，分别计算每个node_fea和聚类中心的L2距离
    Args:
        node_fea (tensor): 更新后的node_fea，require_grade=True
        clu_label (_type_): 每个点的聚类标签
        center_fea (_type_): 几个聚类中心的向量,list of tensor的形式
        mask_nodes:加了mask的点
        mask_weight:对于mask点的聚类权重
    """
    #TODO用矩阵的方式优化
    L2_dist = 0
    for i in range(len(clu_label)):
        L2_dist += F.pairwise_distance(node_fea[i], center_fea[clu_label[i]], p=2)
        if  i in mask_nodes:
            L2_dist += (1 + mask_weight) * F.pairwise_distance(node_fea[i], center_fea[clu_label[i]], p=2)
    return L2_dist

def inter_cluster_loss(node_fea, clu_label, center_fea, mask_nodes, mask_weight):
    """类间loss，先找到每一类的聚类边界，然后最大化边界之间的距离

    Args:
        node_fea (_type_): _description_
        clu_label (_type_): _description_
        center_fea (_type_): _description_
        mask_nodes (_type_): _description_
        mask_weight (_type_): _description_
    """
    #首先找到每一类的聚类半径
    #算出每一类的

def loss():
    pass

if __name__ == '__main__':
    node_fea = torch.randn(3000, 128)
    clu_label = Cluster(node_fea, 6).predict()
    center_fea = [torch.randn(1,128) for i in range(6)]
    loss = inner_cluster_loss(node_fea, clu_label, center_fea,1,1)