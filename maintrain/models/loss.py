from platform import node
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
        L2_dist += F.pairwise_distance(node_fea[i].unsqueeze(0), center_fea[clu_label[i]].unsqueeze(0), p=2)
        if  i in mask_nodes:
            L2_dist += (1 + mask_weight) * F.pairwise_distance(node_fea[i].unsqueeze(0), center_fea[clu_label[i]].unsqueeze(0), p=2)
    return L2_dist

def inter_cluster_loss(node_fea, sort_idx_rst, mask_nodes, mask_weight):
    """类间loss，先找到每一类的聚类边界，然后最大化边界之间的距离


    Args:
        node_fea (_type_): 节点特征
        clu_label (_type_): 不需要了现在，都包含在sort_idx_rst中了。
        center_fea (_type_): 每一类的聚类中心
        sort_idx_rst (_type_):每类的相似度排序结果[[],[],[]....]
        mask_nodes (_type_): 加了mask的点
        mask_weight (_type_): 对于mask点的聚类权重

    Returns:
        _type_: _description_
    """
    #首先找到每一类的边缘点的idx
    edge_nodes = []
    for sort_idx in sort_idx_rst:
        edge_nodes.append(sort_idx[-1])
    #计算这些点之间的距离
    L2_dist = 0
    for i in range(len(edge_nodes)):
        for j in range(i+1, len(edge_nodes)):
            L2_dist += F.pairwise_distance(node_fea[edge_nodes[i]].unsqueeze(0), node_fea[edge_nodes[j]].unsqueeze(0), p=2)
            # if  i in mask_nodes:
            #     L2_dist += (1 + mask_weight) * F.pairwise_distance(node_fea[i].unsqueeze(0), center_fea[clu_label[i]].unsqueeze(0), p=2)
    return -1 * L2_dist #将增大变为减小

def my_loss(node_fea, clu_label, center_fea, mask_nodes, mask_weight, sort_idx_rst):
    inner_loss = inner_cluster_loss(node_fea, clu_label, center_fea, mask_nodes, mask_weight)
    inter_loss = inter_cluster_loss(node_fea, sort_idx_rst, mask_nodes, mask_weight)
    final_loss = inner_loss + inter_loss
    return  final_loss
# if __name__ == '__main__':
#     node_fea = torch.randn(3000, 128)
#     clu_label = Cluster(node_fea, 6).predict()
#     center_fea = [torch.randn(1,128) for i in range(6)]
#      final_loss = inner_cluster_loss(node_fea, clu_label, center_fea,1,1)