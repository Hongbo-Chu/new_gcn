from platform import node
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from torch import tensor
import torch
import time



class Cluster:
    """
    based on sklearn: https://scikit-learn.org/stable/modules/clustering.html
    """
    def __init__(self, node_fea, cluster_num, method = "K-means", **kwargs) -> None:
        """
        inputs: node_features
        
        output: label of each node
            shape: 1 x N (a list of cluster)
        params:
        node_fea: input noode feature, shape -> num_nodes x feature_dims
        method: cluster methods
        cluster_num: number of clusters
        **kwargs: oarameters of spatracl cluster
        """
        
        # our support cluster methods
        self.clusters= {'K-means':self.K_means, 'spectral':self.spectral, 'Affinity':self.Affinity}
        assert method in self.clusters.keys(), "only support K-means, spectral, Affinity"
        self.cluster_num = cluster_num
        self.methods = method
        self.node_fea = node_fea
        
    def predict(self):
        result = self.clusters[self.methods]()
        return result
        
    def K_means(self):
        print("use cluster-method: K-means")
        self.model = KMeans(self.cluster_num)
        pre_label = self.model.fit_predict(self.node_fea)
        return pre_label
    
    def spectral(self):
        Scluster = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',n_neighbors=10)# TODO 补充谱聚类参数
        
        return  Scluster.fit_predict(self.node_fea)# TODO 验证输入格式
    
    def  Affinity(self):
        pass
    

def euclidean_dist(x, center):
    """
    a fast approach to compute X, Y euclidean distace.
    Args:
        x: pytorch Variable, with shape [m, d]
        center: pytorch Variable, with shape [1, d]
    Returns:
        dist: pytorch Variable, with shape [1, m]
    """
    temp = [center for _ in range(x.size(0))]
    y = torch.cat(temp, dim=0)
    #y: pytorch Variable, with shape [n, d]
    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT 
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    res = []
    for i in range(m):
        res.append(dist[i][0]) 
    return res

def samesort(list1, list2):
    """
    用list1的排序方法排序list2
    returns:
        list1, list2
    """
    return zip(*sorted(zip(list1, list2))) 
    

def split2clusters(node_fea, cluster_num, cluster_method = "K-means"):
    """
    split nodes into different clusters
    """
    node_fea_list = [[] for _ in range(cluster_num)] # 用于存放每一类cluster的node_fea(tensor)
    node_idx_list = [[] for _ in range(cluster_num)] # 用于存放每一类cluster的标签(int)
    cluster_res = Cluster(node_fea=node_fea, cluster_num=cluster_num, method=cluster_method).predict()
    #按照聚类标签分类
    for idx, clu in enumerate(cluster_res):
        node_fea_list[clu].append(node_fea[idx].unsqueeze(0))
        node_idx_list[clu].append(idx)
    return node_fea_list, node_idx_list

def chooseMask(node_fea, cluster_num, mask_rate:list, cluster_method = "K-means"):
    """
    choose which nodes to mask of a certain cluster
    args:
        maskrate: list,用于存放三种相似度的mask比例。[高，中，低]
    return: 
        被masknode的idx，list of nodes
    """
    mask_node_idx = [] # 用于存储最终mask点的idx
    node_fea_list, node_idx_list = split2clusters(node_fea, cluster_num, cluster_method)
    
    #对每一类点分别取mask
    for feats, idxs in zip(node_fea_list, node_idx_list):
        #feats的格式是[tensor,tessor....],先要拼成一个tensor
        feats = torch.cat(feats, dim = 0)
        # print(f"feat{feats.size()}")
        cluster_center = feats.mean(dim=0)
        #计算任一点和中心的欧氏距离
        # print(f"center:{cluster_center.size()}")
        dist = euclidean_dist(feats, cluster_center.unsqueeze(0))
        sorted_disrt, sorted_idex = samesort(dist, idxs)
        #对于index取不同位置的点进行mask
        for i, rate in enumerate(mask_rate):
            mask_num = int(len(sorted_idex) * rate)
            if i == 0:#高相似度
                mask_node_idx.extend(sorted_idex[:mask_num])
            elif i == 2:#地相似度
                mask_node_idx.extend(sorted_idex[-mask_num:])
            else: # 中相似度
                mid = len(sorted_idex) // 2
                mid_pre = mid - (mask_num) // 2
                mask_node_idx.extend(sorted_idex[mid_pre:mid_pre + mask_num])
    return mask_node_idx

def mask(node_fea, cluster_nun):
    """
    input:node
    """
    node_idx = torch.range(0, node_fea.size(0) - 1)
    clusters = split2clusters()
    for clu in clusters:
        chooseMask()



if __name__  == '__main__':
    """
    模拟backbone的输入： 600 * 128
    """
    node_fea = torch.randn(3000, 128)
    a = chooseMask(node_fea=node_fea,cluster_num=6, mask_rate=[0.1, 0.1, 0.1])
    print(len(a))