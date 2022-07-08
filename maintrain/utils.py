from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from torch import tensor
import torch



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
    

def euclidean_dist(x, y):
    """
    a fast approach to compute X, Y distace
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """

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
    return dist

    
    

def addMask():
    """
    choose node to mask
    """
