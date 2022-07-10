from errno import EFAULT
from os import rename
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
import dgl
from utils import Cluster
"""
从backbone来的数据：
1. nodefeature:
    shape[node_num, feature_dim]
2. positionfeature:
    shape:[node_num, [pos_x, pos_y]]
构建：
1. edge_feature
    shape[node_num^2, edge_dim]
    
2. adjacent matric

"""
    
class constructGraph:
    def __init__(self, node_feature, pos_feature:list):
        """
        node_feature.size() = [num_nodes, feature_dim]
        pos_feature.size() = [num_nodes, 2]
        
        """
        if isinstance(node_feature, list):
            self.node_feature = torch.tensor(node_feature)
        
        self.node_feature = node_feature
        self.pos_feature = pos_feature
        self.node_num = len(self.pos_feature)
        self.node_name = [i for i in range(self.node_num)] #一个position对应一个name
        self.edge_index = []
        # self.edge_feature = edge_feature
        
        
    def pos_encoding(self, pos:list):
        pass 
    def inital_edge_feature(self):
        self.u, self.v = self.graph.edges()[0], self.graph.edges()[1]
        edge_feature_list = []
        for i in range(len(self.u)):
            node_fea_u = self.node_feature[self.u[i]]
            node_fea_v = self.node_feature[self.v[i]]
            edge_feature = abs(node_fea_u - node_fea_v)
            edge_feature_list.append(edge_feature.detach().numpy().tolist())

        edge_feature_teosr = torch.tensor(edge_feature_list)
        print(f"edge_feature.size(){edge_feature_teosr.size()}")
        self.graph.edata['h'] = edge_feature_teosr
        return self.graph
            
    def inital_graph(self, threshold):
        #inital_edge_index
        print("computing graphs")
        for i in tqdm(range(self.node_num)): #TODO 优化建图的速度
            for j in range(i+1, self.node_num):
                edge = torch.cat([(self.node_feature[i]-self.node_feature[j])**2, self.pos_feature[i], self.pos_feature[j]])#TODO add p_ij as javed
        
        self.edge_index = torch.tensor(self.edge_index)
        
        #construct graph
        u = self.edge_index.permute(1, 0)[0]
        v = self.edge_index.permute(1, 0)[1]
        self.graph = dgl.graph((u, v))
        
        # #add nodefeature & eade feature
        self.graph.ndata['kk'] = self.node_feature
        # self.graph.edata['h'] = self.edge_feature #TODO 初始化边的信息
        return self.graph


def prepoess_file_list(wsi, cluster_num):
    """根据输入的文件名列表构件数据字典,并为每一个文件创建一个唯一idx

    Args:
        wsi (_type_): 格式：[name, node_fea]
    returns:
        {idx: (name, (x, y), ndoe_fea, label)}
        node_fea:[N, dim]
    """
    
    wsi_pos = [[int(kk[0].split("_")[3]), int(kk[0].split("_")[4]), kk[0], kk[1]] for kk in wsi]
    wsi_min_x = min([x[0] for x in wsi_pos])
    wsi_min_y = min([x[1] for x in wsi_pos])
    wsi_pos = [[(x[0]-wsi_min_x) // 512, (x[1]-wsi_min_y) // 512, x[2], x[3] ] for x in wsi_pos]
    ww = sorted(wsi_pos, key = lambda element: (element[0], element[1]))
    ww_dic = {}
    tensor_list = []
    for idx, w in enumerate(ww):
        ww_dic[idx] = [w[2], (w[0], w[1]), w[3]]
        tensor_list.append(w[3])
    #将node-fea按照处理后的顺序变成tensor方便之后使用
    node_fea_tensor = torch.cat(tensor_list, dim = 0)
    # print(f"shape of node_fea{node_fea_tensor.size()}")
    #生成聚类标签
    clu_res = Cluster(node_fea=node_fea_tensor, cluster_num = cluster_num).predict()
    for i in range(len(clu_res)):
        ww_dic[i].append(clu_res[i])
    return ww_dic, node_fea_tensor


class new_graph:
    def __init__(self, wsi, clusterr_num) -> None:
        """根据node_fea和读取的文件名建图
         Args:
            wsi (_type_): 格式：[name, node_fea]
            cluster_num: 聚类的种类
        """
        self.wsi_dic, self.node_fea = prepoess_file_list(wsi, clusterr_num)
        self.node_num = len(self.node_fea)
    def init_edge(self):
        """初始化边，参考javed
        """
        u = []
        v = []
        e_fea = []
        for i in tqdm(range(self.node_num)): #TODO 优化建图的速度
            for j in range(i+1, self.node_num):
                e_fea.append(torch.cat([(self.node_feature[i]-self.node_feature[j])**2, self.wsi_dic[i][], self.pos_feature[j]]))#TODO add p_ij as javed
                u.append(i)
                v.append(j)
        return u, v, e_fea
    def init_graph(self):
        u, v, e_fea = self.init_edge()
        self.graph = dgl.graph((u, v))
        self.graph.ndata['kk'] = self.node_feature
        self.graph.edata['h'] = e_fea
    
                
if __name__ == '__main__':
    #模拟输入
    save_dict_path = 'C:/Users/86136/Desktop/new_gcn/wsi_dict.npy'
    wsi_dict = np.load(save_dict_path, allow_pickle='TRUE')
    wsi = wsi_dict.item()['43']
    wsi = [(w, torch.randn(1, 128)) for w in wsi ]
    a, b = prepoess_file_list(wsi, 6)
    print(a[0])
    
                