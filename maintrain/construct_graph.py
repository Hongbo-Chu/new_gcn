from os import rename
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
import dgl
"""
从mae来的数据：
1. nodefeature:
    shape[node_num, feature_dim]
2. positionfeature:
    shape:[node_num, [pos_x, pos_y]]
构建：
1. edge_feature
    shape[node_num^2, edge_dim]
    
2. adjacent matric

"""

#不一样大小的图怎么解决
# class constructGraph:
#     def __init__(self, node_feature:list, pos_feature:list):
#         """
#         node_feature.size() = [num_nodes, feature_dim]
#         pos_feature.size() = [num_nodes, 2]
        
#         """
#         if isinstance(node_feature, list):
#             self.node_feature = torch.tensor(node_feature)
        
#         self.node_feature = node_feature
#         self.pos_feature = pos_feature
#         self.node_num = len(self.node_feature)
#         self.edge_index = []
#     def pos_encoding(self, pos:list):
#         pass 
#     def inital_edge_feature(self):
#         '''
        
#         '''
            
#     def inital_graph(self, threshold):
#         #inital_edge_index
#         for i in range(self.node_num):
#             for j in range(i+1, self.node_num):
#                 pos_i = self.pos_feature[i]
#                 pos_j = self.pos_feature[j]
#                 dis = ((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)**0.5
#                 if dis < threshold:
#                     self.edge_index.append([pos_i, pos_j])
#                     self.edge_index.append([pos_j, pos_i])
        
#         self.edge_index = torch.tensor(self.edge_index)
        
#         #construct graph
#         u = self.edge_index.permute(1, 0)[0]
#         v = self.edge_index.permute(1, 0)[1]
#         self.graph = dgl.graph((u, v))
        
#         #add nodefeature & eade feature
#         self.graph.ndata['node'] = self.node_feature
#         self.graph.edata['edge'] = self.edge_feature #TODO 初始化边的信息
    
    
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
                pos_i = self.pos_feature[i]
                pos_j = self.pos_feature[j]
                dis = ((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)**0.5
                if dis < threshold:
                    self.edge_index.append([self.node_name[i], self.node_name[j]])
                    self.edge_index.append([self.node_name[j], self.node_name[i]])
        
        self.edge_index = torch.tensor(self.edge_index)
        
        #construct graph
        u = self.edge_index.permute(1, 0)[0]
        v = self.edge_index.permute(1, 0)[1]
        self.graph = dgl.graph((u, v))
        
        # #add nodefeature & eade feature
        self.graph.ndata['kk'] = self.node_feature
        # self.graph.edata['h'] = self.edge_feature #TODO 初始化边的信息
        return self.graph

        
                
                
                