from errno import EFAULT
from os import rename
import numpy as np
import torch
from tqdm import tqdm
import dgl
from maintrain.utils.utils import Cluster
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
def prepoess_file_list(wsi, cluster_num):
    """根据输入的文件名列表构件数据字典,并为每一个文件创建一个唯一idx

    Args:
        wsi (_type_): 格式：[name, node_fea]
    returns:
        {idx: (name, (x, y), ndoe_fea, (x_true, y_true), label)}
        node_fea:[N, dim]
    """
    
    wsi_pos = [[int(kk[0].split("_")[3]), int(kk[0].split("_")[4]), kk[0], kk[1]] for kk in wsi]
    #wsi_pos格式1：[(x_true, y_true), name, node_fea]
    wsi_min_x = min([x[0] for x in wsi_pos])
    wsi_min_y = min([x[1] for x in wsi_pos])
    wsi_pos = [[(x[0]-wsi_min_x) // 512, (x[1]-wsi_min_y) // 512, x[2], x[3], (x[0], x[1]) ] for x in wsi_pos]
    #wsi_pos格式2：[(x, y), name, node_fea，(x_true, y_true)]
    ww = sorted(wsi_pos, key = lambda element: (element[0], element[1]))
    ww_dic = {}
    tensor_list = []
    for idx, w in enumerate(ww):
        ww_dic[idx] = [w[2], (w[0], w[1]), w[3], w[4]]
        tensor_list.append(w[3].unsqueeze(0))
    #将node-fea按照处理后的顺序变成tensor方便之后使用
    node_fea_tensor = torch.cat(tensor_list, dim = 0)
    # print(f"shape of node_fea{node_fea_tensor.size()}")
    #生成聚类标签
    clu_res = Cluster(node_fea=node_fea_tensor, cluster_num = cluster_num).predict()
    for i in range(len(clu_res)):
        ww_dic[i].append(clu_res[i])
    # print(f"sizeof node_fea{node_fea_tensor.size()}")
    return ww_dic, node_fea_tensor, clu_res


class new_graph:
    def __init__(self, wsi, clusterr_num) -> None:
        """根据node_fea和读取的文件名建图
         Args:
            wsi (_type_): 格式：[name, node_fea]
            cluster_num: 聚类的种类
        """
        self.wsi_dic, self.node_fea, self.clu_res = prepoess_file_list(wsi, clusterr_num)
        self.node_num = len(self.node_fea)
    def init_edge(self):
        """初始化边，参考javed
        """
        u = []
        v = []
        e_fea = []
        #仿照javed公式写(r,c,h,w)分别代表了 左上角的坐标和图像的高和宽
        h = w = 128
        for i in tqdm(range(self.node_num)): #TODO 优化建图的速度
            r_i, c_i = self.wsi_dic[i][3]
            for j in range(i+1, self.node_num):
                r_j, c_j = self.wsi_dic[j][3]
                p_i_j = torch.tensor([2 * (r_i - r_j) / (2*h), 2 * (c_i - c_j) / (2*h), 0, 0])
                temp = torch.tensor((self.node_fea[i]-self.node_fea[j]) ** 2)
                temp = torch.cat([temp, p_i_j], dim = 0)
                e_fea.append(temp.unsqueeze(0))
                u.append(i)
                v.append(j)
        e_fea = torch.cat(e_fea, dim = 0)
        return u, v, e_fea
    def init_graph(self):
        u, v, e_fea = self.init_edge()
        self.graph = dgl.graph((u, v))
        # self.graph.ndata['kk'] = self.node_fea
        # self.graph.edata['h'] = e_fea
        
        return self.graph, self.node_fea, self.clu_res, self.wsi_dic, (u, v), e_fea
    
                
if __name__ == '__main__':
    #模拟输入
    save_dict_path = 'C:/Users/86136/Desktop/new_gcn/wsi_dict.npy'
    wsi_dict = np.load(save_dict_path, allow_pickle='TRUE')
    wsi = wsi_dict.item()['43']
    wsi = [[w, torch.randn(1, 128)] for w in wsi ]
    wsi = wsi
    # a, b = prepoess_file_list(wsi, 6)
    # print(a[0])
    aa = new_graph(wsi, 6)
    aa.init_graph()
                