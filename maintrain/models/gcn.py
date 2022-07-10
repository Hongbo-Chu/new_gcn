import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")

def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return None


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias


    
class GCN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False
                 ):
        super(GCN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        
        if num_layers == 1:
            self.gcn_layers.append(GraphConv(
                in_dim, out_dim, residual=last_residual, norm=last_norm, activation=last_activation))
        else:
            # input projection (no residual)
            self.gcn_layers.append(GraphConv(
                in_dim, num_hidden, residual=residual, norm=norm, activation=create_activation(activation)))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gcn_layers.append(GraphConv(
                    num_hidden, num_hidden, residual=residual, norm=norm, activation=create_activation(activation)))
            # output projection
            self.gcn_layers.append(GraphConv(
                num_hidden, out_dim, residual=last_residual, activation=last_activation, norm=last_norm))

        # if norm is not None:
        #     self.norms = nn.ModuleList([
        #         norm(num_hidden)
        #         for _ in range(num_layers - 1)
        #     ])
        #     if not encoding:
        #         self.norms.append(norm(out_dim))
        # else:
        #     self.norms = None
        self.norms = None
        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcn_layers[l](g, h)
            if self.norms is not None and l != self.num_layers - 1:
                h = self.norms[l](h)
            hidden_list.append(h)
        # output projection
        if self.norms is not None and len(self.norms) == self.num_layers:
            h = self.norms[-1](h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)

# mainly copy from DGL with
class GraphConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 norm=None,
                 activation=None,
                 residual=True,
                 ):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim

        self.fc = nn.Linear(in_dim, out_dim)

        if residual:
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._in_feats, self._out_feats, bias=False)
                print("! Linear Residual !")
            else:
                print("Identity Residual ")
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        # if norm == "batchnorm":
        #     self.norm = nn.BatchNorm1d(out_dim)
        # elif norm == "layernorm":
        #     self.norm = nn.LayerNorm(out_dim)
        # else:
        #     self.norm = None
        
        self.norm = norm
        if norm is not None:
            self.norm = norm(out_dim)
        self._activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, graph, feat):
        with graph.local_scope():
            aggregate_fn = fn.copy_src('h', 'm') # 和copy_u一样，将源节点的特征存储到边上
            # if edge_weight is not None:
            #     assert edge_weight.shape[0] == graph.number_of_edges()
            #     graph.edata['_edge_weight'] = edge_weight
            #     aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')
            #如果有边的特征的话就用源节点乘边的权重，然后把得到的值作为消息，存储在边上  

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)#用于支持二分图
            # if self._norm in ['left', 'both']:
            degs = graph.out_degrees().float().clamp(min=1)#计算节点的出度矩阵
            norm = torch.pow(degs, -0.5) #计算C_ij,归一化用
            shp = norm.shape + (1,) * (feat_src.dim() - 1)#resahpe
            norm = torch.reshape(norm, shp)
            feat_src = feat_src * norm

            
            '''
            原DGL判断输入和输出的维度大小来判断是先乘权重矩阵再聚合，还是先聚合再乘权重矩阵，以加速运算。
            '''
            
            # if self._in_feats > self._out_feats:
            #     # mult W first to reduce the feature size for aggregation.
            #     # if weight is not None:
            #         # feat_src = th.matmul(feat_src, weight)
            #     graph.srcdata['h'] = feat_src
            #     graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            #     rst = graph.dstdata['h']
            # else:
            # aggregate first then mult W
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            
            #edge_pre
            
            # graph.apply_edges()
            rst = graph.dstdata['h']
            
            rst = self.fc(rst)
            # 更新函数
            # if self._norm in ['right', 'both']:
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)

            if self.norm is not None:
                rst = self.norm(rst)

            if self._activation is not None:
                rst = self._activation(rst)

            return rst
        
        
        
        
        
        
        
        
        
        
        
#---------------------------------------------------------test----------------------------------------------------------------
if __name__ == '__main__':
    import dgl
    import dgl.function as fn
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from dgl import DGLGraph
    class constructGraph:
        def __init__(self, node_feature, pos_feature:list, edge_feature):
            """
            node_feature.size() = [num_nodes, feature_dim]
            pos_feature.size() = [num_nodes, 2]
            
            """
            if isinstance(node_feature, list):
                self.node_feature = torch.tensor(node_feature)
            
            self.node_feature = node_feature
            self.pos_feature = pos_feature
            self.node_num = len(self.node_feature)
            self.node_name = [i for i in range(self.node_num)] #一个position对应一个name
            self.edge_index = []
            self.edge_feature = edge_feature
        def pos_encoding(self, pos:list):
            pass 
        def inital_edge_feature(self):
            '''
            
            '''
                
        def inital_graph(self, threshold):
            #inital_edge_index
            for i in range(self.node_num):
                for j in range(i+1, self.node_num):
                    pos_i = self.pos_feature[i]
                    pos_j = self.pos_feature[j]
                    dis = ((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)**0.5
                    if dis < threshold:
                        self.edge_index.append([self.node_name[i], self.node_name[j]])
                        self.edge_index.append([self.node_name[j], self.node_name[i]])
            
            self.edge_index = torch.tensor(self.edge_index)
            
            #construct graph
            # print(self.edge_index)
            u = self.edge_index.permute(1, 0)[0]
            v = self.edge_index.permute(1, 0)[1]
            self.graph = dgl.graph((u, v))
            
            # #add nodefeature & eade feature
            self.graph.ndata['kk'] = self.node_feature
            self.graph.edata['h'] = self.edge_feature #TODO 初始化边的信息
            

    node_feature = torch.randn(400, 128).requires_grad_()
    edge_feature = torch.randn(2964, 128).requires_grad_()
    pos_code = []
    for i in range(20):
        for j in range(20):
            pos_code.append([i, j])
    a = constructGraph(node_feature=node_feature, pos_feature=pos_code, edge_feature=edge_feature)
    a.inital_graph(2)
    g = a.graph
    print(g)

    print(g.edges())
    
            
    #模拟backbone输入
    import torch
    node_fea = torch.randn(400, 128)
    a = GCN(in_dim=128, num_hidden=3, out_dim=128, num_layers=3, dropout=0,activation="prelu", residual=True,norm=nn.LayerNorm)
    b = a(g,node_fea)
    print(b.size())