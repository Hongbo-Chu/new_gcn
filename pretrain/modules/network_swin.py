import torch.nn as nn
import torch
import timm
from timm.models.swin_transformer import swin_base_patch4_window12_384
from functools import partial
from torch.nn.functional import normalize
def ConvBlocks(input, output, kernel_size=2, stride=1):
    return nn.Sequential(
        nn.Conv2d(input, output, kernel_size, stride),
        nn.BatchNorm2d(output)
    )
# class MSResNet(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(MSResNet, self).__init__()
#         self.conv1 = ConvBlocks()
#         self.conv2 = ConvBlocks
#         self.conv3 = ConvBlocks
#         self.conv4 = ConvBlocks
#         self.conv5 = ConvBlocks
#         self.linear = nn.Linear()
#         self.BN = nn.BatchNorm2d()
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2)
#         x4 = self.conv4(x3)
#         x5 = self.conv5(x4)
#         out = self.linear(x2)

class swinTrans(timm.models.swin_transformer.SwinTransformer):
    """
    my own swin trnasformer
    """
    def forward(self, x):
        x = self.forward_features(x)
        return x


def swin_base_patch4_window12_224_my(**kwargs):
    """ Swin-B @ 384x384, pretrained ImageNet-22k, fine tune 1k
    """
    model = swinTrans()
    return model


class Network(nn.Module):
    def __init__(self, model, feature_dim, class_num):
        super(Network, self).__init__()
        self.swin = swin_base_patch4_window12_224_my()
        
        self.aap = torch.nn.AdaptiveAvgPool2d(224)
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.vit_output_dim = 768
        self.instance_projector = nn.Sequential(
            nn.Linear(self.vit_output_dim, self.vit_output_dim),
            nn.ReLU(),
            nn.Linear(self.vit_output_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.vit_output_dim, self.vit_output_dim),
            nn.ReLU(),
            nn.Linear(self.vit_output_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        x_i = self.aap(x_i)
        x_j = self.aap(x_j)
        h_i = self.swin(x_i)
        # print(f"swin.shape:{h_i.size()}")
        h_j = self.swin(x_j)
        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        x = self.aap(x)
        h = self.swin(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
    
    def forward_instance(self, x):
        x = self.aap(x)
        h = self.swin(x)
        c = self.instance_projector(h)
        return c
