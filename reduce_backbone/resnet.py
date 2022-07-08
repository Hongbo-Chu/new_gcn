from distutils.command.build import build
import timm
import torch
import torch.nn as nn

"""
use timm resnet34 (params = 21797672)
resnet24 architecture:
cov2d
BN
relu
MP
B0: 3 basic block
B1: 4 basic block
B2: 6 basic block
B3: 3 basic block
AAP
Linear
"""

class resnet(nn.Module):
    def __init__(self, pretrain = None):
        super(resnet, self).__init__()
        self.basemodel = timm.create_model( 'resnet34', pretrained=False)
        self.baselayers = list(self.basemodel.children())
        
        if pretrain is not None:#预训练模型采用无监督自行训练
            self.basemodel.load_state_dict(torch.load(self.pretrain))
            
        #开始精简模型  
        #模型block之前预处理的部分
        self.blocks = [
            self.baselayers[0], # conv
            self.baselayers[1], # BN
            self.baselayers[2], # relu
            self.baselayers[3],  # MP 
            self.baselayers[4],
            # self.baselayers[-2],
            # self.baselayers[-1]
        ]
        self.model = nn.Sequential(*self.blocks)
    def forward(self, x):
        return self.model(x)

def buildresnet():
    return resnet(pretrain = None) 

res = resnet()
image = torch.randn(2, 3, 224, 224)
aa = res(image)