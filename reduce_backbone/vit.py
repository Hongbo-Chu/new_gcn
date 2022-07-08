import timm
import torch
import torch.nn as nn

"""
use timm.vit_base_patch8_224

architecture of vit
PatchEmbed
Droupout

encoder block * 12

layerNorm
Identity
Linear
"""

class vit(nn.Module):
    def __init__(self, block_num, pretrain = None):
        """
        params:
        block_num: num of the vit block, ranging from 0-11
        pretrain: the path of the .pth
        """
        super(vit, self).__init__()
        self.basemodel = timm.create_model( 'vit_base_patch8_224', pretrained=False)
        self.baselayers = list(self.basemodel.children())
        
        if pretrain is not None:#预训练模型采用无监督自行训练
            self.basemodel.load_state_dict(torch.load(self.pretrain))
        
        #开始精简模型
        # self.encoderblocks = [self.baselayers[2] for _ in range(block_num)]
        
        self.blocks = [
            self.baselayers[0], # patch
            self.baselayers[1], # dropout
            self.baselayers[2],
            self.baselayers[-3],
            # self.baselayers[-2],
            # self.baselayers[-1]
            
        ]
        self.model = nn.Sequential(*self.blocks)
    def forward(self, x):
        return self.model(x)

        
def buildvit():
    return vit(pretrain=None)


a = vit(1,pretrain=None)
b = a(torch.randn(1,3,224,224))
print(b)