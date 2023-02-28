import torch
import torch.nn as nn
from torchvision.transforms import Lambda

""" NNVC official SW - NN-based post-filter by Qualcomm (in-loop filter) [JVET-W0131] & Nokia (Content-adaptive post-filter) [JVET-AC0055] """

class Multiplier(nn.Module):
    def __init__(self):
        super(Multiplier, self).__init__()
    
    def _multiplier(self, shape, require_grad=True):
        return nn.parameter.Parameter(torch.ones(size=shape, dtype=torch.float32), requires_grad=require_grad)
    
    def forward(self, x):
        x_shape = x.shape
        multiplier = self._multiplier(shape=(1, x_shape[1], 1, 1)).to(x.device)
        
        return x * multiplier

class MulBlock(nn.Module): # 18432 for 32 channel
    def __init__(self, num_channel=24, inner_channel=72):
        super(MulBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=num_channel, out_channels=inner_channel, kernel_size=1, stride=1, padding=0, dilation=1))
        layers.append(Multiplier())
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        layers.append(nn.Conv2d(in_channels=inner_channel, out_channels=num_channel, kernel_size=1, stride=1, padding=0, dilation=1))
        layers.append(Multiplier())
        layers.append(nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=3, stride=1, padding=1, dilation=1))
        layers.append(Multiplier())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out

class OfficialNet(nn.Module):
    def __init__(self):
        super(OfficialNet, self).__init__()
        
        n_blks = 10
        n_feat_k = 72
        n_feat_m = 24
        n_feat_l = 6
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=n_feat_k, kernel_size=3, stride=1, padding=1, dilation=1),
            Multiplier(),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self._body = nn.Sequential(
            nn.Conv2d(in_channels=n_feat_k, out_channels=n_feat_k, kernel_size=1, stride=1, padding=0, dilation=1),
            Multiplier(),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=n_feat_k, out_channels=n_feat_m, kernel_size=1, stride=1, padding=0, dilation=1),
            Multiplier(),
            nn.Conv2d(in_channels=n_feat_m, out_channels=n_feat_m, kernel_size=3, stride=1, padding=1, dilation=1),
            Multiplier()
        )
        
        body = []
        for _ in range(n_blks):
            body.append(MulBlock(num_channel=n_feat_m, inner_channel=n_feat_k))
        
        self.body = nn.Sequential(*body)
        
        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=n_feat_m, out_channels=n_feat_l, kernel_size=3, stride=1, padding=1, dilation=1),
            Multiplier()
        )
        
        self._reshape1 = Lambda(lambda x: x[:, :6, 4:68, 4:68])
        self._reshape2 = Lambda(lambda x: x[:, :, 4:68, 4:68])
    
    def forward(self, x):
        out = self.head(x)
        out = self._body(out)
        out = self.body(out)
        out = self.tail(out)
        
        x = self._reshape1(x)
        out = self._reshape2(out)
        print(x.shape)
        print(out.shape)
        
        return out + x
