import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

import einops
from einops import rearrange
import numpy as np


#### Cross-layer Attention Fusion Block
class LAM_Module_v2(nn.Module):  
    """ Layer attention module"""
    def __init__(self, in_dim,bias=True):
        super(LAM_Module_v2, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d( self.chanel_in ,  self.chanel_in *3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in*3, self.chanel_in*3, kernel_size=3, stride=1, padding=1, groups=self.chanel_in*3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize,N*C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1+x
        out = out.view(m_batchsize, -1, height, width)
        return out

##########################################################################
class FuseBlock_v2(nn.Module):
    def __init__(self, channels):
        super(FuseBlock_v2, self).__init__()
        self.fre = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.att = LAM_Module_v2(in_dim=int(channels*2))
        # self.spa_att = LAM_Module_v2(in_dim=int(channels*2))
        self.conv_fuss = nn.Conv2d(int(channels * 2), int(channels), kernel_size=1, bias=False)
        
        # self.fuse = nn.Sequential(nn.Conv2d(2*channels, channels, 3, 1, 1), nn.Conv2d(channels, 2*channels, 3, 1, 1), nn.Sigmoid())


    def forward(self, spa, fre):
        ori = spa
        fre = self.fre(fre)
        spa = self.spa(spa)
        
        inp_fusion = torch.cat([fre.unsqueeze(1),spa.unsqueeze(1)],dim=1)    # B,2,C,H,W
        att_out = self.att(inp_fusion)
        fuse = self.conv_fuss(att_out)
        # fre_a, spa_a = fuse.chunk(2, dim=1)
        # spa = spa_a * spa
        # fre = fre * fre_a
        res = fuse + spa

        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res


##########################################################################
class FuseBlock_v3(nn.Module):  # add the spa_res B 3 C H W
    def __init__(self, channels):
        super(FuseBlock_v3, self).__init__()
        self.fre = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.att = LAM_Module_v2(in_dim=int(channels*3))
        # self.spa_att = LAM_Module_v2(in_dim=int(channels*2))
        self.conv_fuss = nn.Conv2d(int(channels * 3), int(channels), kernel_size=1, bias=False)
        
        # self.fuse = nn.Sequential(nn.Conv2d(2*channels, channels, 3, 1, 1), nn.Conv2d(channels, 2*channels, 3, 1, 1), nn.Sigmoid())


    def forward(self, spa, fre,spa_res):
        ori = spa
        fre = self.fre(fre)
        spa = self.spa(spa)
        
        inp_fusion = torch.cat([fre.unsqueeze(1),spa_res.unsqueeze(1),spa.unsqueeze(1)],dim=1)    # B,3,C,H,W
        att_out = self.att(inp_fusion)
        fuse = self.conv_fuss(att_out)
        # fre_a, spa_a = fuse.chunk(2, dim=1)
        # spa = spa_a * spa
        # fre = fre * fre_a
        res = fuse + spa

        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res


##########################################################################
class FuseBlock(nn.Module):
    def __init__(self, channels):
        super(FuseBlock, self).__init__()
        self.fre = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fre_att = Attention(dim=channels)
        self.spa_att = Attention(dim=channels)
        self.fuse = nn.Sequential(nn.Conv2d(2*channels, channels, 3, 1, 1), nn.Conv2d(channels, 2*channels, 3, 1, 1), nn.Sigmoid())


    def forward(self, spa, fre):
        ori = spa
        fre = self.fre(fre)
        spa = self.spa(spa)
        
        
        
        fre = self.fre_att(fre, spa)+fre
        spa = self.fre_att(spa, fre)+spa
        fuse = self.fuse(torch.cat((fre, spa), 1))
        fre_a, spa_a = fuse.chunk(2, dim=1)
        spa = spa_a * spa
        fre = fre * fre_a
        res = fre + spa

        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res
    


class Attention(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
