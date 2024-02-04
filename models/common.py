import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale




class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


class ConvBNReLU2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, act=None, norm=None):
        super(ConvBNReLU2D, self).__init__()

        self.layers = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.act = None
        self.norm = None
        if norm == 'BN':
            self.norm = torch.nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm = torch.nn.InstanceNorm2d(out_channels)
        elif norm == 'GN':
            self.norm = torch.nn.GroupNorm(2, out_channels)
        elif norm == 'WN':
            self.layers = torch.nn.utils.weight_norm(self.layers)
        elif norm == 'Adaptive':
            self.norm = AdaptiveNorm(n=out_channels)

        if act == 'PReLU':
            self.act = torch.nn.PReLU()
        elif act == 'SELU':
            self.act = torch.nn.SELU(True)
        elif act == 'LeakyReLU':
            self.act = torch.nn.LeakyReLU(negative_slope=0.02, inplace=True)
        elif act == 'ELU':
            self.act = torch.nn.ELU(inplace=True)
        elif act == 'ReLU':
            self.act = torch.nn.ReLU(True)
        elif act == 'Tanh':
            self.act = torch.nn.Tanh()
        elif act == 'Sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'SoftMax':
            self.act = torch.nn.Softmax2d()

    def forward(self, inputs):

        out = self.layers(inputs)

        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, num_features):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            ConvBNReLU2D(num_features, out_channels=num_features, kernel_size=3, act='ReLU', padding=1),
            ConvBNReLU2D(num_features, out_channels=num_features, kernel_size=3, padding=1)
        )
        
    def forward(self, inputs):
        return F.relu(self.layers(inputs) + inputs)


class ResBlock_v2(nn.Module):
    def __init__(self, num_features):
        super(ResBlock_v2, self).__init__()
        self.layers = nn.Sequential(
            ConvBNReLU2D(num_features, out_channels=num_features, kernel_size=3, act='ReLU', padding=1),
            ConvBNReLU2D(num_features, out_channels=num_features, kernel_size=3, padding=1)
        )
        
        # add fre channel attention
        self.att = MultiSpectralAttentionLayer(channel=num_features,dct_h=128,dct_w=128,   # test的input为256x256?
                                               reduction=16,freq_sel_method='top16')

    def forward(self, inputs):
        out1 = self.layers(inputs)
        out2 = self.att(out1)
        return F.relu(out2 + inputs)


class SpaGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, act,norm, n_resblocks):
        super(SpaGroup, self).__init__()
        modules_body = [
            ResBlock(n_feat) for _ in range(n_resblocks)]

        modules_body.append(ConvBNReLU2D(n_feat, n_feat, kernel_size, padding=1, act=act, norm=norm))
        self.body = nn.Sequential(*modules_body)
        self.re_scale = Scale(1)

    def forward(self, x):
        res = self.body(x)
        return res + self.re_scale(x)

# add fre channel attention
class SpaGroup_v2(nn.Module):
    def __init__(self, n_feat, kernel_size, act,norm, n_resblocks):
        super(SpaGroup_v2, self).__init__()
        modules_body = [
            ResBlock_v2(n_feat) for _ in range(n_resblocks)]

        modules_body.append(ConvBNReLU2D(n_feat, n_feat, kernel_size, padding=1, act=act, norm=norm))
        self.body = nn.Sequential(*modules_body)
        self.re_scale = Scale(1)

    def forward(self, x):
        res = self.body(x)
        return res + self.re_scale(x)   



class FreBlock(nn.Module):
    def __init__(self, channels, args):
        super(FreBlock, self).__init__()

        self.fpre = nn.Conv2d(channels, channels, 1, 1, 0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, channels, 3, 1, 1))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, channels, 3, 1, 1))
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)


    def forward(self, x):
        # print("x: ", x.shape)
        _, _, H, W = x.shape
        msF = torch.fft.rfft2(self.fpre(x)+1e-8, norm='backward')

        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        # print("msf_amp: ", msF_amp.shape)
        amp_fuse = self.amp_fuse(msF_amp)
        # print(amp_fuse.shape, msF_amp.shape)
        amp_fuse = amp_fuse + msF_amp
        pha_fuse = self.pha_fuse(msF_pha)
        pha_fuse = pha_fuse + msF_pha

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        out = self.post(out)
        out = out + x
        out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        # print("out: ", out.shape)
        return out


class FreBlock_v2(nn.Module):
    def __init__(self, channels, args):
        super(FreBlock_v2, self).__init__()

        self.fpre = nn.Conv2d(channels, channels, 1, 1, 0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, channels, 3, 1, 1))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, channels, 3, 1, 1))
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

        # add fre channel attention
        self.att = MultiSpectralAttentionLayer(channel=channels,dct_h=128,dct_w=128,   # test的input为256x256?
                                               reduction=16,freq_sel_method='top16')
    def forward(self, x):
        # print("x: ", x.shape)
        _, _, H, W = x.shape
        msF = torch.fft.rfft2(self.fpre(x)+1e-8, norm='backward')

        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        # print("msf_amp: ", msF_amp.shape)
        amp_fuse = self.amp_fuse(msF_amp)
        # print(amp_fuse.shape, msF_amp.shape)
        amp_fuse = amp_fuse + msF_amp
        pha_fuse = self.pha_fuse(msF_pha)
        pha_fuse = pha_fuse + msF_pha

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        out = self.post(out)
        
        out = self.att(out)    # add fre channel attention
        out = out + x
        out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        # print("out: ", out.shape)
        return out

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

class FuseBlock_v2(nn.Module):
    def __init__(self, channels):
        super(FuseBlock_v2, self).__init__()
        self.fre = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fre_att = Attention(dim=channels)
        self.spa_att = Attention(dim=channels)   # spa att
        # add fre channel attention
        self.cha_att = MultiSpectralAttentionLayer(channel=channels,dct_h=128,dct_w=128,   # test的input为256x256?
                                               reduction=16,freq_sel_method='top2')
        
        self.fuse = nn.Sequential(nn.Conv2d(2*channels, channels, 3, 1, 1), nn.Conv2d(channels, 2*channels, 3, 1, 1), nn.Sigmoid())

    def forward(self, spa, fre):
        ori = spa
        fre = self.fre(fre)
        spa = self.spa(spa)
        fre = self.cha_att(self.fre_att(fre, spa)) + fre     # fca
        spa = self.cha_att(self.fre_att(spa, fre)) + spa     # fca
        fuse = self.fuse(torch.cat((fre, spa), 1))        
        fre_a, spa_a = fuse.chunk(2, dim=1)
        spa = spa_a * spa
        fre = fre * fre_a
        res = fre + spa

        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res




class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m
    
    
    
#---------------------------------------------------------------#
#--------------------------fre Channel Attention----------------#
#---------------------------------------------------------------#

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)       # 16
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)    # 64 //  16 = 4

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter

