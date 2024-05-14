
import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

    def initialize(self):
        weight_init(self)

def weight_init(module):
    for n, m in module.named_children():
      #  print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d,nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Sequential,nn.ModuleList,nn.ModuleDict)):
            weight_init(m)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (LayerNorm,nn.ReLU,nn.ReLU,nn.AdaptiveAvgPool2d,nn.Softmax,nn.AvgPool2d)):
            pass
        else:
            m.initialize()

#----------------------------------------DTF--------------------------------------------------------
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, lh, hl, hh

class BasicBlockL(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1, norm_layer=None):
        super(BasicBlockL, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                  padding=1, groups=groups)
        self.conv1_1 = nn.Conv2d(inplanes, planes, kernel_size=1, dilation=3)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2)

        self.conv2 =  nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                  padding=1, groups=groups)
        self.conv2_1 = nn.Conv2d(inplanes, planes, kernel_size=1, dilation=3)
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x):
        identity = x

        out1 = self.conv1(x)
        out1 = self.conv1_1(out1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 += identity
        out2 = self.conv2(out1)
        out2 = self.conv2_1(out2)
        out2 = self.bn2(out2)
        out = self.relu(out2)

        return out

class  BasicBlockH(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1, norm_layer=None):
        super(BasicBlockH, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.conv1_1 = nn.Conv2d(inplanes, planes, kernel_size=(1,3), padding=(0, 1))
        self.conv1_2 = nn.Conv2d(inplanes, planes, kernel_size=(3,1), padding=(1, 0))
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.conv2_1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 3), padding=(0, 1))
        self.conv2_2 = nn.Conv2d(inplanes, planes, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x):
        identity = x

        out1 = self.conv1(x)
        out1 = self.conv1_1(out1)
        out1 = self.conv1_2(out1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out = identity + out1

        out = self.conv2(out)
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class UpSampler(nn.Sequential):     # 上采样操作
    def __init__(self, scale, n_feats):

        m = []
        if scale == 8:
            kernel_size = 3
        elif scale == 16:
            kernel_size = 5
        else:
            kernel_size = 1

        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(in_channels=n_feats, out_channels=4 * n_feats, kernel_size=kernel_size, stride=1,
                                   padding=kernel_size // 2))
                m.append(nn.PixelShuffle(upscale_factor=2))
                m.append(nn.PReLU())
        super(UpSampler, self).__init__(*m)

class ConvBNReLu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, padding=1, dilation=1):
        super(ConvBNReLu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=1,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = LayerNorm(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def initialize(self):
        weight_init(self)

def window_partition(x, window_size):
    # input B C H W
    x = x.permute(0, 2, 3, 1)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows  # B_ H_ W_ C

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x.permute(0, 3, 1, 2)

class MLP(nn.Module):
    def __init__(self, inchannel, outchannel, bias=False):
        super(MLP, self).__init__()
        self.conv1 = nn.Linear(inchannel, outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.ln = nn.LayerNorm(outchannel)

    def forward(self, x):
        return self.relu(self.ln(self.conv1(x)) + x)

    def initialize(self):
        weight_init(self)

class FAI(nn.Module):  # x hf  y  lf
    def __init__(self, dim, num_heads=8, level=8, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.level = level
        self.mul = nn.Sequential(ConvBNReLu(dim, dim), ConvBNReLu(dim, dim, kernel_size=1, padding=0))
        self.add = nn.Sequential(ConvBNReLu(dim, dim), ConvBNReLu(dim, dim, kernel_size=1, padding=0))

        self.conv_x = nn.Sequential(ConvBNReLu(dim, dim), ConvBNReLu(dim, dim, kernel_size=1, padding=0))

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)

        self.lnx = nn.LayerNorm(dim)
        self.lny = nn.LayerNorm(dim)
        self.ln = nn.LayerNorm(dim)

        self.shortcut = nn.Linear(dim, dim)

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            LayerNorm(dim)
        )
        self.mlp = MLP(dim, dim)

    def forward(self, x, y):
        origin_size = x.shape[2]
        ws = origin_size // self.level // 4
        x = self.conv_x(x)

        x = window_partition(x, ws)
        y = window_partition(y, ws)

        x = x.view(x.shape[0], -1, x.shape[3])
        sc1 = x
        x = self.lnx(x)
        y = y.view(y.shape[0], -1, y.shape[3])
        y = self.lny(y)
        B, N, C = x.shape
        y_kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q = x_q[0]
        y_k, y_v = y_kv[0], y_kv[1]
        attn = (x_q @ y_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ y_v).transpose(1, 2).reshape(B, N, C)
        x = self.act(x + sc1)
        x = self.act(x + self.mlp(x))
        x = x.view(-1, ws, ws, C)
        x = window_reverse(x, ws, origin_size, origin_size)
        x = self.act(self.conv2(x) + x)
        return x

    def initialize(self):
        weight_init(self)
# ----------------------------------------Decoder----------------------------------------
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MFE(nn.Module):
    def __init__(self, inplanes, planes):
        super(MFE, self).__init__()
        self.conv_rgb0 = nn.Sequential(nn.Conv2d(inplanes, planes, 1, 1, 0),
                                       nn.BatchNorm2d(planes), nn.ReLU())
        self.conv_1 = nn.Sequential(nn.Conv2d(planes, planes, 3, dilation=1, padding=1),
                                       nn.BatchNorm2d(planes), nn.ReLU())
        self.conv_2 = nn.Sequential(nn.Conv2d(planes, planes, 3, dilation=3, padding=3),
                                       nn.BatchNorm2d(planes), nn.ReLU())
        self.conv_3 = nn.Sequential(nn.Conv2d(planes, planes, 3, dilation=5, padding=5),
                                       nn.BatchNorm2d(planes), nn.ReLU())
        self.conv_4 = nn.Sequential(nn.Conv2d(planes, planes, 3, dilation=7, padding=7),
                                       nn.BatchNorm2d(planes), nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=1), nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * planes, planes, kernel_size=1), nn.BatchNorm2d(planes), nn.PReLU()
        )
        self.up = nn.Sequential(*UpSampler(scale=4, n_feats=inplanes))
    def forward(self, f):
        f_0 = self.conv_rgb0(f)
        f_1 = self.conv_1(f_0)
        f_2 = self.conv_2(f_0)
        f_3 = self.conv_3(f_0)
        f_4 = self.conv_4(f_0)
        f_6 = self.fuse(torch.cat((f_1, f_2, f_3, f_4), 1))
        return f_6

class Attention(nn.Module):
    def __init__(self, dim=64,out=64,  num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, out, kernel_size=1, bias=bias)

    def forward(self, x, y):    # (fre, spa)
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # torch.Size([1, 8, 16, 144])
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)       # torch.Size([1, 8, 16, 144])
        k = torch.nn.functional.normalize(k, dim=-1)    # 归一化 torch.Size([1, 8, 16, 144])

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # 计算注意力权重
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out) + x     # torch.Size([1, 128, 12, 12])
        return out

class CIU(nn.Module):

    def __init__(self, inplanes, planes):
        super(CIU, self).__init__()
        self.GAP_Conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, inplanes, 1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )

        self.conv_5 = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1)
        self.att1 = Attention( dim=inplanes, out=inplanes)
        self.att = MFE(inplanes, planes)
        self.ra_conv1 = BasicConv2d(planes + planes, planes, kernel_size=3, padding=1)

    def forward(self, x, F5, edge):

        out = self.att(x)
        if F5.size()[2:] != x.size()[2:]:
            F5 = F.interpolate(F5, size=x.size()[2:], mode='nearest')
            edge = F.interpolate(edge, size=x.size()[2:], mode='nearest')

        v_attention = self.GAP_Conv(x)
        out_1 = self.att1(F5, x)
        y = out_1 * out * v_attention
        y = self.conv(y)
        y = self.ra_conv1(torch.cat((y, edge), dim=1))
        return y

#------------------------------------------------Decoder--------------------------------------------------------------
class Conv(nn.Module):
    def __init__(self, channel):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, 1, 1, 1, 0)
        )

    def forward(self, x):
        y = self.conv(x)
        return y






















