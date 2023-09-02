import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

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


class conv_ffn(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pointwise1 = torch.nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.depthwise = torch.nn.Conv2d(hidden_features,hidden_features, kernel_size=3,stride=1,padding=1,dilation=1,groups=hidden_features)
        self.pointwise2 = torch.nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise2(x)
        return x


class TransformerBlock(nn.Module):
    """
    from restormer
    input size: (B,C,H,W)
    output size: (B,C,H,W)
    H, W could be different
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = conv_ffn(dim, dim*ffn_expansion_factor, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Conv_Transformer(nn.Module):
    """
    sepconv replace conv_out to reduce GFLOPS
    """
    def __init__(self, in_channel,num_heads=8,ffn_expansion_factor=2):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv = nn.Conv2d(in_channels=in_channel,out_channels=in_channel,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1))
        self.Transformer = TransformerBlock(dim=in_channel, num_heads=num_heads,
                                            ffn_expansion_factor=ffn_expansion_factor, bias=True)
        self.channel_reduce = nn.Conv2d(in_channels=in_channel*2,out_channels=in_channel,kernel_size=1,stride=1)
        self.Conv_out = nn.Conv2d(in_channels=in_channel,out_channels=in_channel,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1))
    def forward(self, x):
        conv = self.lrelu(self.conv(x))
        trans = self.Transformer(x)
        x = torch.cat([conv, trans], 1)
        x = self.channel_reduce(x)
        x = self.lrelu(self.Conv_out(x))
        return x


class Cat_former(nn.Module):
    def __init__(self, in_ch=3, down_rate=4):
        super().__init__()
        self.downsample = nn.PixelUnshuffle(down_rate)
        self.model = Conv_Transformer(in_ch * down_rate * down_rate)

    def forward(self, x):
        """
        input x channel = c
        output x channel = c * down_rate * down_rate * 4
        """
        x = self.downsample(x)
        x0 = self.model(x)
        x1 = self.model(x0)
        x2 = self.model(x1)
        x = torch.concat((x, x0, x1, x2), dim=1)
        # x = self.downsample(x)
        return x


# img = torch.randn(8, 3, 512, 512)
#
# model = Cat_former(3)
# x = model(img)
#
# print(x.shape)