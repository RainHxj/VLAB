import math
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union
from functools import partial
import torch.nn.functional as F

from collections import OrderedDict
from torch.functional import Tensor
from torch.nn.modules.utils import _triple

# from utils.logger import LOGGER

import ipdb

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        # orig_type = x.dtype
        # ret = super().forward(x.type(torch.float32))
        # return ret.type(orig_type)
        return super().forward(x)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class RouteFuncMLP(nn.Module):
    """
    The routing function for generating the calibration weights.
    """

    def __init__(self,
                 c_in,
                 c_out,
                 mid_dim,
                 kernels,
                 bn_eps=1e-5,
                 bn_mmt=0.1,
                 concat=True):
        """
        Args:
            c_in (int): number of input channels.
            # ratio (int): reduction ratio for the routing function.
            mid_dim: emb dim after reduction
            kernels (list): temporal kernel size of the stacked 1D convolutions
        """
        super(RouteFuncMLP, self).__init__()
        self.c_in = c_in
        self.concat = concat
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        self.a = nn.Conv3d(
            in_channels=c_in * 2 if concat else 1,
            out_channels=mid_dim,
            kernel_size=[kernels[0], 1, 1],
            padding=[kernels[0] // 2, 0, 0],
        )
        self.bn = nn.BatchNorm3d(mid_dim, eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(in_channels=mid_dim,
                           out_channels=c_out,
                           kernel_size=[kernels[1], 1, 1],
                           padding=[kernels[1] // 2, 0, 0],
                           bias=False)
        self.b.skip_init = True
        self.b.weight.data.zero_(
        )  # to make sure the initial values  for the output is 1.

    def forward(self, cc_emb, cls_emb, v_mask):
        calibrate_emb = torch.cat(
            [cls_emb,
             self.g(cc_emb).expand(-1, -1, cls_emb.shape[2], -1, -1)],
            dim=1) if self.concat else cls_emb + self.g(cc_emb)
        calibrate_emb = torch.einsum('bcshw,bs->bcshw', calibrate_emb,
                                     v_mask.type(calibrate_emb.dtype))
        x = self.a(calibrate_emb)
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x


class TAdaConv2d(nn.Module):
    """
    Performs temporally adaptive 2D convolution.
    Currently, only application on 5D tensors is supported, which makes TAdaConv2d 
        essentially a 3D convolution with temporal kernel size of 1.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 cal_dim="cin"):
        super(TAdaConv2d, self).__init__()
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (list): kernel size of TAdaConv2d. 
            stride (list): stride for the convolution in TAdaConv2d.
            padding (list): padding for the convolution in TAdaConv2d.
            dilation (list): dilation of the convolution in TAdaConv2d.
            groups (int): number of groups for TAdaConv2d. 
            bias (bool): whether to use bias in TAdaConv2d.
        """

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1
        assert cal_dim in ["cin", "cout"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cal_dim = cal_dim

        # base weights (W_b)
        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups,
                         kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, alpha):
        """
        Args:
            x (tensor): feature to perform convolution on.
            alpha (tensor): calibration weight for the base weights.
                W_t = alpha_t * W_b
        """
        _, _, c_out, c_in, kh, kw = self.weight.size()
        b, c_in, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(1, -1, h, w)

        if self.cal_dim == "cin":
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, 1, C, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(2) *
                      self.weight).reshape(-1, c_in // self.groups, kh, kw)
        elif self.cal_dim == "cout":
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, C, 1, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(3) *
                      self.weight).reshape(-1, c_in // self.groups, kh, kw)

        bias = None
        if self.bias is not None:
            # in the official implementation of TAda2D,
            # there is no bias term in the convs
            # hence the performance with bias is not validated
            bias = self.bias.repeat(b, t, 1).reshape(-1)
        output = F.conv2d(x,
                          weight=weight,
                          bias=bias,
                          stride=self.stride[1:],
                          padding=self.padding[1:],
                          dilation=self.dilation[1:],
                          groups=self.groups * b * t)

        output = output.view(b, t, c_out, output.size(-2),
                             output.size(-1)).permute(0, 2, 1, 3, 4)

        return output

    def __repr__(self):
        return f"TAdaConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, " +\
            f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}, cal_dim=\"{self.cal_dim}\")"


class VideoAdapter(nn.Module):

    def __init__(self,
                 embed_dim=512,
                 mid_dim=64,
                 conv_mid=16,
                 n_head=2,
                 attn_mask=None,
                 scale=0.1,
                 pos=None,
                 placeholder=True) -> None:
        super().__init__()
        self.placeholder = placeholder
        if placeholder:
            return
        self.embed_dim = embed_dim
        self.mid_dim = mid_dim
        self.conv_mid = conv_mid
        self.n_head = n_head
        self.down = nn.Linear(embed_dim, mid_dim)
        self.up = nn.Linear(mid_dim, embed_dim)
        self.patch_down = nn.Linear(embed_dim, conv_mid)
        self.patch_up = nn.Linear(conv_mid, embed_dim)
        self.act = QuickGELU()
        self.scale = scale
        self.ln_pre = LayerNorm(mid_dim)
        self.block = ResidualAttentionAdapterBlock(mid_dim, n_head, attn_mask)
        self.positional_embedding = nn.Parameter(torch.randn(12, mid_dim))
        if pos is not None:
            self.positional_embedding = nn.Parameter(pos.clone()[:, :mid_dim])
        else:
            nn.init.normal_(self.positional_embedding, std=0.01)
        self.cc = nn.Parameter(mid_dim**-.5 * torch.randn(mid_dim))
        self.conv_rf = RouteFuncMLP(
            c_in=mid_dim,  # number of input filters
            c_out=conv_mid,
            mid_dim=16,  # reduction ratio for MLP
            kernels=[3, 3],  # list of temporal kernel sizes
        )
        self.conv_tada = TAdaConv2d(
            in_channels=conv_mid,
            out_channels=conv_mid,
            kernel_size=[
                1, 3, 3
            ],  # usually the temporal kernel size is fixed to be 1
            stride=[1, 1, 1],  # usually the temporal stride is fixed to be 1
            padding=[0, 1, 1],  # usually the temporal padding is fixed to be 0
            bias=False,
            cal_dim="cin")
        self.init()

    def forward(self, x: Tensor, bs, mask=None):
        """
        input shape v16 197 * (bs * frames) * 512
        cls 1 * (bs * frames) * 512
        video input: cls bs * frames
        """
        if self.placeholder:

            return x

        flag_list=False
        if isinstance(x,(list,tuple)):
           y = x[0]
           x = x[1]
           flag_list=True
        cls_emb, patch_emb = x[0], x[1:]
        cls_down = self.act(self.down(cls_emb))
        patch_down = self.act(self.patch_down(patch_emb))
        # down = self.act(self.down(x))
        patch_num = int((x.shape[0] - 1)**.5)
        cls_down = cls_down.reshape(bs, -1, self.mid_dim)
        patch_down = patch_down.reshape(patch_num, patch_num, bs,
                                        -1, self.conv_mid).permute(
                                            2, 4, 3, 0,
                                            1)  # bs, e, frames, pn, pn

        # cls_down = self.down(cls_emb).reshape(bs, -1, self.mid_dim)
        # cls_down = self.act(cls_down)   # bs, frame, mid_dim
        dt, device = cls_down.dtype, cls_down.device
        mid_dim = cls_down.shape[-1]
        cls_down = torch.cat([
            self.cc.to(dt) +
            torch.zeros(bs, 1, mid_dim, dtype=dt, device=device), cls_down
        ],
                             dim=1)
        pos_emd = self.positional_embedding[:cls_down.size(1), :].to(x.dtype)
        cls_down = cls_down + pos_emd

        cls_down = self.ln_pre(cls_down)
        cls_down = cls_down.permute(1, 0, 2)

        if mask is None:
            mask = torch.ones(bs, x.size(1) // bs).to(device)
        v_mask = mask
        mask = torch.cat([torch.ones(bs, 1).to(device), mask], dim=1)
        e_mask = (1.0 - mask.unsqueeze(1)) * -1000000.0
        e_mask = e_mask.expand(-1, mask.size(1), -1)
        attn_mask_ = e_mask.repeat_interleave(self.n_head, dim=0)
        final_mask = self.block.attn(cls_down,
                                     cls_down,
                                     cls_down,
                                     need_weights=False,
                                     attn_mask=attn_mask_)[0]

        cls_down = self.block((cls_down, None, None, 0, None), final_mask)[0]
        cls_down = cls_down.permute(1, 0, 2)  # bs, frames + 1, e_a
        alpha = self.conv_rf(cls_down[:, 0][..., None, None, None],
                             cls_down[:, 1:].permute(0, 2, 1)[..., None,
                                                              None], v_mask)
        # patch_down = self.conv_tada(patch_down.float(), alpha).to(dt).permute(
        patch_down = self.conv_tada(patch_down, alpha).to(dt).permute(
            3, 4, 0, 2, 1).reshape(patch_num**2, -1, self.conv_mid)

        cls_down = self.act(cls_down)
        patch_down = self.act(patch_down)
        delta_x = torch.cat([
            self.up(cls_down[:, 1:].reshape(1, -1, mid_dim)),
            self.patch_up(patch_down)
        ])
        x_out = x + self.scale * delta_x

        if flag_list:
            return (y, x_out)
        else:
            return x_out

    def init(self):
        proj_std = ((2 * self.embed_dim)**-0.5)
        attn_std = self.embed_dim**-0.5
        fc_std = (2 * self.embed_dim)**-0.5
        nn.init.normal_(self.block.attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.block.attn.out_proj.weight, std=proj_std)
        nn.init.normal_(self.block.mlp.c_fc.weight, std=fc_std)
        nn.init.normal_(self.block.mlp.c_proj.weight, std=proj_std)

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.patch_up.weight)
        nn.init.zeros_(self.patch_down.bias)
        nn.init.zeros_(self.up.bias)


class ResidualAttentionAdapterBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask=None,
                 seq_len=32,
                 adapt=None,
                 pos=None,
                 idx=0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        # self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)),
                         ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        txt_len = x.shape[0]
        attn_mask = self.attn_mask.to(
            dtype=x.dtype, device=x.device
        )[:txt_len, :txt_len] if self.attn_mask is not None else None
        # attn_mask_ = self.attn_mask
        # if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
        # attn_mask_ = self.attn_mask(x.size(0))  # LND

        # attn_mask_ = attn_mask_.to(
        # dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x_tuple: tuple, attn=None):
        """
        video adapter 中 attn不为None，其他为None，transformer中为mask不为None用于video adapter中调用
        """
        x, video_frame, bs, idx, mask = x_tuple
        if attn is None:
            attn = self.attention(self.ln_1(x))
        x1 = x + attn

        x2 = self.mlp(self.ln_2(x1))
        x3 = x1 + x2
        return (x3, video_frame, bs, idx + 1, mask)


class ResidualAttentionBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask=None,
                 seq_len=32,
                 pos=None,
                 idx=0,
                 use_checkpoint=False,
                 adapt=None,
                 start_adapt_layer=-1,
                 adapt_cls_dim=64,
                 adapt_patch_dim=16,
                 adapt_order='adapt_last'):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)),
                         ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.video_adapter, self.img_adapter, self.text_adapter = nn.Identity(
        ), nn.Identity(), nn.Identity()

        self.adapt = adapt
        self.adapt_order = adapt_order
        self.video_adapter = VideoAdapter(
            d_model,
            attn_mask=attn_mask,
            pos=pos,
            mid_dim=adapt_cls_dim,
            conv_mid=adapt_patch_dim,
            placeholder=not (adapt == 'video' and idx > start_adapt_layer))
        # self.img_adapter = ConvAdapter(
        #     d_model,
        #     placeholder=not (idx <= -1 and adapt in ['image', 'video']))
        # self.text_adapter = Adapter(d_model, placeholder=not adapt == 'text')

    def attention(self, x: torch.Tensor):
        txt_len = x.shape[0]
        attn_mask = self.attn_mask.to(
            dtype=x.dtype, device=x.device
        )[:txt_len, :txt_len] if self.attn_mask is not None else None
        # attn_mask_ = self.attn_mask
        # if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
        # attn_mask_ = self.attn_mask(x.size(0))  # LND

        # attn_mask_ = attn_mask_.to(
        # dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        if self.training and self.use_checkpoint:
            blk = partial(self.attn, need_weights=False, attn_mask=attn_mask)
            x = torch.utils.checkpoint.checkpoint(blk, x, x, x)[0]
            return x
        else:
            return self.attn(x, x, x, need_weights=False,
                             attn_mask=attn_mask)[0]


    def forward(self, x_tuple:tuple, attn=None):

        x, video_frame, bs, idx, mask = x_tuple
        
        if isinstance(x, (list,tuple)):
            x1 = []
            x2 = []
            for x_v in x:
                if attn is None:
                    attn = self.attention(self.ln_1(x_v))
                x_v1 = x_v + attn

                x1.append(x_v1)

                if self.training and self.use_checkpoint:
                    x_v2 = torch.utils.checkpoint.checkpoint(self.mlp, self.ln_2(x_v1))
                else:
                    x_v2 = self.mlp(self.ln_2(x_v1))
                x2.append(x_v2)
        else:
            if attn is None:
                attn = self.attention(self.ln_1(x))
            x1 = x + attn

            if self.training and self.use_checkpoint:
                x2 = torch.utils.checkpoint.checkpoint(self.mlp, self.ln_2(x1))
            else:
                x2 = self.mlp(self.ln_2(x1))

        # if self.adapt_order == 'adapt_last':
        #     x3 = self.video_adapter(x2, bs, mask)
        # else: 
        #     raise NotImplementedError
        
        # if self.adapt is not None and bs != x2.shape[1]:
        if self.adapt_order == 'adapt_last':
            # x2 = torch.utils.checkpoint.checkpoint(self.video_adapter, x2, bs,
            #                                     mask)
            x2 = self.video_adapter(x2, bs, mask)
        else:
            # x1 = torch.utils.checkpoint.checkpoint(self.video_adapter, x1, bs,
            #                                     mask)
            x1 = self.video_adapter(x1, bs, mask)
        # elif self.adapt is not None and bs == x2.shape[1]:
        #     x2 = x2 + 0.0 * self.video_adapter(x2, bs, mask)
        
        if isinstance(x1,(list,tuple)) and isinstance(x2,(list,tuple)):
            return ((x1[0]+x2[0],x1[1]+x2[1]), video_frame, bs, idx + 1, mask)
        elif not isinstance(x1,(list,tuple)) and isinstance(x2,(list,tuple)):
            return ((x1+x2[0],x1+x2[1]), video_frame, bs, idx + 1, mask)
        elif not isinstance(x1,(list,tuple)) and not isinstance(x2,(list,tuple)):
            return (x1+x2, video_frame, bs, idx + 1, mask)





class ConvAdapter(nn.Module):

    def __init__(self,
                 embed_dim=512,
                 mid_dim=16,
                 mid_k=5,
                 xavier_init=False,
                 scale=.1,
                 dropout=0.,
                 placeholder=True) -> None:
        super().__init__()
        self.placeholder = placeholder
        if placeholder:
            return
        self.down = nn.Linear(embed_dim, mid_dim)
        self.up = nn.Linear(mid_dim, embed_dim)
        self.conv = nn.Conv2d(mid_dim, mid_dim, mid_k, 1, (mid_k - 1) // 2)
        self.act = QuickGELU()
        self.dim = mid_dim
        self.dropout = nn.Dropout(dropout)
        self.scale = scale
        if xavier_init:
            nn.init.xavier_uniform_(self.conv.weight)
        else:
            nn.init.zeros_(self.conv.weight)
            self.conv.weight.data[:, :, 1, 1] += torch.eye(mid_dim,
                                                           dtype=torch.float)
        nn.init.zeros_(self.conv.bias)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.droppath = nn.Dropout(dropout)

    def forward(self, x, res=None):
        if self.placeholder:
            return x
        if res is None: res = x
        x = x.permute(1, 0, 2)
        B, N, C = x.shape
        P = int(math.sqrt(N - 1))
        x_down = self.down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_patch = x_down[:, 1:].reshape(B, P, P, self.dim).permute(0, 3, 1, 2)
        x_patch = self.conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, P**2, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.up(x_down)  # equivalent to 1 * 1 Conv
        return res + (x_up * self.scale).permute(1, 0, 2)


class Adapter(nn.Module):

    def __init__(self,
                 embed_dim=512,
                 mid_dim=64,
                 dropout=0,
                 init_option="lora",
                 adapter_scalar="0.1",
                 adapter_layernorm_option=None,
                 double_relu=False,
                 placeholder=True):
        super().__init__()
        self.placeholder = placeholder
        if placeholder:
            return
        self.embed_dim = embed_dim
        self.down_size = mid_dim

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = LayerNorm(self.embed_dim)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1) * .1)
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.embed_dim, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.embed_dim)

        self.double_relu = double_relu
        self.dropout = dropout
        if init_option == "lora":
            # with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, res=None):
        if self.placeholder:
            return x
        residual = x if res is None else res
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down,
                                     p=self.dropout,
                                     training=self.training)
        up = self.up_proj(down)
        if self.double_relu:
            up = self.non_linear_func(up)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Transformer(nn.Module):

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask=None,
                 seq_len=32,
                 adapt=None,
                 start_adapt_layer=-1,
                 adapt_cls_dim=64,
                 adapt_patch_dim=16,
                 adapt_order='adapt_last',
                 pos=None,
                 use_checkpoint=False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.use_checkpoint = use_checkpoint
        # self.resblocks = nn.Sequential(*[
        #     ResidualAttentionBlock(
        #         width, heads, attn_mask, seq_len, adapt, pos, idx=i)
        #     for i in range(layers)
        # ])
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width,
                                   heads,
                                   attn_mask,
                                   seq_len,
                                   pos=pos,
                                   idx=i,
                                   use_checkpoint=use_checkpoint,
                                   adapt=adapt,
                                   adapt_order=adapt_order,
                                   start_adapt_layer=start_adapt_layer,
                                   adapt_cls_dim=adapt_cls_dim,
                                   adapt_patch_dim=adapt_patch_dim)
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor, mask=None, video_frame=-1, bs=-1):
        # return self.resblocks((x, video_frame, bs, 0, mask))[0]
        # return (x3, video_frame, bs, idx + 1, mask)
        for idx, blk in enumerate(self.resblocks):
            # if self.use_checkpoint and self.training:
            #     x, video_frame, bs, _, mask  = torch.utils.checkpoint.checkpoint(
            #         blk,
            #         (x,
            #         video_frame,
            #         bs,
            #         idx,
            #         mask),
            #     )
            # else:
            x, video_frame, bs, _, mask = blk((x, video_frame, bs, idx, mask))
            # ipdb.set_trace()

        if isinstance(x,(list,tuple)):
            return x
        return x


class VisualTransformer(nn.Module):

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        linear_patch: str = '2d',
        use_checkpoint: bool = False,
        adapt_type='video',
        adapt_order='adapt_last',
        start_adapt_layer=-1,
        adapt_cls_dim=64,
        adapt_patch_dim=16,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.seq_len = (input_resolution //
                        patch_size)**2 + 1  # + self.prompt_len
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.seq_len, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width,
                                       layers,
                                       heads,
                                       seq_len=self.seq_len,
                                       adapt=adapt_type,
                                       adapt_order=adapt_order,
                                       start_adapt_layer=start_adapt_layer,
                                       adapt_cls_dim=adapt_cls_dim,
                                       adapt_patch_dim=adapt_patch_dim,
                                       pos=self.positional_embedding,
                                       use_checkpoint=use_checkpoint)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # For 3D
        assert linear_patch in ['2d', '3d']
        self.linear_patch = linear_patch
        if self.linear_patch == '3d':
            self.conv2 = nn.Conv3d(in_channels=3,
                                   out_channels=width,
                                   kernel_size=(3, patch_size, patch_size),
                                   stride=(1, patch_size, patch_size),
                                   padding=(1, 0, 0),
                                   bias=False)

    def forward(self, x: torch.Tensor, video_frame=-1, bs=-1, mask=None):

        if self.linear_patch == '3d':
            assert video_frame != -1
            x_3d = x.reshape(-1, video_frame, x.shape[-3], x.shape[-2],
                             x.shape[-1])
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            x_3d = self.conv2(x_3d)  # shape = [*, width, frame, grid, grid]
            x_3d = x_3d.permute(0, 2, 1, 3,
                                4)  # shape = [*, frame, width, grid, grid]
            x = x_3d.reshape(
                -1, x_3d.shape[-3], x_3d.shape[-2],
                x_3d.shape[-1]).contiguous()  # shape = [*, width, grid, grid]
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                    device=x.device),
                # self.prompt.to(x.dtype) + torch.zeros(x.shape[0], self.prompt_len, x.shape[-1], dtype=x.dtype, device=x.device),
                x
            ],
            dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, video_frame=video_frame, bs=bs, mask=mask)
        
        if isinstance(x,(list,tuple)):
            y = []
            for x_v in x:
                x_v = x_v.permute(1, 0, 2)  # LND -> NLD

                # Move the three lines below to `encode_image` for entire hidden sequence
                x_v = self.ln_post(x_v)

                y.append(x_v)
            return y


        x = x.permute(1, 0, 2)  # LND -> NLD

        # Move the three lines below to `encode_image` for entire hidden sequence
        x = self.ln_post(x)
        # x = self.ln_post(x[:, 0, :])
        # if self.proj is not None:
        #     x = x @ self.proj

        return x


class VideoCLIP(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        linear_patch: str = '2d',
        use_checkpoint=False,
        adapt_type='video',
        adapt_order='adapt_last',
        start_adapt_layer=-1,
        adapt_cls_dim=64,
        adapt_patch_dim=16,
        frozen_vision=False
        # token_rolling
        # frame_num: int
    ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            linear_patch=linear_patch,
            use_checkpoint=use_checkpoint,
            adapt_type=adapt_type,
            adapt_order=adapt_order,
            start_adapt_layer=start_adapt_layer,
            adapt_cls_dim=adapt_cls_dim,
            adapt_patch_dim=adapt_patch_dim,
        )

        self.transformer = Transformer(width=transformer_width,
                                       layers=transformer_layers,
                                       heads=transformer_heads,
                                       attn_mask=self.build_attention_mask(),
                                       adapt=None)

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        # frozen_vision = False
        self.frozen_vision = frozen_vision

        if self.frozen_vision:
            self.frozen_vision_parameters()

    def frozen_vision_parameters(self,):
        
        print("all vision parameters are frozen.")

        for k,p in self.visual.named_parameters():
            if 'adapt' in k:
                continue
            p.requires_grad = False


    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers)**-0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width)**-0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection,
                            std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        b, n, _, h, w = image.shape
        image = image.reshape(b * n, 3, h, w)
        return self.visual(image.type(self.dtype), bs=b)

    def encode_text(self, text):
        # if isinstance(text,dict):
        #     text = text['clip_tokens']
        x = self.token_embedding(text).type(
            self.dtype)  # [batch_size, n_ctx, d_model]
        x_len = x.shape[1]
        x = x + self.positional_embedding.type(self.dtype)[:x_len]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        #x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1,
                                                              keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                    *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                    "in_proj_bias", "bias_k", "bias_v"
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_adapter(state_dict: dict,
                  rolling_ratio=0,
                  frame_num=0,
                  use_checkpoint=False,
                  start_adapt_layer=-1,
                  adapt_cls_dim=64,
                  adapt_patch_dim=16,
                  adapt_type='video',
                  adapt_order='adapt_last',
                  frozen_vision=False):
    vit = "visual.proj" in state_dict

    # ipdb.set_trace()
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([
        k for k in state_dict.keys()
        if k.startswith("visual.") and k.endswith("attn.in_proj_weight") and "video_adapter" not in k
    ])

    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]

    grid_size = round(
        (state_dict["visual.positional_embedding"].shape[0] - 1)**0.5)
    image_resolution = vision_patch_size * grid_size
    if image_resolution == 336:  ##### use 224 to save memory
        # pass
        image_resolution = 224
        src = state_dict["visual.positional_embedding"]
        src_cls = src[0:1]
        src_oth = src[1:]
        src_oth = F.interpolate(src_oth.reshape(24, 24,
                                                1024).permute(2, 0,
                                                              1).unsqueeze(0),
                                (16, 16),
                                mode='bilinear')
        src_oth = src_oth[0].permute(1, 2, 0).reshape(-1, 1024)
        tgt = torch.cat((src_cls, src_oth), dim=0)
        state_dict["visual.positional_embedding"] = tgt

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    #context_length = max_txt_len+2
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2] for k in state_dict
            if k.startswith(f"transformer.resblocks")))

    model = VideoCLIP(embed_dim,
                      image_resolution,
                      vision_layers,
                      vision_width,
                      vision_patch_size,
                      context_length,
                      vocab_size,
                      transformer_width,
                      transformer_heads,
                      transformer_layers,
                      linear_patch='2d',
                      use_checkpoint=use_checkpoint,
                      adapt_type=adapt_type,
                      adapt_order=adapt_order,
                      start_adapt_layer=start_adapt_layer,
                      adapt_cls_dim=adapt_cls_dim,
                      adapt_patch_dim=adapt_patch_dim,
                      frozen_vision=frozen_vision)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict,
                                                          strict=False)
    LOGGER.info(f"CLIP Unexpected keys {unexpected_keys}")
    LOGGER.info(f"CLIP Missing_keys  {missing_keys}")

    return model.eval()
