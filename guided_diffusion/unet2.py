from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import einops
import collections
from functools import partial
from itertools import repeat

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    LayerNorm32,
    Linear32,
    freeze_module,
    positional_embedding,
)

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class CondTimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, **kwargs):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, CondTimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, **kwargs):
        for layer in self:
            if isinstance(layer, CondTimestepBlock):
                x = layer(x, emb, **kwargs)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            Linear32(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class CrossAttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.q = conv_nd(1, channels, channels, 1)
        self.kv = conv_nd(1, channels, channels * 2, 1)
        self.norm_q = normalization(channels)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.use_checkpoint)

    def _forward(self, x, context):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        cb, cc, *_cspatial = context.shape
        assert cc == c
        context = context.reshape(cb, cc, -1)
        q = self.q(self.norm_q(context))
        kv = self.kv(self.norm(x))
        qkv = th.cat([q, kv], dim=1)
        h = self.attention(qkv)
        h = self.proj_out(h)
        return h.reshape(b, c, *spatial)

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class Attention(nn.Module):
    """
    """
    def __init__(self, embed_size, heads, bias, seq_len=None, dtype=th.float32):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = Linear32(self.embed_size, self.embed_size, bias=bias)
        self.keys = Linear32(self.embed_size, self.embed_size, bias=bias)
        self.queries = Linear32(self.embed_size, self.embed_size, bias=bias)
        self.fc_out = Linear32(heads * self.head_dim, embed_size)
        self.register_buffer("mask", None)
        self.seq_len = seq_len
        self.dtype = dtype
        self.fused_attn = False
        

    @staticmethod
    def generate_multihead_mask(seq_len, num_heads, device, dtype):
        mask = 1. - th.tril(th.ones(seq_len, seq_len, device=device, dtype=dtype))
        mask = einops.repeat(mask, 'l1 l2 ->t h l1 l2', h=num_heads, t=1)
        return mask

    def forward(self, x, idx_t=-2, context=None):
        # TODO: only compute the attention at the last element of the sequence to reduce computation
        b_x, t_x, _ = x.shape
        if idx_t != -2:
            if idx_t == -1:
                x = x[:, -1:, :]
            else:
                x = x[:, idx_t:idx_t+1, :]
            t_x = 1
        # Split the embedding into self.heads different pieces
        if context is None:
            context = x
        values  = self.values(x).view(b_x, t_x, self.heads, self.head_dim)
        keys    = self.keys(context).view(b_x, t_x, self.heads, self.head_dim)
        queries = self.queries(context).view(b_x, t_x, self.heads, self.head_dim)
        mask = self.generate_multihead_mask(self.seq_len if idx_t == -2 else 1, self.heads, x.device, self.dtype)

        # Attention calculation
        if self.fused_attn:
            out = F.scaled_dot_product_attention(queries, keys, values, is_causal=True).reshape(
                b_x, t_x, self.heads * self.head_dim)
        else:
            energy = th.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)
            energy = energy + mask*th.tensor(-1e9, dtype=self.dtype)
            attention = th.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # (N, heads, query_len, key_len) # TODO: check if this is correct, embed_size should be head_dim

            out = th.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
                b_x, t_x, self.heads * self.head_dim
            )  # (N, query_len, heads, head_dim) -> (N, query_len, heads * head_dim)

        return self.fc_out(out)

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilations=[1]):
        super(CausalConv1d, self).__init__()
        layers = []
        padding = 0
        
        for dilation in dilations:
            padding = (kernel_size - 1) * dilation
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False))
            in_channels = out_channels

        self.conv1 = nn.Sequential(*layers)

    def forward(self, x):
        x = einops.rearrange(x, 'b t c -> b c t')
        in_shape = x.shape
        out = self.conv1(x)[..., :in_shape[2]]
        return einops.rearrange(out, 'b c t -> b t c')


class CausalGatedResNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilations):
        super(CausalGatedResNet, self).__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilations)
        self.conv2 = CausalConv1d(out_channels, 2*out_channels, kernel_size, dilations)
        self.sigmoid = nn.Sigmoid()
        self.act     = nn.SiLU()

    def forward(self, x):

        x = self.conv1(self.act(x))
        x = self.conv2(self.act(x))
        a, b = th.chunk(x, 2, dim=-1)
        gate_x = a * self.sigmoid(b)

        return gate_x

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def modulate32(x, shift, scale):
    if x.dtype == th.float32:
        return x * (1 + scale) + shift
    else:
        return (x.float() * (1 + scale) + shift).type(x.dtype)

def mul32(a, b):
    if b.dtype == th.float32:
        return a * b
    else:
        return (a.float() * b.float()).type(b.dtype)
        
    
def add32(a, b):
    if b.dtype == th.float32:
        return a + b
    else:
        return (a.float() + b.float()).type(b.dtype)
        

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else Linear32

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class DiTBlockCausalGate(CondTimestepBlock):
    """
    DiTBlock with causal attention mask
    """
    def __init__(self, hidden_size, cond_dims, num_heads, emb_shape, mlp_ratio=4.0, seq_len=5, dtype=th.float32, dilations=[1, 1, 2, 2, 2]):
        super().__init__()
        self.norm1 = LayerNorm32(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, heads=num_heads, bias=True, seq_len=seq_len, dtype=dtype)
        self.causal_gate = CausalGatedResNet(hidden_size, hidden_size, 3, dilations)
        self.norm2 = LayerNorm32(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            Linear32(cond_dims, 6 * hidden_size, bias=True)
        )
        self.emb_shape = emb_shape

    def forward(self, x, emb, **kwargs):
        # TODO: pass idx_t to the attention block
        if 'idx_t' in kwargs.keys():
            idx_t = kwargs['idx_t']
        else:
            idx_t = -2
        emb = einops.rearrange(emb, '(b t) c -> b t c', t = self.emb_shape[0])[:,0,:] #TODO: could be optimized
        emb = einops.repeat(emb, 'b c -> b c h w', h = self.emb_shape[2], w = self.emb_shape[3])
        emb = SwitchDims((1, self.emb_shape[1], self.emb_shape[2], self.emb_shape[3]))(emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(emb).chunk(6, dim=2)
        gate_mlp = gate_mlp + self.causal_gate(x)
        gate_msa = gate_msa + self.causal_gate(x)
        
        x = x + mul32(gate_msa, self.attn(modulate32(self.norm1(x), shift_msa, scale_msa), idx_t=idx_t))
        x = x + mul32(gate_mlp, self.mlp(modulate32(self.norm2(x), shift_mlp, scale_mlp)))
        return x

class DiTBlock(CondTimestepBlock):
    """
    DiTBlock with causal attention mask
    """
    def __init__(self, hidden_size, cond_dims, num_heads, emb_shape, mlp_ratio=4.0, seq_len=5, pos=False, dtype=th.float32):
        super().__init__()
        self.norm1 = LayerNorm32(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, heads=num_heads, bias=True, seq_len=seq_len, dtype=dtype)
        self.norm2 = LayerNorm32(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            Linear32(cond_dims, 6 * hidden_size, bias=True)
        )

        self.pos_emb = pos
        if self.pos_emb:
            #self.pemb = positional_embedding(cond_dims, seq_len, seq_dim=1)
            self.ada_pos_emb = nn.Sequential(
                nn.SiLU(),
                Linear32(cond_dims, 3 * hidden_size, bias=True)
            )
            self.mlpe = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
            self.norm3 = LayerNorm32(hidden_size, elementwise_affine=False, eps=1e-6)

        self.emb_shape = emb_shape

    def forward(self, x, emb, **kwargs):
        # TODO: pass idx_t to the attention block
        if 'idx_t' in kwargs.keys():
            idx_t = kwargs['idx_t']
        else:
            idx_t = -2
        emb = einops.rearrange(emb, '(b t) c -> b t c', t = self.emb_shape[0])[:,0,:]
        emb = einops.repeat(emb, 'b c -> b c h w', h = self.emb_shape[2], w = self.emb_shape[3])
        emb = SwitchDims((1, self.emb_shape[1], self.emb_shape[2], self.emb_shape[3]))(emb)
        
        if self.pos_emb and 'pemb' in kwargs.keys():
            pos_emb = kwargs['pemb']
            shift_pe, scale_pe, gate_pe = self.ada_pos_emb(pos_emb).chunk(3, dim=2)
            x = x + mul32(gate_pe, self.mlpe(modulate32(self.norm3(x), shift_pe, scale_pe)))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(emb).chunk(6, dim=2)
        
        x = x + mul32(gate_msa, self.attn(modulate32(self.norm1(x), shift_msa, scale_msa), idx_t=idx_t))
        x = x + mul32(gate_mlp, self.mlp(modulate32(self.norm2(x), shift_mlp, scale_mlp)))
        return x

class MergeDims(nn.Module):
    """
    Change the dimensions of the input tensor from 'b t c h w -> (b t) c h w',
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        t, c, h, w = self.shape
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w', h=h, w=w, t=t, c=c)
        return x

class MergeDimsT(nn.Module):
    """
    Change the dimensions of the input tensor from '(b t) c h w -> b t c h w',
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape 

    def forward(self, x):
        t, c, h, w = self.shape
        x = einops.rearrange(x, '(b t) c h w -> b t c h w', h=h, w=w, t=t, c=c)
        return x

class SwitchDims(nn.Module):
    """
    Change the dimensions of the input tensor from '(b t) c h w -> (b h w) t c',
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape 

    def forward(self, x):
        t, c, h, w = self.shape
        x = einops.rearrange(x, '(b t) c h w -> (b h w) t c', h=h, w=w, t=t, c=c)
        return x
    
class SwitchDimsT(nn.Module):
    """
    Change the dimensions of the input tensor from '(b h w) t c -> (b t) c h w',
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape 

    def forward(self, x):
        t, c, h, w = self.shape
        x = einops.rearrange(x, '(b h w) t c -> (b t) c h w', h=h, w=w, t=t, c=c)
        return x
   
class ConnectorPlus(TimestepBlock):
    """
    the connector module for the two-stage training
    param x: the input tensor
    param c: the conditional tensor
    """
    def __init__(self, nf_x0, nf_x1, use_concat=False, use_checkpoint=False, use_attn_con=False):
        super().__init__()

        self.use_concat = use_concat
        if not use_concat:
            self.use_attention = use_attn_con
            if self.use_attention:
                factor = 6
                self.norm1 = normalization(nf_x1)
                self.attn = AttentionBlock(nf_x1, num_heads=4, use_checkpoint=use_checkpoint,)    
            else:
                factor = 3

            self.modulation = nn.Sequential(
                nn.SiLU(),
                nn.Conv2d(nf_x0, factor*nf_x1, kernel_size=1, padding=0),
            )

            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(in_features=nf_x1, hidden_features=4*nf_x1, act_layer=approx_gelu, drop=0, use_conv=True)
            self.norm2 = normalization(nf_x1)

    def forward(self, h, c, emb):

        # h is main path, c is conditional path
        
        if not self.use_concat:
            if self.use_attention:
                a_gate, a_shift, a_scale, m_gate, m_shift, m_scale = th.chunk(self.modulation(c), 6, dim=1)
            else:
                m_gate, m_shift, m_scale = th.chunk(self.modulation(c), 3, dim=1)

            if self.use_attention:
                h = h + a_gate*self.attn(self.norm1(h) * (a_scale + 1) + a_shift)
            
            h = h + m_gate*self.mlp(self.norm2(h) * (m_scale + 1) + m_shift)
        
        else:
            h = th.cat([h, c], dim=1)

        return h

class CausalUNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :param use_causal_gated: use causal gated convolutions for TSC feature
    :param dilations: dilations for the causal gated convolutions
    :param complicated_x0: if True, use the complicated x0 connection
    :param num_complicated: number of complicated x0 connections
    :param concat_cond: if True concatenate the TSC feature to the encoder's feature
        else use connector modules. if True doesn't support two-stages training
    """

    def __init__(
        self,
        image_size,
        seq_len,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_causal_gated=False,
        dilations=[1, 1, 2, 2, 2],
        complicated_x0=False,
        num_complicated=0,
        concat_cond=True, 
        pos_emb=False,    # positional embedding length, if 0, no positional embedding
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_causal_gated = use_causal_gated
        self.dilations = dilations
        self.complicated_x0 = complicated_x0
        self.concat_cond = concat_cond  
        self.pos_emb = pos_emb
        self.seq_len = seq_len

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            Linear32(model_channels, time_embed_dim),
            nn.SiLU(),
            Linear32(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self.x0_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )

        connect_x1_ch = []
        self.connectors = nn.ModuleList([])

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        self.x0_existed = [True]
        for level, mult in enumerate(channel_mult):
            
            for i in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch*2 if i == 0 and self.concat_cond else ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
                self.x0_existed.append(True if i == 0 else False)
                if i == 0:
                    connect_x1_ch.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
                self.x0_existed.append(False)

        res = 1
        chs = input_block_chans[0]
        prev_chs = chs
        for level, mult in enumerate(channel_mult):

            connect_shape = [seq_len, prev_chs, image_size//res, image_size//res]
            if complicated_x0:
                down_flag = False
            elif level == 0:
                down_flag = False
            else:
                down_flag = True

            layers = [
                SwitchDims(connect_shape),
                DiTBlock(prev_chs, time_embed_dim, num_heads, [seq_len, time_embed_dim, image_size//res, image_size//res], seq_len=seq_len, dtype=self.dtype) if not use_causal_gated 
                else DiTBlockCausalGate(prev_chs, time_embed_dim, num_heads, [seq_len, time_embed_dim, image_size//res, image_size//res], seq_len=seq_len, dtype=self.dtype),
                SwitchDimsT(connect_shape),
                ResBlock(
                        prev_chs,
                        time_embed_dim,
                        dropout,
                        out_channels=chs,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=down_flag,)
            ]
            if self.complicated_x0:
                connect_shape = [seq_len, chs, image_size//res, image_size//res]
                for _ in range(num_complicated):
                    layers = layers + [
                    SwitchDims(connect_shape),
                    DiTBlock(chs, time_embed_dim, num_heads, [seq_len, time_embed_dim, image_size//res, image_size//res], seq_len=seq_len, pos=pos_emb, dtype=self.dtype) if not use_causal_gated 
                    else DiTBlockCausalGate(chs, time_embed_dim, num_heads, [seq_len, time_embed_dim, image_size//res, image_size//res], seq_len=seq_len, dtype=self.dtype),
                    SwitchDimsT(connect_shape),
                    ResBlock(
                            chs,
                            time_embed_dim,
                            dropout,
                            out_channels=chs,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=False)]
                layers = layers + [
                    SwitchDims(connect_shape),
                    DiTBlock(chs, time_embed_dim, num_heads, [seq_len, time_embed_dim, image_size//res, image_size//res], seq_len=seq_len, pos=pos_emb, dtype=self.dtype) if not use_causal_gated 
                    else DiTBlockCausalGate(chs, time_embed_dim, num_heads, [seq_len, time_embed_dim, image_size//res, image_size//res], seq_len=seq_len, dtype=self.dtype),
                    SwitchDimsT(connect_shape),
                    ResBlock(
                            chs,
                            time_embed_dim,
                            dropout,
                            out_channels=chs,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=False if level == 0 else True,)]
            self.x0_blocks.append(
                TimestepEmbedSequential(*layers)
            )
            self._feature_size += int(mult * model_channels)
            self.connectors.append(ConnectorPlus(chs, chs, self.concat_cond, use_checkpoint, use_attn_con=True if res in attention_resolutions else False))
            prev_chs = chs
            chs = int(mult * model_channels)
            res = res if level == 0 else res * 2
            

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.x0_blocks.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.x0_blocks.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None, multistage=0, sampling=False, idx_t=-1):
        """
        forward_sample is for warmed up temporal sampling
        multistage = 0: one-stage training
        multistage = 1: the first stage in two-stage training
        multistage = 2: the second stage in two-stage training
        """
        if not sampling:
            return self.forward_normal(x, timesteps, y, multistage=multistage)
        else:
            return self.forward_sample(x, timesteps, y, idx_t)

    def forward_normal(self, x, timesteps, y=None, multistage=0):
        """
        Apply the model to an input batch when training.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        if self.concat_cond:
            assert multistage == 0, "concat_cond doesn't support two-stage training"

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []

        time_embed = freeze_module(self.time_embed) if multistage == 2 else self.time_embed
        semb = time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.pos_emb:
            pemb = positional_embedding(self.model_channels*4, self.seq_len, seq_dim=1, device=semb.device)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            semb = semb + self.label_emb(y)

        x0, x1 = th.chunk(x, 2, dim=-1)
        _, t_x, c_x, h_x, w_x = x1.shape
        if multistage !=1 :
            x0 = MergeDims((t_x, c_x, h_x, w_x))(x0)
        x1 = MergeDims((t_x, c_x, h_x, w_x))(x1)
        emb = einops.repeat(semb, 'b c -> (b t) c', t=t_x)

        h = x1.type(self.dtype)
        if multistage !=1 :
            x0 = x0.type(self.dtype)
        idx = 0
        for module, x0_e in zip(self.input_blocks, self.x0_existed):
            module = freeze_module(module) if multistage == 2 else module

            if x0_e and multistage !=1:
                
                x0 = self.x0_blocks[idx](x0, emb, **{"pemb": pemb} if self.pos_emb else {})
                idx += 1
                if idx > 1:
                    h = self.connectors[idx-2](h, x0, emb)
                h = module(h, emb)
            else:
                h = module(h, emb)

            hs.append(h)

        middle_block = freeze_module(self.middle_block) if multistage == 2 else self.middle_block
        h = middle_block(h, emb)

        for module in self.output_blocks:
            module = freeze_module(module) if multistage == 2 else module
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x1.dtype)

        out = freeze_module(self.out) if multistage == 2 else self.out
        return einops.rearrange(out(h), '(b t) c h w -> b t c h w', t = t_x)

    @th.no_grad()
    def forward_sample(self, x, timesteps, y=None, idx_t=-1):
        """
        Apply the model to an input batch (x0_idx_t, x) when sampling.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        semb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.pos_emb:
            pemb = positional_embedding(self.model_channels*4, self.seq_len, seq_dim=1, device=semb.device)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            semb = semb + self.label_emb(y)

        
        x1 = x[:, -1, ...]
        x0 = x[:, :-1, ...]
        _, t_x, c_x, h_x, w_x = x0.shape
        x0 = MergeDims((t_x, c_x, h_x, w_x))(x0)
        emb = einops.repeat(semb, 'b c -> (b t) c', t=t_x)

        h = x1.type(self.dtype)
        x0 = x0.type(self.dtype)
        idx = 0
        for module, con in zip(self.input_blocks, self.x0_existed):
            if con:
                x0 = self.x0_blocks[idx](x0, emb, **{"idx_t": idx_t, "pemb": pemb} if self.pos_emb else {"idx_t": idx_t})
                idx += 1
                if idx > 1:
                    x0_T = MergeDimsT((t_x, x0.size(1), x0.size(2), x0.size(3)))(x0)
                    h = self.connectors[idx-2](h, x0_T[:, idx_t, ...], semb)
                h = module(h, semb)
            else:
                h = module(h, semb)

            hs.append(h)

        h = self.middle_block(h, semb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, semb)
        h = h.type(x1.dtype)

        return einops.rearrange(self.out(h), '(b t) c h w -> b t c h w', t = 1)
