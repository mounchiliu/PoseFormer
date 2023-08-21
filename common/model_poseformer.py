## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# similar to ViT
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)  # layer normalization,
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP中包含两层linear，这里定义为第一层Linear输出特征图的维度，第二层Linear的输出由out_features决定（没有指定则为in_features）
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)  # dim:32,

    def forward(self, x):
        # 1. layer normalization
        # 2. multi-head attention
        # 3. drop out several sample results
        # 4. residual connect
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # 1. layer normalization
        # 2. mlp
        # 3. drop path
        # 4. residual connect
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PoseTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3     #### output dimension is num_joints * 3

        ### spatial patch embedding： 论文中先进行spatial transformer从joints中提取features，
        # （一帧图像提取embed_dim_ratio * num_joints维的特征）
        # 再进行temporal transformer，使用连续num_frame的信息输入transformer，最终获取3D坐标
        # Spatial embedding
        # patch embedding, 一个patch对应一个joint，一个joint对应embed_dim_ratio维度的特征向量
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)  # patch embedding: appy Linear on patch to get feature of the patch
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))  # 根据 num_joints*embed_dim_ratio定义特征向量

        # Temporal embedding
        # temporal transformer中，直接使用spatial transformer输出的特征图作为embedded features
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)  # pos drop after doing patch and position embedding


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule， as illustrated in ViT pose

        # transformer blocks for spatial block
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)  # normal layer for spatial transformer output
        self.Temporal_norm = norm_layer(embed_dim)  # normal layer for Temporal transformer output

        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(   # head for get output
            nn.LayerNorm(embed_dim),  # layer norm
            nn.Linear(embed_dim , out_dim),  # linear layer to get output
        )


    def Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )  # after rearrange x: (b*f) * num_joints * channel
        # (Conv需要输入channel first， liner不需要)
        # torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        # in_features：指的是输入的二维张量的大小，即输入的[batch_size, size]中的size。
        # out_features:指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size，out_features]

        x = self.Spatial_patch_to_embedding(x)  # spatial patch embedding, 每一个joints embed一个32维vector
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)  # pos drop to avoid over-fitting

        # input to multiple transformer blocks
        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)  # layer norm after transformer, as coded in ViTPose  # b*num_f, 17, embed_dim_ratio
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)  # b, num_f, 17 * embed_dim_ratio
        return x

    def forward_features(self, x):
        b  = x.shape[0]  # b, num_f, 17 * embed_dim_ratio
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)  # pos drop to avoid over-fitting
        for blk in self.blocks:  # input to multiple transformer blocks
            x = blk(x)

        x = self.Temporal_norm(x)  # layer norm after transformer, as coded in ViTPose
        ##### x size [b, f, 17*emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = self.weighted_mean(x)  # x has multiple channels (relates to num_frame) as feature maps, weighted feature maps
        # x: b * 1 * (17 * emb_dim)
        x = x.view(b, 1, -1)  # x: b, 1, (17 * emb_dim)
        return x


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # 2d kps x:num_data(batch) * num_frames * 17 * 2 -> batch * 2 * num_frames * 17 (channel first for pytorch)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Spatial_forward_features(x)  # spatial transformer, get x as encoded features (b, num_f, 17 * embed_dim_ratio)
        x = self.forward_features(x)  # temporal transformer, x: batch * 1 * (17 * embed_dim_ratio)
        x = self.head(x)  # x: batch * 1 * (17*3), 3 for dim of 3D positions, get the 3D position of joints of the center frame

        x = x.view(b, 1, p, -1)  # x: batch, 1, 17 * 3

        return x

