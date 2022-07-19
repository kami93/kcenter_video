# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition
from functools import partial
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from .helpers import named_apply, load_pretrained
from .vit_utils import trunc_normal_, lecun_normal_
from .vit_layers import PatchEmbed, Mlp, DropPath
from ..build import MODEL_REGISTRY
from ..patch_sampler.kcenter_sampler import KCenterSampler

_logger = logging.getLogger(__name__)

class DotProduct(nn.Module):
    """ Explicit dot product layer for pretty flops count printing.
    """
    def __init__(self, scale=None):
        super().__init__()
        self.scale = scale
    
    def forward(self, x, y):
        if self.scale is not None:
            x = x * self.scale
        out = x @ y

        return out

    def extra_repr(self) -> str:
        return 'scale={}'.format(
            self.scale
        )

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.scaled_dot_product = DotProduct(scale=head_dim ** -0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.dot_product = DotProduct()

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = self.scaled_dot_product(q, k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = self.dot_product(attn, v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.temporal_fc = nn.Linear(dim, dim)
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, T, N):
        """
        x (Tensor): shape (B, (1 + T N), C)
        T (int): input time length
        N (int): input num patches
        """
        init_cls_tokens, x = torch.split(x, [1, T*N], dim=1)

        # Temporal attention
        xt = rearrange(x, 'b (t n) c -> (b n) t c', t=T, n=N)
        xt = self.temporal_attn(self.temporal_norm1(xt))
        xt = self.drop_path(xt)
        xt = rearrange(self.temporal_fc(xt), '(b n) t c -> b (t n) c', t=T, n=N)

        x = x + xt

        # Spatial attention
        cls_token = init_cls_tokens.expand(-1, T, -1) # expand cls_token over time dimension
        cls_token = rearrange(cls_token, 'b t c -> (b t) () c')
        xs = rearrange(x, 'b (t n) c -> (b t) n c', t=T, n=N)

        xs = torch.cat([cls_token, xs], dim=1)
        xs = self.attn(self.norm1(xs))

        xs = self.drop_path(xs)

        cls_token, xs = torch.split(xs, [1, N], dim=1)
        cls_token = reduce(cls_token, '(b t) () c -> b () c', 'mean', t=T) # average cls tkn over time dimension
        xs = rearrange(xs, '(b t) n c -> b (t n) c', t=T, n=N)

        x = torch.cat([init_cls_tokens, x], dim=1) + torch.cat([cls_token, xs], dim=1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class kcenter_model(nn.Module):
    """
    """
    def __init__(self, cfg):
        super().__init__()
        self.img_size = cfg.DATA.CROP_SIZE
        self.num_frames = cfg.DATA.NUM_FRAMES
        self.num_classes = cfg.MODEL.NUM_CLASSES

        # Model architecture hyperparameters
        self.patch_size = cfg.KCENTER_TIMESFORMER.PATCH_SIZE
        self.embed_dim = cfg.KCENTER_TIMESFORMER.EMBED_DIM
        self.depth = cfg.KCENTER_TIMESFORMER.DEPTH
        self.num_heads = cfg.KCENTER_TIMESFORMER.NUM_HEADS
        self.mlp_ratio = cfg.KCENTER_TIMESFORMER.MLP_RATIO
        self.qkv_bias = cfg.KCENTER_TIMESFORMER.QKV_BIAS
        self.drop_rate = cfg.KCENTER_TIMESFORMER.DROP
        self.drop_path_rate = cfg.KCENTER_TIMESFORMER.DROP_PATH
        self.attn_drop_rate = cfg.KCENTER_TIMESFORMER.ATTN_DROPOUT

        self.num_pos_t = self.num_frames // 1 # >1 if tube embedding (removed in this release) is used.
        self.num_pos_s = self.img_size // self.patch_size

        self.total_sample_patches = cfg.KCENTER_TIMESFORMER.TOTAL_SAMPLE_PATCHES

        self.kcenter_ws = cfg.KCENTER_TIMESFORMER.KCENTER_SPATIAL_COEFFICIENT
        self.kcenter_wt = cfg.KCENTER_TIMESFORMER.KCENTER_TEMPORAL_COEFFICIENT

        self.kcenter_ds = cfg.KCENTER_TIMESFORMER.KCENTER_SPATIAL_DIVISION
        assert self.total_sample_patches % self.kcenter_ds == 0
        self.kcenter_dt = cfg.KCENTER_TIMESFORMER.KCENTER_TEMPORAL_DIVISION
        assert self.total_sample_patches % self.kcenter_dt == 0

        self.num_hybrid_frames = cfg.KCENTER_MOTIONFORMER.NUM_HYBRID_FRAMES
        self.num_kcenter_patches = self.total_sample_patches - self.num_hybrid_frames * (self.num_pos_s**2)
        assert self.num_kcenter_patches >= 0

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.cfg = cfg

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=3,
            embed_dim=self.embed_dim
        )

        # Pre-defined space & time index to help gathering positional embedding.
        T_index = torch.arange(self.num_pos_t)
        S_index = torch.arange(self.num_pos_s**2)
        TS_index = torch.stack(torch.meshgrid(T_index, S_index), dim=-1)
        if self.num_hybrid_frames > 0:
            hybrid_seg_indices = torch.linspace(0, self.kcenter_dt-1, steps=self.num_hybrid_frames, dtype=torch.long)
            kcenter_seg_indices = torch.tensor([x for x in range(self.kcenter_dt) if x not in hybrid_seg_indices], dtype=torch.long)

            self.num_pos_t_per_segment = self.num_pos_t // self.kcenter_dt
            self.num_patches_per_segment = self.num_pos_t_per_segment * (self.num_pos_s ** 2)

            hybrid_frame_index_in_segment = self.num_pos_t_per_segment // 2 # assume center sampling
            hybrid_frame_indices = self.num_pos_t_per_segment * hybrid_seg_indices + hybrid_frame_index_in_segment

            hybrid_TS_index = TS_index[hybrid_frame_indices]
            hybrid_TS_index = rearrange(hybrid_TS_index, 't hw c -> (t hw) c')
            hybrid_T_index, hybrid_S_index = torch.split(hybrid_TS_index, 1, dim=-1)

            self.register_buffer("hybrid_T_index", hybrid_T_index.squeeze(-1), persistent=False) # initialize self.hybrid_T_index
            self.register_buffer("hybrid_S_index", hybrid_S_index.squeeze(-1), persistent=False) # initialize self.hybrid_S_index
            self.register_buffer("hybrid_frame_indices", hybrid_frame_indices, persistent=False) # initialize self.hybrid_frame_indices
            if self.num_kcenter_patches:
                kcenter_frame_indices = []
                for kcenter_seg_index in kcenter_seg_indices:
                    kcenter_frame_indices.append(torch.arange(self.num_pos_t_per_segment*kcenter_seg_index, self.num_pos_t_per_segment*(kcenter_seg_index+1)))
                kcenter_frame_indices = torch.cat(kcenter_frame_indices)

                self.register_buffer("kcenter_frame_indices", kcenter_frame_indices, persistent=False) # initialize self.kcenter_frame_indices
                TS_index = TS_index[kcenter_frame_indices]
                self.sampler = KCenterSampler(kcenter_ws=self.kcenter_ws,
                                              kcenter_wt=self.kcenter_wt,
                                              time_division=self.kcenter_dt-self.num_hybrid_frames,
                                              space_division=self.kcenter_ds,
                                              independent_time_segment=True,
                                              sort_indices=True)

        else:
            self.sampler = KCenterSampler(kcenter_ws=self.kcenter_ws,
                                          kcenter_wt=self.kcenter_wt,
                                          time_division=self.kcenter_dt,
                                          space_division=self.kcenter_ds,
                                          independent_time_segment=True,
                                          sort_indices=True)

        TS_index = rearrange(TS_index, 't hw c -> (t hw) c')
        T_index, S_index = torch.split(TS_index, 1, dim=-1)
        self.register_buffer("T_index", T_index.squeeze(-1), persistent=False) # initialize self.T_index
        self.register_buffer("S_index", S_index.squeeze(-1), persistent=False) # initialize self.S_index

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_pos_s**2 + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=cfg.KCENTER_MOTIONFORMER.POS_DROPOUT)
        self.time_embed = nn.Parameter(
            torch.zeros(1, self.num_pos_t, self.embed_dim))
        self.time_drop = nn.Dropout(p=cfg.KCENTER_MOTIONFORMER.POS_DROPOUT)

        dpr = [x.item() for x in torch.linspace(
            0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer)
            for i in range(self.depth)])
        
        self.norm = norm_layer(self.embed_dim)
        self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.init_weights()

        i = 0
        for m in self.blocks.modules():
            m_str = str(m)
            if 'Block' in m_str:
                if i > 0:
                    nn.init.constant_(m.temporal_fc.weight, 0)
                    nn.init.constant_(m.temporal_fc.bias, 0)
                i += 1

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)

        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'time_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def prepare_tokens(self, x, t_index, s_index):
        B, N, DP = x.shape
        T = self.kcenter_dt
        PH = PW = self.kcenter_ds
        patch_size = self.patch_embed.patch_size

        x = rearrange(x, 'b (t ph pw) (c p1 p2) -> (b t) c (ph p1) (pw p2)', t=T, ph=PH, pw=PW, p1=patch_size[0], p2=patch_size[1])
        x = self.patch_embed(x)
        S = x.size(1) # assume x.shape: (B T) x S x C'
        
        cls_token = self.cls_token.expand(B*T, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1) # for vectorized addtion of self.pos_embed

        s_index = rearrange(s_index, 'b (t s) -> (b t) s', b=B, t=T, s=PH*PW)
        cls_s_index = torch.cat([s_index.new_zeros(B*T, 1), s_index], dim=-1)
        pos_embed = self.pos_embed[0, cls_s_index]

        x = x + pos_embed
        x = self.pos_drop(x)

        # Time Embeddings
        cls_tokens, x = torch.split(x, [1, S], dim=1)
        cls_tokens = cls_tokens[:B] # thorow-away overly expanded cls_tokens in the above lines 

        x = rearrange(x, '(b t) s c -> (b s) t c', t=T, s=S)
        time_embed = self.time_embed[0, t_index]
        time_embed = rearrange(time_embed, 'b (t s) c -> (b s) t c', t=T, s=S)
        x = x + time_embed
        x = self.time_drop(x)
        x = rearrange(x, '(b s) t c -> b (t s) c', t=T, s=S)
        
        x = torch.cat([cls_tokens, x], dim=1)

        return x, T, S

    def forward_features(self, x, t_index, s_index):
        x, T, S = self.prepare_tokens(x, t_index, s_index)
        for blk in self.blocks:
            x = blk(x, T, S)
        x = self.norm(x)

        return self.pre_logits(x[:, 0])

    @torch.no_grad()
    def preprocess_inputs(self, x, meta_dict=None):
        B, T, PH, PW, C = x.shape
        if self.num_hybrid_frames > 0:
            x_hybrid = x[:, self.hybrid_frame_indices, ..., :C-3]
            x_hybrid = x_hybrid.flatten(start_dim=1, end_dim=-2)
            hybrid_t_index = self.hybrid_T_index.unsqueeze(0).expand(B, -1)
            hybrid_s_index = self.hybrid_S_index.unsqueeze(0).expand(B, -1)

            if self.num_kcenter_patches > 0:
                x_kcenter = x[:, self.kcenter_frame_indices]
                x_kcenter, sampling_index = self.sampler(x_kcenter, k=self.num_kcenter_patches)
            
                kcenter_t_index, kcenter_s_index = self.T_index[sampling_index], self.S_index[sampling_index]

                x = torch.cat([x_hybrid, x_kcenter], dim=1)
                t_index = torch.cat([hybrid_t_index, kcenter_t_index], dim=1)
                s_index = torch.cat([hybrid_s_index, kcenter_s_index], dim=1)
            
            else:
                x = x_hybrid
                t_index = hybrid_t_index
                s_index = hybrid_s_index

        else:
            x, sampling_index = self.sampler(x, k=self.num_kcenter_patches)
            t_index, s_index = self.T_index[sampling_index], self.S_index[sampling_index]
        
        s_index = s_index + 1 # compensate for [cls] tokens

        return x, t_index, s_index

    def forward(self, x, meta_dict=None):
        if meta_dict is None:
            meta_dict = {}

        x, t_index, s_index = self.preprocess_inputs(x, meta_dict)    
        x = self.forward_features(x, t_index, s_index)
        x = self.head(x)

        return x

def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

def resize_time_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]

    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = posemb_grid
    gs_new = ntok_new.shape[1]
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    
    posemb_grid = posemb_grid.permute(0, 2, 1)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)

        elif k == 'time_embed' and v.shape != model.time_embed.shape:
            v = resize_time_embed(
                v, model.time_embed)
        
        elif 'blocks' in k and 'attn' in k:
            new_key = k.replace('attn','temporal_attn')
            if not new_key in state_dict:
                out_dict[new_key] = v

        elif 'blocks' in k and 'norm1' in k:
            new_key = k.replace('norm1','temporal_norm1')
            if not new_key in state_dict:
                out_dict[new_key] = v

        out_dict[k] = v

    return out_dict

@MODEL_REGISTRY.register()
def kcenter_timesformer(cfg):
    assert cfg.DATA.CHANNEL_STANDARD == 'rgb'
    model = kcenter_model(cfg)

    load_pretrained(
        model, cfg=cfg, num_classes=cfg.MODEL.NUM_CLASSES,
        filter_fn=checkpoint_filter_fn, strict=False)

    return model