# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Part et al. further modified the codebase

from functools import partial
import math
import logging

import torch
from torch import _assert
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class kcenter_model(nn.Module):
    """
    """
    def __init__(self, cfg):
        """
        """
        super().__init__()
        self.img_size = cfg.DATA.CROP_SIZE
        self.num_frames = cfg.DATA.NUM_FRAMES
        self.num_classes = cfg.MODEL.NUM_CLASSES

        # Model architecture hyperparameters
        self.patch_size = cfg.KCENTER_VIT.PATCH_SIZE
        self.embed_dim = cfg.KCENTER_VIT.EMBED_DIM
        self.depth = cfg.KCENTER_VIT.DEPTH
        self.num_heads = cfg.KCENTER_VIT.NUM_HEADS
        self.mlp_ratio = cfg.KCENTER_VIT.MLP_RATIO
        self.qkv_bias = cfg.KCENTER_VIT.QKV_BIAS
        self.drop_rate = cfg.KCENTER_VIT.DROP
        self.drop_path_rate = cfg.KCENTER_VIT.DROP_PATH

        self.num_pos_t = self.num_frames // 1 # >1 if tube embedding (removed in this release) is used.
        self.num_pos_s = self.img_size // self.patch_size

        self.total_sample_patches = cfg.KCENTER_VIT.TOTAL_SAMPLE_PATCHES

        self.kcenter_ws = cfg.KCENTER_VIT.KCENTER_SPATIAL_COEFFICIENT
        self.kcenter_wt = cfg.KCENTER_VIT.KCENTER_TEMPORAL_COEFFICIENT

        self.num_hybrid_frames = cfg.KCENTER_VIT.NUM_HYBRID_FRAMES
        self.num_kcenter_patches = self.total_sample_patches - self.num_hybrid_frames * (self.num_pos_s**2)
        assert self.num_kcenter_patches >= 0

        self.attn_drop_rate = cfg.KCENTER_VIT.ATTN_DROPOUT

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.cfg = cfg

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=3,
            embed_dim=self.embed_dim)

        T_index = torch.arange(self.num_pos_t)
        S_index = torch.arange(self.num_pos_s**2)
        TS_index = torch.stack(torch.meshgrid(T_index, S_index), dim=-1)
        if self.num_hybrid_frames > 0:
            hybrid_frame_indices = torch.linspace(0, self.num_pos_t-1, steps=self.num_hybrid_frames, dtype=torch.long)
            
            hybrid_TS_index = TS_index[hybrid_frame_indices]
            hybrid_TS_index = rearrange(hybrid_TS_index, 't hw c -> (t hw) c')
            hybrid_T_index, hybrid_S_index = torch.split(hybrid_TS_index, 1, dim=-1)

            self.register_buffer("hybrid_T_index", hybrid_T_index.squeeze(-1), persistent=False) # initialize self.hybrid_T_index
            self.register_buffer("hybrid_S_index", hybrid_S_index.squeeze(-1), persistent=False) # initialize self.hybrid_S_index
            self.register_buffer("hybrid_frame_indices", hybrid_frame_indices, persistent=False) # initialize self.hybrid_frame_indices
            if self.num_kcenter_patches:
                kcenter_frame_indices = torch.tensor([x for x in range(self.num_pos_t) if x not in hybrid_frame_indices], dtype=torch.long)
                self.register_buffer("kcenter_frame_indices", kcenter_frame_indices, persistent=False) # initialize self.kcenter_frame_indices
                TS_index = TS_index[kcenter_frame_indices]
                self.sampler = KCenterSampler(kcenter_ws=self.kcenter_ws,
                                              kcenter_wt=self.kcenter_wt,
                                              time_division=1,
                                              space_division=1,
                                              independent_time_segment=True,
                                              sort_indices=False)

        else:
            self.sampler = KCenterSampler(kcenter_ws=self.kcenter_ws,
                                          kcenter_wt=self.kcenter_wt,
                                          time_division=1,
                                          space_division=1,
                                          independent_time_segment=True,
                                          sort_indices=False)

        TS_index = rearrange(TS_index, 't hw c -> (t hw) c')
        T_index, S_index = torch.split(TS_index, 1, dim=-1)
        self.register_buffer("T_index", T_index.squeeze(-1), persistent=False) # initialize self.T_index
        self.register_buffer("S_index", S_index.squeeze(-1), persistent=False) # initialize self.S_index

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_pos_s**2 + 1, self.embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, self.num_pos_t, self.embed_dim))

        self.pos_drop = nn.Dropout(p=cfg.KCENTER_VIT.POS_DROPOUT)
        self.time_drop = nn.Dropout(p=cfg.KCENTER_VIT.POS_DROPOUT)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.drop_rate,
                attn_drop=self.attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.depth)])


        self.norm = norm_layer(self.embed_dim)
        self.pre_logits = nn.Identity() # originally defined for distilation in Pytorch vision models; not used here

        # Classifier head(s)
        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.init_weights()

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

    def prepare_tokens(self, x, t_index, s_index):
        B, N, DP  = x.shape # assume x.shape: B x N(=N1xN2) x dim(patch)
        patch_size = self.patch_embed.patch_size
        _assert(DP==patch_size[0]*patch_size[1]*3, f"Patch dimension wrong {patch_size[0]*patch_size[1]*3}")

        x = rearrange(x, 'b n (c p1 p2) -> (b n) c p1 p2', c=3, p1=patch_size[0], p2=patch_size[1])
        x = self.patch_embed(x) # assume x.shape: (B N) x 1 x C'
        x = rearrange(x, '(b n) () c -> b n c', b=B, n=self.total_sample_patches)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1) # for vectorized addtion of self.pos_embed

        cls_s_index = torch.cat([s_index.new_zeros(B, 1), s_index], dim=-1)
        pos_embed = self.pos_embed[0, cls_s_index]

        x = x + pos_embed
        x = self.pos_drop(x)

        # Time Embeddings
        cls_tokens, x = torch.split(x, [1, N], dim=1)

        time_embed = self.time_embed[0, t_index]

        x = x + time_embed
        x = self.time_drop(x)

        x = torch.cat([cls_tokens, x], dim=1)

        return x

    def forward_features(self, x, t_index, s_index):
        x = self.prepare_tokens(x, t_index, s_index)
        
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        
        return self.pre_logits(x[:, 0])

    @torch.no_grad()
    def preprocess_inputs(self, x, meta_dict):
        B, T, PH, PW, C = x.shape
        device = x.device
        num_input_frames = T
        num_patches = self.patch_embed.num_patches
        _assert(num_patches==PH*PW, f"Spatial patches mismatch. defined: {num_patches}, got: {PH*PW}.")
        
        patch_count = 0
        if self.num_hybrid_frames > 0:
            x_hybrid = x[:, self.hybrid_frame_indices, ..., :C-3]
            x_hybrid = x_hybrid.flatten(start_dim=1, end_dim=-2)
            hybrid_t_index = self.hybrid_T_index.unsqueeze(0).expand(B, -1)
            hybrid_s_index = self.hybrid_S_index.unsqueeze(0).expand(B, -1)

            num_sampling_patches = self.total_sample_patches - self.num_hybrid_frames * (self.num_pos_s**2)
            if num_sampling_patches > 0:
                x_kcenter = x[:, self.kcenter_frame_indices]

                if 'kpatch_record' in meta_dict:
                    kpatch_record = meta_dict['kpatch_record']
                    sampling_index = kpatch_record['k_center_index'].long()
                    assert sampling_index.size(1) == num_sampling_patches

                    # x_kcenter = rearrange(x_kcenter, 'b t h w c -> b (t h w) c')
                    x_kcenter = x_kcenter.reshape(B, -1, C)
                    sampling_index_ = sampling_index.unsqueeze(-1).expand(-1, -1, C-3)
                    x_kcenter = torch.gather(x_kcenter, dim=1, index=sampling_index_)
                else:
                    x_kcenter, sampling_index = self.sampler(x_kcenter, k=num_sampling_patches)
            
                kcenter_t_index, kcenter_s_index = self.T_index[sampling_index], self.S_index[sampling_index]

                x = torch.cat([x_hybrid, x_kcenter], dim=1)
                t_index = torch.cat([hybrid_t_index, kcenter_t_index], dim=1)
                s_index = torch.cat([hybrid_s_index, kcenter_s_index], dim=1)
            
            else:
                sampling_index = torch.empty(0, device=device)
                x = x_hybrid
                t_index = hybrid_t_index
                s_index = hybrid_s_index

        else:
            num_sampling_patches = self.total_sample_patches
            if 'kpatch_record' in meta_dict:
                kpatch_record = meta_dict['kpatch_record']
                sampling_index = kpatch_record['k_center_index'].long()
                assert sampling_index.size(1) == num_sampling_patches

                # x = rearrange(x, 'b t h w c -> b (t h w) c')
                x_kcenter = x_kcenter.reshape(B, -1, C)
                sampling_index_ = sampling_index.unsqueeze(-1).expand(-1, -1, C-3)
                x = torch.gather(x, dim=1, index=sampling_index_)
            else:
                x, sampling_index = self.sampler(x, k=num_sampling_patches)
            
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
    model_state_dict = model.state_dict()
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

        out_dict[k] = v

    return out_dict

@MODEL_REGISTRY.register()
def kcenter_vit(cfg):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    assert cfg.DATA.CHANNEL_STANDARD == 'rgb' # requires rgb standard
    model = kcenter_model(cfg)

    load_pretrained(
        model, cfg=cfg, num_classes=cfg.MODEL.NUM_CLASSES,
        filter_fn=checkpoint_filter_fn, strict=False)

    return model