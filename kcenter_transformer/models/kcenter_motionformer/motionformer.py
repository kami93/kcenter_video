# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from . import vit_helper
from .vit_utils import trunc_normal_
from ..build import MODEL_REGISTRY
from ..patch_sampler.kcenter_sampler import KCenterSampler


class kcenter_model(nn.Module):
    """ KCenter Motionformer
    """

    def __init__(self, cfg):
        super().__init__()
        self.img_size = cfg.DATA.CROP_SIZE
        self.num_frames = cfg.DATA.NUM_FRAMES
        self.num_classes = cfg.MODEL.NUM_CLASSES

        # Model architecture hyperparameters
        self.patch_size = cfg.KCENTER_MOTIONFORMER.PATCH_SIZE
        self.embed_dim = cfg.KCENTER_MOTIONFORMER.EMBED_DIM
        self.depth = cfg.KCENTER_MOTIONFORMER.DEPTH
        self.num_heads = cfg.KCENTER_MOTIONFORMER.NUM_HEADS
        self.mlp_ratio = cfg.KCENTER_MOTIONFORMER.MLP_RATIO
        self.qkv_bias = cfg.KCENTER_MOTIONFORMER.QKV_BIAS
        self.drop_rate = cfg.KCENTER_MOTIONFORMER.DROP
        self.drop_path_rate = cfg.KCENTER_MOTIONFORMER.DROP_PATH

        self.num_pos_t = self.num_frames // 1 # >1 if tube embedding (removed in this release) is used.
        self.num_pos_s = self.img_size // self.patch_size

        self.total_sample_patches = cfg.KCENTER_MOTIONFORMER.TOTAL_SAMPLE_PATCHES

        self.kcenter_ws = cfg.KCENTER_MOTIONFORMER.KCENTER_SPATIAL_COEFFICIENT
        self.kcenter_wt = cfg.KCENTER_MOTIONFORMER.KCENTER_TEMPORAL_COEFFICIENT

        self.kcenter_dt = cfg.KCENTER_MOTIONFORMER.KCENTER_TEMPORAL_DIVISION
        assert self.total_sample_patches % self.kcenter_dt == 0

        self.num_hybrid_frames = cfg.KCENTER_MOTIONFORMER.NUM_HYBRID_FRAMES
        self.num_kcenter_patches = self.total_sample_patches - self.num_hybrid_frames * (self.num_pos_s**2)
        assert self.num_kcenter_patches >= 0

        self.attn_layer = cfg.KCENTER_MOTIONFORMER.ATTN_LAYER

        self.use_mlp = cfg.KCENTER_MOTIONFORMER.USE_MLP
        self.head_dropout = cfg.KCENTER_MOTIONFORMER.HEAD_DROPOUT

        self.attn_drop_rate = cfg.KCENTER_MOTIONFORMER.ATTN_DROPOUT
        self.head_act = cfg.KCENTER_MOTIONFORMER.HEAD_ACT

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.cfg = cfg

        # Patch Embedding
        self.patch_embed = vit_helper.PatchEmbed(
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
                                              space_division=1,
                                              independent_time_segment=True,
                                              sort_indices=False)

        else:
            self.sampler = KCenterSampler(kcenter_ws=self.kcenter_ws,
                                          kcenter_wt=self.kcenter_wt,
                                          time_division=self.kcenter_dt,
                                          space_division=1,
                                          independent_time_segment=True,
                                          sort_indices=False)

        TS_index = rearrange(TS_index, 't hw c -> (t hw) c')
        T_index, S_index = torch.split(TS_index, 1, dim=-1)
        self.register_buffer("T_index", T_index.squeeze(-1), persistent=False) # initialize self.T_index
        self.register_buffer("S_index", S_index.squeeze(-1), persistent=False) # initialize self.S_index

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.cls_token, std=.02)

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_pos_s**2 + 1, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        self.temp_embed = nn.Parameter(
            torch.zeros(1, self.num_pos_t, self.embed_dim)) # temp_embed is initialized to zeros.
        self.pos_drop = nn.Dropout(p=cfg.KCENTER_MOTIONFORMER.POS_DROPOUT)
        
        # Layer Blocks
        dpr = [x.item() for x in torch.linspace(
            0, self.drop_path_rate, self.depth)]
        if self.attn_layer == "divided":
            self.blocks = nn.ModuleList([
                vit_helper.DividedSpaceTimeBlock(
                    attn_type=self.attn_layer,
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop=self.drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(self.depth)
            ])
        elif self.attn_layer in ['joint', 'trajectory']:
            self.blocks = nn.ModuleList([
                vit_helper.Block(
                    attn_type=self.attn_layer,
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop=self.drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer
                )
                for i in range(self.depth)
            ])
        else:
            raise NotImplementedError()

        self.norm = norm_layer(self.embed_dim)

        # MLP head
        if self.use_mlp:
            if self.head_act == 'tanh':
                act = nn.Tanh()
            elif self.head_act == 'gelu':
                act = nn.GELU()
            elif self.head_act == 'relu':
                act = nn.ReLU()
            else:
                raise NotImplementedError()
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, self.embed_dim)),
                ('act', act),
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier Head
        self.head_drop = nn.Dropout(p=self.head_dropout)
        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            for a, i in enumerate(range(len(self.num_classes))):
                setattr(self, "head%d" % a, nn.Linear(self.embed_dim, self.num_classes[i]))
        else:
            self.head = (nn.Linear(self.embed_dim, self.num_classes)
                         if self.num_classes > 0 else nn.Identity())

        self.apply(self._init_weights) # Initialize weights (only for those in nn.Linear & nn.LayerNorm)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', 'pos_embed', 'temp_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = (nn.Linear(self.embed_dim, num_classes) if num_classes > 0
                     else nn.Identity())

    def forward_features(self, x, t_index, s_index):
        B, N, DP  = x.shape # assume x.shape: B x N(=N1xN2) x dim(patch)

        x = rearrange(x, 'b n (c p1 p2) -> (b n) c p1 p2', c=3, p1=self.patch_size, p2=self.patch_size)
        x = self.patch_embed(x) # assume x.shape: (B N) x 1 x C'
        x = rearrange(x, '(b n) () c -> b n c', b=B, n=self.total_sample_patches)
        
       # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        cls_embed = self.pos_embed[:, 0, :].unsqueeze(1).expand(B, -1, -1)
        tile_pos_embed = self.pos_embed[0, s_index]
        tile_temporal_embed = self.temp_embed[0, t_index]

        total_pos_embed = tile_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)
        x = x + total_pos_embed

        x = self.pos_drop(x)

        #  Encoding using transformer layers
        for blk in self.blocks:
            x = blk(
                x,
                seq_len=self.num_pos_s**2,
                num_frames=self.kcenter_dt
            )

        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)
        return x

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
        x = self.head_drop(x)
        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            output = []
            for head in range(len(self.num_classes)):
                x_out = getattr(self, "head%d" % head)(x)
                if not self.training:
                    x_out = torch.nn.functional.softmax(x_out, dim=-1)
                output.append(x_out)
            return output, meta_dict
        else:
            x = self.head(x)
            if not self.training:
                x = torch.nn.functional.softmax(x, dim=-1)
            return x, meta_dict

@MODEL_REGISTRY.register()
def kcenter_motionformer(cfg):
    assert cfg.DATA.CHANNEL_STANDARD == 'rgb'
    model = kcenter_model(cfg)

    vit_helper.load_pretrained(
        model, cfg=cfg, num_classes=cfg.MODEL.NUM_CLASSES, filter_fn=vit_helper._conv_filter, strict=False
    )

    return model
