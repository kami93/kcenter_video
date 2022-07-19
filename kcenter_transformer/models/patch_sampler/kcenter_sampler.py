import torch
import torch.nn as nn

from typing import Optional

class KCenterSampler(nn.Module):
    def __init__(self, kcenter_ws=1.0, kcenter_wt=1.0, time_division=1, space_division=1, independent_time_segment=True, sort_indices=True):
        super().__init__()
        self.kcenter_ws = kcenter_ws
        self.kcenter_wt = kcenter_wt
        self.time_division = time_division
        self.space_division = space_division
        self.independent_time_segment = independent_time_segment
        self.sort_indices = sort_indices

        k_center_weight = torch.FloatTensor([self.kcenter_wt, self.kcenter_ws, self.kcenter_ws])
        self.register_buffer("k_center_weight", k_center_weight.unsqueeze(0).unsqueeze(0), persistent=False) # self.k_center_weight
        
    @torch.no_grad()
    def forward(self, x: torch.Tensor, k: int, init_: Optional[torch.Tensor]=None):
        """
        x: spatiotemporal torch.Tensor with shape (B, T, H, W, C)
        k: number of points to sample among the spatiotemporal coordinates of x. k <= T*H*W.
        init_: deterministic initial point, instead of torch.randint. Shape (B*self.time_division, )

        Important messages:
        1. This module currently supports only independently sampling in each divided time chunk.
        2. When self.space_division > 1, this model supports only the case where one and only one point is sampled per divided space chunk.
        In other words, (self.time_division ** 2) should equal to (k // self.time_divison).
        """
        # start = timer()
        B, T, H, W, C = x.shape
        device = x.device

        x = x.view(B, T*H*W, C)

        TT = T // self.time_division
        HH = H // self.space_division
        WW = W // self.space_division

        x_ = x
        if self.kcenter_ws != 1.0 or self.kcenter_wt != 1.0:
            pe = x_[..., -3:].clone()
            x_[..., -3:] = pe * self.k_center_weight

        if self.independent_time_segment:
            x_ = x_.view(B*self.time_division, TT*H*W, C)
            time_offset = torch.arange(self.time_division, device=device).view(1, self.time_division, 1) * TT
            b_index = torch.arange(B*self.time_division, dtype=torch.int64, device=device)
            k_ = k // self.time_division

            cdist = torch.cdist(x_, x_) # b x i==(t h w) x j==(t h w)
            diagonal = cdist.diagonal(dim1=1, dim2=2)
            diagonal[:] = -1.0

            distance = torch.full((B*self.time_division, TT*H*W), 5e4, device=device)
            distance_ = distance.view(B*self.time_division, TT, self.space_division, HH, self.space_division, WW).permute(0, 2, 4, 1, 3, 5).view(B*self.time_division, self.space_division*self.space_division, TT*HH*WW)
            nones = torch.tensor([-1.0], device=device).unsqueeze(-1).unsqueeze(-1).expand(B*self.time_division, -1, TT*HH*WW)

            hd_indices = torch.arange(self.space_division, device=device).repeat_interleave(W*HH).repeat(TT)
            wd_indices = torch.arange(self.space_division, device=device).repeat_interleave(WW).repeat(TT*H)
            hwd_indices = hd_indices*self.space_division + wd_indices
            hwd_indices_ = hwd_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, TT*HH*WW)

            if init_ is not None:
                assert(init_.size(0) == B*self.time_division)
                farthest = torch.remainder(init_, TT*H*W)
            else:
                farthest = torch.randint(0, TT*H*W, size=(B*self.time_division, ), dtype=torch.int64, device=device)

            patch_count = 1
            sampling_index = [farthest]
            while patch_count < k_:
                dist = cdist[b_index, farthest]
                distance = torch.minimum(dist, distance, out=distance)

                if self.space_division != 1:
                    hwd_index = hwd_indices_[farthest]
                    distance_.scatter_(dim=1, index=hwd_index, src=nones)

                farthest = distance.argmax(dim=-1)
                sampling_index.append(farthest)
                patch_count += 1

            sampling_index = torch.stack(sampling_index, dim=-1)

            if self.sort_indices:
                hwd_index = hwd_indices[sampling_index]
                argsort = hwd_index.argsort(-1)
                sampling_index = torch.gather(sampling_index, dim=-1, index=argsort)

            sampling_index = sampling_index.view(B, self.time_division, k_)
            
            index_offset = time_offset * H * W
            sampling_index = index_offset + sampling_index
            sampling_index = sampling_index.view(B, self.time_division*k_)

        else:
            raise NotImplementedError
        
        patches = torch.gather(x, dim=1, index=sampling_index.unsqueeze(-1).expand(-1, -1, C-3))

        return patches, sampling_index