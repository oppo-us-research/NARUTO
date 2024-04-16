"""
MIT License

Copyright (c) 2024 OPPO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import numpy as np
import torch
from typing import List, Dict, Tuple


class ActiveRaySampler():
    ''' 
    Original CoSLAM proposed a comprehensive sampling strategy by sampling rays from 
        global key frame database and current observation.
        (1) Global: randomly samples N (2048) samples from the keyframedatabase
        (2) Current: M = max(N/KeyFrameNum, min_pixels_cur). 
            i.e. Samples more from current frame when there are a few key frames.
                 Meanwhile, maintain a minimum number of samples from current observation 
                 when there are more key frame observations.
        The final set includes [N global, M current]
    
    With ActiveRaySampler, we retain the diversity of the original strategy but enhance it 
        by substituing a subset of the random samples (N') with uncertain ray samples.
        Selecting high uncertain rays from querying all ray samples is infeasible.
        To increase the chance of sampling high uncertain ray samples, we oversample 
        4 times of the original sample nubmer (i.e. 16384) from the keyframedatabase 
        and current keyframe first. 
        We replace a subset of global rays by K uncertain rays sampled from both global and current rays.
        i.e. [N-N' global, N' uncertain rays, M current]
        (1) Global: randomly samples N (8192) samples from the keyframedatabase
        (2) Current: sample_num = max(N/KeyFrameNum, min_pixels_cur'). Note that min_pixels_cur' = 4*min_pixels_cur
    '''
    def __init__(self, 
                 config: Dict = None,
                 num_uncert_sample: int = 500, 
                 oversample_mul: int = 4, 
                 ) -> None:
        """
    
        Args:
            num_uncert_sample (int): number of uncertain samples (K)
            oversample_mul (int)   : oversampling multiplier
    
        Returns:
            None
    
        Attributes:
            num_uncert_sample (int): number of uncertain samples (K)
            oversample_mul (int)   : oversampling multiplier
        """
        self.num_uncert_sample = num_uncert_sample
        self.oversample_mul = oversample_mul
        self.base_sample_num = config['mapping']['sample'] 
        self.oversample_num = self.base_sample_num * self.oversample_mul
        self.min_pixels_cur = config['mapping']['min_pixels_cur'] * self.oversample_mul
    
    def sample_rays(self,
                      rays_o    : torch.Tensor,
                      rays_d    : torch.Tensor,
                      target_s  : torch.Tensor,
                      target_d  : torch.Tensor,
                      idx_cur   : List,
                      uncert_vol: torch.Tensor,
                      bbox      : List
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
    
        Args:
            rays_o (torch.Tensor, [4N,3])     : rays origin
            rays_d (torch.Tensor, [4N,3])     : rays direction
            target_s (torch.Tensor, [4N,3])   : target color
            target_d (torch.Tensor, [4N,3])   : target depth
            idx_cur (List)                    : index of sampled rays in the current keyframe
            uncert_vol (torch.Tensor, [X,Y,Z]): Uncertainty volume
            model (JointEncoding)             : model
            bbox (List)                       : bounding box
    
        Returns:
            rays_o (torch.Tensor, [4N,3])  : rays origin
            rays_d (torch.Tensor, [4N,3])  : rays direction
            target_s (torch.Tensor, [4N,3]): target color
            target_d (torch.Tensor, [4N,3]): target depth
    
        """
        base_sample_num = self.base_sample_num

        ##################################################
        ### obtain points for uncertainty sampling
        ##################################################
        pts = rays_o + rays_d * target_d
        pts = pts[base_sample_num:-len(idx_cur)//self.oversample_mul]
        pts_loc = ((pts - torch.tensor(bbox)[:,0].cuda()) * 10).detach().cpu().numpy()
        pts_idx = pts_loc.round().astype(int)
        x_max, y_max, z_max = uncert_vol.shape[0] - 1,uncert_vol.shape[1] - 1, uncert_vol.shape[2] - 1
        pts_idx[:, 0] = np.clip(pts_idx[:, 0], 0, x_max)
        pts_idx[:, 1] = np.clip(pts_idx[:, 1], 0, y_max)
        pts_idx[:, 2] = np.clip(pts_idx[:, 2], 0, z_max)

        ##################################################
        ### Query uncertainty
        ##################################################
        pts_uncert = uncert_vol[pts_idx[:,0], pts_idx[:,1], pts_idx[:,2]]

        ##################################################
        ### recombine sampling rays
        ##################################################
        min_indices = np.argpartition(pts_uncert, self.num_uncert_sample, axis=None)[:self.num_uncert_sample]
        rays_o = torch.cat([
            rays_o[min_indices+base_sample_num], 
            rays_o[:base_sample_num-self.num_uncert_sample], 
            rays_o[-len(idx_cur)//self.oversample_mul:]]
            )
        rays_d = torch.cat([
            rays_d[min_indices+base_sample_num], 
            rays_d[:base_sample_num-self.num_uncert_sample], 
            rays_d[-len(idx_cur)//self.oversample_mul:]]
            )
        target_s = torch.cat([
            target_s[min_indices+base_sample_num], 
            target_s[:base_sample_num-self.num_uncert_sample], 
            target_s[-len(idx_cur)//self.oversample_mul:]]
            )
        target_d = torch.cat([
            target_d[min_indices+base_sample_num], 
            target_d[:base_sample_num-self.num_uncert_sample], 
            target_d[-len(idx_cur)//self.oversample_mul:]]
            )

        return rays_o, rays_d, target_s, target_d