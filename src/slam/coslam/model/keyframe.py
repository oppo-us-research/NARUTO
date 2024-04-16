"""
We have reused part of CoSLAM's code in this file and include our code for NARUTO.
For CoSLAM License, refer to https://github.com/HengyiWang/Co-SLAM/blob/main/LICENSE.

"""


''' Differences v.s. original CoSLAM implementation
- store ray in add_keyframe
'''

import random
import torch

from third_parties.coslam.model.keyframe import KeyFrameDatabase

class KeyFrameDatabaseNaruto(KeyFrameDatabase):
    def __init__(self, config, H, W, num_kf, num_rays_to_save, device) -> None:
        super(KeyFrameDatabaseNaruto, self).__init__(config, H, W, num_kf, num_rays_to_save, device)
    
    def sample_single_keyframe_rays(self, rays, option='random'):
        '''
        Sampling strategy for current keyframe rays
        '''
        if option == 'random':
            idxs = random.sample(range(0, self.H*self.W), self.num_rays_to_save)
        elif option == 'filter_depth':
            valid_depth_mask = (rays[..., -1] > 0.0) & (rays[..., -1] <= self.config["cam"]["depth_trunc"])
            rays_valid = rays[valid_depth_mask, :]  # [n_valid, 7]
            num_valid = len(rays_valid)
            idxs = random.sample(range(0, num_valid), min(num_valid, self.num_rays_to_save))

        else:
            raise NotImplementedError()
        rays = rays[:, idxs]
        return rays
    
    def add_keyframe(self, batch, filter_depth=False):
        '''
        Add keyframe rays to the keyframe database
        '''
        # batch direction (Bs=1, H*W, 3)
        rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        rays = rays.reshape(1, -1, rays.shape[-1])
        if filter_depth:
            rays = self.sample_single_keyframe_rays(rays, 'filter_depth')
        else:
            rays = self.sample_single_keyframe_rays(rays)
        
        if not isinstance(batch['frame_id'], torch.Tensor):
            batch['frame_id'] = torch.tensor([batch['frame_id']])

        self.attach_ids(batch['frame_id'])

        # Store the rays
        if rays.shape[1] ==  0:
            return
        while rays.shape[1] < self.num_rays_to_save:
            rays = torch.cat([rays, rays], dim=1)
        self.rays[len(self.frame_ids)-1] = rays[:, :self.num_rays_to_save, :]
