"""
We have reused part of CoSLAM's code in this file and include our code for NARUTO.
For CoSLAM License, refer to https://github.com/HengyiWang/Co-SLAM/blob/main/LICENSE.

"""


''' Differences v.s. original CoSLAM scene_rep
- include uncertainty grid initalization
- include uncertainty as embedding
- add uncertainty map into rendering pipeline
'''

import numpy as np
import torch
import torch.nn as nn

from src.slam.coslam.model.decoder import ColorSDFNet_v2_Naruto as ColorSDFNet_v2

from third_parties.coslam.model.scene_rep import JointEncoding
from third_parties.coslam.model.decoder import ColorSDFNet
from third_parties.coslam.model.utils import sample_pdf, get_sdf_loss, mse2psnr, compute_loss, batchify


class JointEncodingNaruto(JointEncoding):
    def __init__(self, config, bound_box):
        super(JointEncoding, self).__init__()
        self.config = config
        self.bounding_box = bound_box
        self.get_resolution()
        self.get_encoding(config)
        self.get_decoder(config)

        if self.config['decoder']['pred_uncert'] or self.config['decoder']['uncert_grid']:
            self.act_uncertainty = nn.Softplus()
    
    def get_decoder(self, config):
        '''
        Get the decoder of the scene representation
        '''
        if not self.config['grid']['oneGrid']:
            self.decoder = ColorSDFNet(config, input_ch=self.input_ch, input_ch_pos=self.input_ch_pos)
        else:
            self.decoder = ColorSDFNet_v2(config, input_ch=self.input_ch, input_ch_pos=self.input_ch_pos)
        
        self.color_net = batchify(self.decoder.color_net, None)
        self.sdf_net = batchify(self.decoder.sdf_net, None)

    def get_uncert_grid(self, voxel_size):
        Nx = round((self.bounding_box[0, 1] - self.bounding_box[0, 0]).item() / voxel_size + 0.0005) + 1
        Ny = round((self.bounding_box[1, 1] - self.bounding_box[1, 0]).item() / voxel_size + 0.0005) + 1
        Nz = round((self.bounding_box[2, 1] - self.bounding_box[2, 0]).item() / voxel_size + 0.0005) + 1
        # Uncertainty initialize to 3
        self.uncert_grid = torch.nn.parameter.Parameter(torch.ones([Nx, Ny, Nz], device="cuda").float() * 3)
        self.cache_uncert = np.zeros([Nx, Ny, Nz], dtype=np.float32)
        return self.uncert_grid

    def calc_embedding(self, inputs):
        embed = self.embed_fn(inputs)
        if self.config['decoder']['uncert_grid']:
            grid = (inputs * 2 - 1)[None, None, None, ...]
            uncert = torch.nn.functional.grid_sample(self.uncert_grid[None, None, ...], grid, align_corners=False)
            embed = torch.cat([uncert.squeeze()[..., None], embed], dim=1)
        return embed

    def raw2outputs(self, raw, z_vals, white_bkgd=False):
        '''
        Perform volume rendering using weights computed from sdf.

        Params:
            raw: [N_rays, N_samples, 4]  if pred uncertainty then last channel 5
            z_vals: [N_rays, N_samples]
        Returns:
            rgb_map: [N_rays, 3]
            disp_map: [N_rays]
            acc_map: [N_rays]
            weights: [N_rays, N_samples]
        '''
        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        weights = self.sdf2weights(raw[..., 3], z_vals, args=self.config)
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]  # apply weights
        
        depth_map = torch.sum(weights * z_vals, -1)
        depth_var = torch.sum(weights * torch.square(z_vals - depth_map.unsqueeze(-1)), dim=-1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[...,None])
        
        if self.config['decoder']['pred_uncert'] or self.config['decoder']['uncert_grid']:
            uncert = self.act_uncertainty(raw[..., 4]) + 0.01  # 0.01 is the min uncertainty # (HY-000): uncertainty grid/net source
            uncert_map = torch.sum(weights * weights * uncert, -1)
            return rgb_map, disp_map, acc_map, weights, depth_map, depth_var, uncert_map

        return rgb_map, disp_map, acc_map, weights, depth_map, depth_var
    
    def query_sdf(self, query_points, return_geo=False, embed=False, return_uncert=False):
        '''
        Get the SDF value of the query points
        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            sdf: [N_rays, N_samples]
            geo_feat: [N_rays, N_samples, channel]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
  
        if embed:
            embedded = self.embed_fn(inputs_flat)  # without the possible uncertainty grid embedding
            return torch.reshape(embedded, list(query_points.shape[:-1]) + [embedded.shape[-1]])
        embedded = self.calc_embedding(inputs_flat)  # with the possible uncertainty grid embedding

        embedded_pos = self.embedpos_fn(inputs_flat)
        out = self.sdf_net(torch.cat([embedded, embedded_pos], dim=-1))
        sdf, geo_feat = out[..., :1], out[..., 1:]
        if self.config['decoder']['pred_uncert'] or self.config['decoder']['uncert_grid']:
            geo_feat, uncert = geo_feat[..., :-1], geo_feat[..., -1:]
            
        sdf = torch.reshape(sdf, list(query_points.shape[:-1]))
        
        if return_uncert:
            uncert = torch.reshape(uncert, list(query_points.shape[:-1]))
            sdf = torch.stack([sdf, uncert], -1)  # stack uncertainty to the sdf.

        if not return_geo:
            return sdf
        geo_feat = torch.reshape(geo_feat, list(query_points.shape[:-1]) + [geo_feat.shape[-1]])

        return sdf, geo_feat

    def query_color_sdf(self, query_points):
        '''
        Query the color and sdf at query_points.

        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            raw: [N_rays, N_samples, 4]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])

        embed = self.calc_embedding(inputs_flat)
        embe_pos = self.embedpos_fn(inputs_flat)
        if not self.config['grid']['oneGrid']:
            embed_color = self.embed_fn_color(inputs_flat)
            return self.decoder(embed, embe_pos, embed_color)
        return self.decoder(embed, embe_pos)
    
    def render_rays(self, rays_o, rays_d, target_d=None):
        '''
        Params:
            rays_o: [N_rays, 3]
            rays_d: [N_rays, 3]
            target_d: [N_rays, 1]

        '''
        n_rays = rays_o.shape[0]

        # Sample depth
        if target_d is not None:
            z_samples = torch.linspace(-self.config['training']['range_d'], self.config['training']['range_d'], steps=self.config['training']['n_range_d']).to(target_d) 
            z_samples = z_samples[None, :].repeat(n_rays, 1) + target_d
            z_samples[target_d.squeeze()<=0] = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], steps=self.config['training']['n_range_d']).to(target_d) 

            if self.config['training']['n_samples_d'] > 0:
                z_vals = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], self.config['training']['n_samples_d'])[None, :].repeat(n_rays, 1).to(rays_o)
                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            else:
                z_vals = z_samples
        else:
            z_vals = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], self.config['training']['n_samples']).to(rays_o)
            z_vals = z_vals[None, :].repeat(n_rays, 1) # [n_rays, n_samples]

        # Perturb sampling depths
        if self.config['training']['perturb'] > 0.:
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            z_vals = lower + (upper - lower) * torch.rand(z_vals.shape).to(rays_o)

        # Run rendering pipeline
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        raw = self.run_network(pts)  # [n_sample_ray 2048, n_sample_pts 43, SDF+RGB 4]
        
        if self.config['decoder']['pred_uncert'] or self.config['decoder']['uncert_grid']:
            rgb_map, disp_map, acc_map, weights, depth_map, depth_var, uncert_map = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])
        else:
            rgb_map, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])

        # Importance sampling
        if self.config['training']['n_importance'] > 0:

            rgb_map_0, disp_map_0, acc_map_0, depth_map_0, depth_var_0 = rgb_map, disp_map, acc_map, depth_map, depth_var

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.config['training']['n_importance'], det=(self.config['training']['perturb']==0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

            raw = self.run_network(pts)
            rgb_map, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])

        # Return rendering outputs
        ret = {'rgb' : rgb_map, 'depth' :depth_map, 
               'disp_map' : disp_map, 'acc_map' : acc_map, 
               'depth_var':depth_var,}
        ret = {**ret, 'z_vals': z_vals}

        ret['raw'] = raw

        if self.config['training']['n_importance'] > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['depth0'] = depth_map_0
            ret['depth_var0'] = depth_var_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)
        
        if self.config['decoder']['pred_uncert'] or self.config['decoder']['uncert_grid']:
            ret['uncert_map'] = uncert_map

        return ret
    
    def forward(self, rays_o, rays_d, target_rgb, target_d, global_step=0):
        '''
        Params:
            rays_o: ray origins (Bs, 3)
            rays_d: ray directions (Bs, 3)
            frame_ids: use for pose correction (Bs, 1)
            target_rgb: rgb value (Bs, 3)
            target_d: depth value (Bs, 1)
            c2w_array: poses (N, 4, 4) 
             r r r tx
             r r r ty
             r r r tz
        '''

        # Get render results
        rend_dict = self.render_rays(rays_o, rays_d, target_d=target_d)

        if not self.training:
            return rend_dict
        
        # Get depth and rgb weights for loss
        valid_depth_mask = (target_d.squeeze() > 0.) * (target_d.squeeze() < self.config['cam']['depth_trunc'])
        rgb_weight = valid_depth_mask.clone().unsqueeze(-1)
        rgb_weight[rgb_weight==0] = self.config['training']['rgb_missing']

        # Get render loss
        rgb_loss = compute_loss(rend_dict["rgb"]*rgb_weight, target_rgb*rgb_weight)
        psnr = mse2psnr(rgb_loss)
        depth_loss = compute_loss(rend_dict["depth"].squeeze()[valid_depth_mask], target_d.squeeze()[valid_depth_mask])

        if 'rgb0' in rend_dict:
            rgb_loss += compute_loss(rend_dict["rgb0"]*rgb_weight, target_rgb*rgb_weight)
            depth_loss += compute_loss(rend_dict["depth0"][valid_depth_mask], target_d.squeeze()[valid_depth_mask])
        
        # Get sdf loss
        z_vals = rend_dict['z_vals']  # [N_rand, N_samples + N_importance]
        sdf = rend_dict['raw'][..., 3]  # [N_rand, N_samples + N_importance]
        truncation = self.config['training']['trunc'] * self.config['data']['sc_factor']
        fs_loss, sdf_loss = get_sdf_loss(z_vals, target_d, sdf, truncation, 'l2', grad=None)         
        

        ret = {
            "rgb": rend_dict["rgb"],
            "depth": rend_dict["depth"],
            "rgb_loss": rgb_loss,
            "depth_loss": depth_loss,
            "sdf_loss": sdf_loss,
            "fs_loss": fs_loss,
            "psnr": psnr,
        }
        
        if 'uncert_map' in rend_dict:
            uncert_map = rend_dict['uncert_map']
            assert uncert_map.min() > 0
            uncert_map = uncert_map[valid_depth_mask]
            x = rend_dict["depth"].squeeze()[valid_depth_mask]
            y = target_d.squeeze()[valid_depth_mask]
            uncert_loss = torch.mean((1 / (2*(uncert_map+1e-9).unsqueeze(-1))) *((x - y) ** 2)) + 0.5*torch.mean(torch.log(uncert_map+1e-9))
            ret['uncert_loss'] = uncert_loss

        return ret