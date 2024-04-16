"""
We have reused part of CoSLAM's code in this file and include our code for NARUTO.
For CoSLAM License, refer to https://github.com/HengyiWang/Co-SLAM/blob/main/LICENSE.
"""

import json
import mmengine
import numpy as np
import os
import sys
import random
import torch
import torch.optim as optim
from typing import Dict, List

from src.utils.config_utils import load_config
from src.utils.general_utils import InfoPrinter
from src.slam.coslam.coslam_utils import query_point_sdf, get_map_volumes
from src.slam.slam_model import SlamModel

### modified coslam modules ###
from src.slam.coslam.model.scene_rep import JointEncodingNaruto as JointEncoding
from src.slam.coslam.model.keyframe import KeyFrameDatabaseNaruto as KeyFrameDatabase
from src.slam.coslam.datasets.dataset import get_dataset_extra as get_dataset
from src.slam.coslam.coslam_utils import extract_mesh

### original Co-SLAM modules ###
sys.path.append("third_parties/coslam")
from third_parties.coslam.coslam import CoSLAM
from third_parties.coslam.datasets.utils import get_camera_rays


class CoSLAMNaruto(SlamModel, CoSLAM):
    def __init__(self, 
                 main_cfg: mmengine.Config, info_printer: InfoPrinter
                 ) -> None:
        SlamModel.__init__(self, main_cfg, info_printer)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ### CoSLAM Config loading and save ###
        config = load_config(self.slam_cfg.room_cfg)
        config = self.override_config(config)
        self.config = config
        
        ### Save configuration ###
        self.info_printer("Saving config and script...", 0, self.__class__.__name__)
        save_path = os.path.join(self.main_cfg.dirs.result_dir, 'coslam')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, 'config.json'),"w", encoding='utf-8') as f:
            f.write(json.dumps(config, indent=4))

        ### volume configs ###
        self.voxel_size = self.slam_cfg.voxel_size

        ### Initialize camera parameters and rays ###
        self.init_cam_rays(config)

        ### initialization setting ###
        self.dataset = get_dataset(config)
        self.create_bounds()
        self.create_pose_data()
        self.get_pose_representation()
        self.model = JointEncoding(config, self.bounding_box).to(self.device)
        self.keyframeDatabase = self.create_kf_database(config) ### TODO: rm the requirement for ReplicaSLAM data ###

        ### initialize optimizers ###
        self.create_optimizer()
        if self.config['decoder']['uncert_grid']:
            self.init_uncert_grid_optim(voxel_size=self.voxel_size)
        
        ### Initialize Active Ray Sampler ###
        if self.config['mapping']['active_ray']:
            self.init_active_ray_sampler()
            
    def create_kf_database(self, config):  
        '''
        Create the keyframe database
        '''
        num_kf = int(self.dataset.num_frames // self.config['mapping']['keyframe_every'] + 1)  
        print('#kf:', num_kf)
        print('#Pixels to save:', self.dataset.num_rays_to_save)
        return KeyFrameDatabase(config, 
                                self.dataset.H, 
                                self.dataset.W, 
                                num_kf, 
                                self.dataset.num_rays_to_save, 
                                self.device)

    def override_config(self, cfg: Dict) -> Dict:
        """ override configs 
        """
        ### output_dir ###
        cfg["data"]["output"] = os.path.join(self.main_cfg.dirs.result_dir, "coslam")
        cfg["data"]["datadir"] = self.slam_cfg.SLAMData_dir

        ### experiement setup ###
        cfg['mapping']['active_planning'] = self.slam_cfg.enable_active_planning
        cfg['mapping']['active_ray'] = self.slam_cfg.enable_active_ray

        ### visualization ###
        cfg['mesh']['vis'] = self.main_cfg.visualizer.get('mesh_vis_freq', cfg['mesh']['vis'])

        return cfg
    
    def init_active_ray_sampler(self):
        """ initialize Active Ray Sampler

        Attributes:
            active_ray_sampler(ActiveRaySampler): active ray sampler
            
        """
        from src.slam.coslam.active_ray_sampler import ActiveRaySampler
        self.active_ray_sampler = ActiveRaySampler(
            config = self.config,
            num_uncert_sample = self.slam_cfg.act_ray_num_uncert_sample,
            oversample_mul = self.slam_cfg.act_ray_oversample_mul,
        )
        return

    def init_cam_rays(self, cfg: Dict) -> None:
        """ initialize camera parameters and camera rays
    
        Args:
            cfg (Dict): CoSLAM config
    
        Attributes:
            H (int)                       : image height
            W (int)                       : image width
            fx (float)                    : focal length (x)
            fy (float)                    : focal length (y)
            cx (float)                    : principal point (x)
            cy (float)                    : principal point (y)
            rays_d (torch.Tensor, [H,W,3]): camera rays

        """
        self.H, self.W = cfg['cam']['H']//cfg['data']['downsample'],\
            cfg['cam']['W']//cfg['data']['downsample']
        self.fx, self.fy =  cfg['cam']['fx']//cfg['data']['downsample'],\
             cfg['cam']['fy']//cfg['data']['downsample']
        self.cx, self.cy = cfg['cam']['cx']//cfg['data']['downsample'],\
             cfg['cam']['cy']//cfg['data']['downsample']
        self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)
        
    def create_pose_data(self):
        '''
        Create the pose data
        '''
        self.est_c2w_data = {}
        self.est_c2w_data_rel = {}
        # self.load_gt_pose() 
  
    def get_loss_from_ret(self, ret, rgb=True, sdf=True, depth=True, fs=True, uncert=True, smooth=False):
        '''Get the training loss, includes uncertainty loss
        '''
        loss = 0
        if rgb:
            loss += self.config['training']['rgb_weight'] * ret['rgb_loss']
        if depth:
            loss += self.config['training']['depth_weight'] * ret['depth_loss']
        if sdf:
            loss += self.config['training']['sdf_weight'] * ret["sdf_loss"]
        if fs:
            loss +=  self.config['training']['fs_weight'] * ret["fs_loss"]
        
        if smooth and self.config['training']['smooth_weight']>0:
            loss += self.config['training']['smooth_weight'] * self.smoothness(self.config['training']['smooth_pts'], 
                                                                                  self.config['training']['smooth_vox'], 
                                                                                  margin=self.config['training']['smooth_margin'])
        if uncert and (self.config['decoder']['pred_uncert'] or self.config['decoder']['uncert_grid']):
            loss += self.config['training']['uncert_weight'] * ret['uncert_loss']
        
        return loss             

    def first_frame_mapping(self, batch, n_iters=100):
        '''
        First frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        
        '''        
        self.info_printer("First frame mapping...", 
                          self.step, self.__class__.__name__)
        c2w = batch['c2w'][0].to(self.device)
        self.est_c2w_data[0] = c2w
        self.est_c2w_data_rel[0] = c2w

        self.model.train()

        # Training
        if self.config['decoder']['uncert_grid']:
            self.uncert_optim.zero_grad()
        for i in range(n_iters):
            self.map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])
            
            # indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            indice_h, indice_w = indice % (self.dataset.H), torch.div(indice, (self.dataset.H), rounding_mode="trunc")
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)  # camera translation (vector to origin)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)  # ray direction vector in world coords

            # Forward
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.map_optimizer.step()
        if self.config['decoder']['uncert_grid']:
            self.uncert_optim.step()
        
        # First frame will always be a keyframe
        self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
        
        self.info_printer("First frame mapping done", 
                          self.step, self.__class__.__name__)
        return ret, loss


    
    def freeze_model(self):
        '''
        Freeze the model parameters
        '''
        for param in self.model.embed_fn.parameters():
            param.require_grad = False
        
        for param in self.model.decoder.parameters():
            param.require_grad = False
    
    def init_uncert_grid_optim(self, voxel_size = 0.1):
        """ Initialzie Uncertainty Grid optimizer
        """
        self.uncert_optim = torch.optim.Adam(params=[self.model.get_uncert_grid(voxel_size)], lr=1)

    
    def global_BA(self, batch, cur_frame_id):
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        '''
        pose_optimizer = None

        # all the KF poses: 0, 5, 10, ...
        poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id, self.config['mapping']['keyframe_every'])])
        
        # frame ids for all KFs, used for update poses after optimization
        frame_ids_all = torch.tensor(list(range(0, cur_frame_id, self.config['mapping']['keyframe_every'])))

        if len(self.keyframeDatabase.frame_ids) < 2 or self.config['tracking']['disable']:
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]
            poses_all = torch.cat([poses_fixed, current_pose], dim=0)
        
        else:
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]

            if self.config['mapping']['optim_cur']:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(torch.cat([poses[1:], current_pose]))
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

            else:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(poses[1:])
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)
        
        # Set up optimizer
        self.map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()
        if self.config['decoder']['uncert_grid']:
            self.uncert_optim.zero_grad()
        
        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1])

        for i in range(self.config['mapping']['iters']):

            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]

            ##################################################
            ### update sample number and min_pixels_cur
            ##################################################
            if self.config['mapping']['active_ray']:
                sample_num = self.active_ray_sampler.oversample_num
                min_pixels_cur = self.active_ray_sampler.min_pixels_cur
            else:
                sample_num = self.config['mapping']['sample']
                min_pixels_cur = self.config['mapping']['min_pixels_cur']

            
            rays, ids = self.keyframeDatabase.sample_global_rays(sample_num)

            #TODO: Checkpoint...

            ##################################################
            ### filter current keyframe samples using depth measurement.
            ###     valid samples: within a depth range
            ##################################################
            num_rays_to_sample_cur = max(sample_num // len(self.keyframeDatabase.frame_ids), min_pixels_cur)
            if not self.config['mapping']['filter_depth']:
                idx_cur = random.sample(range(0, self.dataset.H * self.dataset.W), num_rays_to_sample_cur)
            else:
                cur_valid_depth = (current_rays[..., -1] > 0.0) & (current_rays[..., -1] <= self.config["cam"]["depth_trunc"])
                cur_rays_valid = current_rays[cur_valid_depth, :]  # [n_valid, 7]
                cur_num_valid = len(cur_rays_valid)
                idx_cur = random.sample(range(0, cur_num_valid), min(cur_num_valid, num_rays_to_sample_cur))
            
            
            ##################################################
            ### Combine global and current rays
            ##################################################
            current_rays_batch = current_rays[idx_cur, :]

            rays = torch.cat([rays, current_rays_batch], dim=0) # N, 7
            ids_all = torch.cat([torch.div(ids, self.config['mapping']['keyframe_every'], rounding_mode='trunc'), -torch.ones((len(idx_cur)))]).to(torch.int64)


            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            
            ##################################################
            ### Active Ray Sampling (uncertainty-aware sampling)
            ##################################################
            if self.config['mapping']['active_ray']:
                    rays_o, rays_d, target_s, target_d = self.active_ray_sampler.sample_rays(
                        rays_o,
                        rays_d,
                        target_s,
                        target_d,
                        idx_cur,
                        self.cached_uncert,
                        # self.model.cache_uncert,
                        self.config['mapping']['bound']
                )
                    
            ##################################################
            ### Model forward, loss computation, and optimization
            ##################################################
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)

            loss = self.get_loss_from_ret(ret, smooth=True)
            
            loss.backward(retain_graph=True)
            
            if (i + 1) % self.config["mapping"]["map_accum_step"] == 0:
               
                if (i + 1) > self.config["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print('Wait update')
                self.map_optimizer.zero_grad()

            if pose_optimizer is not None and (i + 1) % self.config["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                # get SE3 poses to do forward pass
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans)
                pose_optim = pose_optim.to(self.device)
                # So current pose is always unchanged
                if self.config['mapping']['optim_cur']:
                    poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
                
                else:
                    current_pose = self.est_c2w_data[cur_frame_id][None,...]
                    # SE3 poses

                    poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)


                # zero_grad here
                pose_optimizer.zero_grad()
            
            if self.config['decoder']['uncert_grid'] and (i + 1) % 5 == 0:
                self.uncert_optim.step()
                self.uncert_optim.zero_grad()
        
        if pose_optimizer is not None and len(frame_ids_all) > 1:
            for i in range(len(frame_ids_all[1:])):
                self.est_c2w_data[int(frame_ids_all[i+1].item())] = self.matrix_from_tensor(cur_rot[i:i+1], cur_trans[i:i+1]).detach().clone()[0]
        
            if self.config['mapping']['optim_cur']:
                print('Update current pose')
                self.est_c2w_data[cur_frame_id] = self.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]
 
    def create_optimizer(self):
        '''
        Create optimizer for mapping
        '''
        trainable_parameters = [{'params': self.model.decoder.parameters(), 'weight_decay': 1e-6, 'lr': self.config['mapping']['lr_decoder']},
                                {'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
    
        if not self.config['grid']['oneGrid']:
            trainable_parameters.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})
        
        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))
    
    def save_mesh(self, 
                  i            : int  = None,
                  voxel_size  : float = 0.05,
                  suffix       : str  = "",
                  mesh_savedir: str   = None
                  ) -> None:
        """ save mesh

        Args:
            i (int)           : step
            voxel_size (float): voxel size. Unit: meter
            suffix (str)      : suffix to the filename
            mesh_savedir (str): directory to save mesh
        
        Returns:
            None
        """
        ### create mesh dir and get mesh_savepath ###
        if mesh_savedir is None:
            mesh_savedir = os.path.join(self.main_cfg.dirs.result_dir, 'coslam', "mesh")
        os.makedirs(mesh_savedir, exist_ok=True)
        mesh_name = f"mesh_{i:04}{suffix}.ply" if i is not None else f"mesh_{self.step:04}{suffix}.ply"
        mesh_savepath = os.path.join(mesh_savedir, mesh_name)

        ### get color render func ###
        if self.config['mesh']['render_color']:
            color_func = self.model.render_surface_color
        else:
            color_func = self.model.query_color

        ### extract mesh ###
        extract_mesh(self.model.query_sdf, 
                        self.config, 
                        self.bounding_box, 
                        color_func=color_func, 
                        marching_cube_bound=self.marching_cube_bound, 
                        voxel_size=voxel_size, 
                        mesh_savepath=mesh_savepath)      

    def save_uncert_mesh(self, 
                         i            : int  = None,
                         voxel_size  : float = 0.05,
                         suffix       : str  = "",
                         mesh_savedir: str   = None,
                         ) -> None:
        """ save mesh

        Args:
            i (int)           : step
            voxel_size (float): voxel size. Unit: meter
            suffix (str)      : suffix to the filename
            mesh_savedir (str): directory to save mesh
        
        Returns:
            None
        """
        ### create mesh dir and get mesh_savepath ###
        if mesh_savedir is None:
            mesh_savedir = os.path.join(self.main_cfg.dirs.result_dir, 'coslam', "uncert_mesh")
        os.makedirs(mesh_savedir, exist_ok=True)
        mesh_name = f"mesh_{i:04}{suffix}.ply" if i is not None else f"mesh_{self.step:04}{suffix}.ply"
        mesh_savepath = os.path.join(mesh_savedir, mesh_name)

        ### extract mesh ###
        extract_mesh(self.model.query_sdf, 
                        self.config, 
                        self.bounding_box, 
                        color_func=None, 
                        marching_cube_bound=self.marching_cube_bound, 
                        voxel_size=voxel_size, 
                        render_uncert=True,
                        mesh_savepath=mesh_savepath)  
    
    def save_ckpt(self, 
                  i     : int = None,
                  suffix: str = ""
                  ) -> None:
        """ save checkpoint

        Args:
            i (int)           : step
            suffix (str)      : suffix to the filename
        
        Returns:
            None
        """
        ### create ckpt dir and get ckpt_savepath ###
        ckpt_savedir = os.path.join(self.main_cfg.dirs.result_dir, 'coslam', "checkpoint")
        os.makedirs(ckpt_savedir, exist_ok=True)
        ckpt_name = f"ckpt_{i:04}{suffix}.pt" if i is not None else f"ckpt_{self.step:04}{suffix}.pt"
        ckpt_savepath = os.path.join(ckpt_savedir, ckpt_name)
        
        ### save checkpoint ###
        save_dict = {'pose': self.est_c2w_data,
                    'pose_rel': self.est_c2w_data_rel,
                    'model': self.model.state_dict()}
        torch.save(save_dict, ckpt_savepath)

    def predict_sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
    
        Args:
            points (torch.Tensor, [N,K,3]): 3D points
    
        Returns:
            sdf (torch.Tensor, [N,K,1])
    
        """
        sdf = query_point_sdf(
            self.model.query_sdf,
            points,
            self.config,
            self.bounding_box
        )
        return sdf
        
    def online_recon_step(self, 
                          i    : int,
                          color: torch.Tensor,
                          depth: torch.Tensor,
                          c2w  : torch.Tensor
                          ) -> List:
        ''' Run one step of the co-slam process.

        Args:
            i                            : Current frame id (int): step
            color (torch.Tensor, [H,W,3]): color
            depth (torch.Tensor, [H,W])  : depth map
            c2w (torch.Tensor, [4,4])    : pose. Format: RUB camera-to-world
        
        Returns:
            uncert_sdf_vols (List)
                - uncert_vol (np.ndarray, [X,Y,Z]): uncertainty volume
                - sdf_vol (np.ndarray, [X,Y,Z])   : SDF volume
        '''
        ##################################################
        ### Initialization
        ##################################################
        ### load data ###
        batch = {}
        batch['frame_id'] = torch.tensor([i])
        batch['c2w'] = c2w.unsqueeze(0)
        batch['rgb'] = color.unsqueeze(0)
        batch['depth'] = depth.unsqueeze(0)
        batch['direction'] = self.rays_d.unsqueeze(0)
        
        ### Initialize map volumes ###
        uncert_sdf_vols = None

        ### save mesh ###
        if i % self.config['mesh']['vis']==0: # self.config['mesh']['vis']: visualization frequency. Now: 500
            self.info_printer("Extract and save mesh",
                              self.step, self.__class__.__name__)
            self.save_mesh(i, voxel_size=self.config['mesh']['voxel_eval'])

        ##################################################
        ### First frame mapping
        ##################################################
        if i == 0:
            self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
            self.info_printer("Get map volumes (UncertaintyVolume, SDFVolume)",
                              self.step, self.__class__.__name__)
            uncert_sdf_vols = get_map_volumes(self.model.query_sdf, 
                            self.bounding_box, 
                            voxel_size=self.voxel_size,
            )
        
        ##################################################
        ### Tracking and Mapping
        ##################################################
        else:
            ##################################################
            ### Tracking
            ##################################################
            if self.config['tracking']['disable']:
                self.est_c2w_data[i] = batch['c2w'][0].cuda()
            else:
                self.info_printer("Running tracking",
                              self.step, self.__class__.__name__)
                if self.config['tracking']['iter_point'] > 0:
                    self.tracking_pc(batch, i)
                self.tracking_render(batch, i)

            ##################################################
            ### Global BA
            ##################################################
            if i%self.config['mapping']['map_every']==0:
                self.info_printer("Running global BA",
                              self.step, self.__class__.__name__)
                self.global_BA(batch, i)

                self.info_printer("Get map volumes (UncertaintyVolume, SDFVolume)",
                              self.step, self.__class__.__name__)
                uncert_sdf_vols = get_map_volumes(self.model.query_sdf, 
                            self.bounding_box, 
                            voxel_size=self.voxel_size,
                )
                
            ##################################################
            ### Add keyframe data
            ##################################################
            if i % self.config['mapping']['keyframe_every'] == 0:
                self.info_printer(f"Add keyframe-{self.step} to KeyframeDatabase",
                                  self.step, self.__class__.__name__)
                self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
        
        
        ##################################################
        ### cached uncertainty for Active Ray Sampling
        ##################################################
        if self.config['mapping']['active_ray'] and uncert_sdf_vols is not None:
            self.cached_uncert = uncert_sdf_vols[0]
        return uncert_sdf_vols
