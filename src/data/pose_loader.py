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


import mmengine
import numpy as np
import os
import torch
from typing import List


class PoseLoader():
    def __init__(self, main_cfg: mmengine.Config):
        """
    
        Args:
            main_cfg (mmengine.Config): Configuration
    
        Attributes:
            predefined_traj (List): each element is a pose
                - pose (torch.Tensor, [4,4]): pre-defined trajectory pose. Format: camera-to-world, RDF
            
        """
        self.main_cfg = main_cfg

        ### load pre-defined trajectory ###
        if self.main_cfg.slam.use_traj_pose:
            self.predefined_traj = self.load_traj_pose()
    
    def load_traj_pose(self) -> List:
        """ load predefined trajectory poses
    
        Returns:
            poses (List): each element is a (torch.Tensor, [4,4]). Format: camera-to-world, RDF        
        """
        ##################################################
        ### Replica data
        ##################################################
        if self.main_cfg.general.dataset == 'Replica':
            traj_txt = os.path.join(self.main_cfg.slam.SLAMData_dir, 'traj.txt')
            with open(traj_txt, 'r') as f:
                lines = f.readlines()
                poses = [self.load_Replica_pose(line) for line in lines]
        
        ##################################################
        ### Matterport3D data
        ##################################################
        elif self.main_cfg.general.dataset == 'MP3D':
            traj_txt = os.path.join(self.main_cfg.slam.SLAMData_dir, 'traj.txt')
            with open(traj_txt, 'r') as f:
                lines = f.readlines()
                poses = [self.load_MP3D_pose(line) for line in lines]
        else:
            raise NotImplementedError
        return poses
    
    def load_Replica_pose(self, line: str) -> torch.Tensor:
        """ load Replica pose from trajectory file
    
        Args:
            line (str): pose data as txt line. Format: camera-to-world, RUB
    
        Returns:
            c2w (torch.Tensor, [4,4]): pose. Format: camera-to-world, RDF
        """
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4) 
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        c2w = torch.from_numpy(c2w).float()
        return c2w
    
    def load_MP3D_pose(self, line: str) -> torch.Tensor:
        """ load MP3D pose from trajectory file
    
        Args:
            line (str): pose data as txt line. Format: camera-to-world, RUB
    
        Returns:
            c2w (torch.Tensor, [4,4]): pose. Format: camera-to-world, RDF
        """
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4) 
        c2w = torch.from_numpy(c2w).float()
        return c2w
    
    def load_init_pose(self) -> torch.Tensor:
        """ load initial step's pose for SLAM
    
        Args:
            step (int): step number
    
        Returns:
            pose (torch.Tensor, [4,4]): initial pose
        """
        ##################################################
        ### Active Mapping
        ##################################################
        ### load from predefined pose ###
        if self.main_cfg.slam.enable_active_planning:
            if self.main_cfg.slam.use_traj_pose and self.predefined_traj is not None:
                c2w_slam = self.predefined_traj[0]
            elif self.main_cfg.slam.get("start_c2w", None) is not None:
                c2w_slam = torch.from_numpy(self.main_cfg.slam.start_c2w).float()
            else:
                raise NotImplementedError

        ##################################################
        ### Passive Mapping (Using predefined trajectory)
        ##################################################
        else:
            ### loading pre-defined poses ###
            c2w_slam = self.predefined_traj[0]


        ### force init pose to be within motion range (RRT_Z_range) ###
        if self.main_cfg.planner.get("rrt_z_range", None) is not None:
            rrt_z_range = self.main_cfg.planner.rrt_z_range
            min_z_metric = rrt_z_range[0] * self.main_cfg.planner.voxel_size + self.main_cfg.planner.bbox[2][0]
            max_z_metric = rrt_z_range[1] * self.main_cfg.planner.voxel_size + self.main_cfg.planner.bbox[2][0]
            c2w_slam[2, 3] = torch.clip(c2w_slam[2, 3], min_z_metric, max_z_metric)
    
        return c2w_slam
    
    def update_pose(self, planned_c2w: None, step: int) -> torch.Tensor:
        """ update pose. return planned_c2w or predefined trajectory.
        Depending on active/passive mapping
    
        Args:
            planned_c2w (torch.Tensor, [4,4]): planned camera pose. Format: camera-to-world, RDF
    
        Returns:
            torch.Tensor: next pose
        """
        ##################################################
        ### Active Mapping
        ##################################################
        if self.main_cfg.slam.enable_active_planning:
            return planned_c2w

        ##################################################
        ### Passive Mapping
        ##################################################
        else:
            return self.predefined_traj[step]


def habitat_pose_conversion(pose: np.ndarray, method:str = None) -> None:
    """ convert pose

    Args:
        pose (np.ndarray, [4,4]): original pose. Format: camera-to-world, RDF
        method (str)            : method

    Returns:
        new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB
        
    """
    if method == "coslam_replica2habitat":
        return coslam_replica2habitat(pose)
    elif method == "coslam_mp3d2habitat":
        return coslam_mp3d2habitat(pose)
    elif method == "coslam_naruto2habitat":
        return coslam_naruto2habitat(pose)
    else:
        raise NotImplementedError


def coslam_replica2habitat(pose: np.ndarray) -> np.ndarray:
    """ convert pose

    Args:
        pose (np.ndarray, [4,4]): original pose. Format: camera-to-world, RDF

    Returns:
        new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB
        
    """
    new_pose = pose.copy()
    new_pose[1:3, :] *= -1
    return new_pose


def coslam_mp3d2habitat(pose: np.ndarray) -> np.ndarray:
    """ convert pose

    Args:
        pose (np.ndarray, [4,4]): original pose. Format: camera-to-world, RDF

    Returns:
        new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB
        
    """
    T = np.array([[1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]])
    new_pose = T @ pose
    new_pose[1,3] = pose[2,3]
    new_pose[2,3] = -pose[1,3]
    return new_pose


def coslam_naruto2habitat(pose: np.ndarray) -> np.ndarray:
    """ convert pose

    Args:
        pose (np.ndarray, [4,4]): original pose. Format: camera-to-world, RDF

    Returns:
        new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB
        
    """
    new_pose = pose
    return new_pose
