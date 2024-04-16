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

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict

from third_parties.coslam.datasets.utils import get_camera_rays
from third_parties.coslam.datasets.dataset import get_dataset, BaseDataset


def get_dataset_extra(config: Dict) -> Dataset:
    """Get extra dataset class that is not included in the original coslam from the config file.

    Args:
        config: dataset configuration

    Returns:
        Dataset: Dataset object
    """
    if config['dataset'] == "replica":
        dataset = ReplicaDataset
    elif config['dataset'] == "mp3d":
        dataset = MP3DDataset
    elif config['dataset'] == "NARUTO":
        dataset = NARUTODataset
    return dataset(config, 
                config['data']['datadir'], 
                trainskip=config['data']['trainskip'], 
                downsample_factor=config['data']['downsample'], 
                sc_factor=config['data']['sc_factor'])


class ReplicaDataset(BaseDataset):
    def __init__(self, cfg, basedir, trainskip=1, 
                 downsample_factor=1, translation=0.0, 
                 sc_factor=1., crop=0):
        super(ReplicaDataset, self).__init__(cfg)

        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop
        self.img_files = [""] * 20000
        self.depth_paths = [""] * 20000
        # self.img_files = sorted(glob.glob(f'{self.basedir}/results/frame*.jpg'))
        # self.depth_paths = sorted(
        #     glob.glob(f'{self.basedir}/results/depth*.png'))
        # self.load_poses(os.path.join(self.basedir, 'traj.txt'))
        
        self.rays_d = None
        self.tracking_mask = None
        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)
    
    def __len__(self):
        return self.num_frames

    
    def __getitem__(self, index):
        color_path = self.img_files[index]
        depth_path = self.depth_paths[index]

        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            raise NotImplementedError()
        if self.distortion is not None:
            raise NotImplementedError()

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor

        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))

        if self.downsample_factor > 1:
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.rays_d is None:
            self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        color_data = torch.from_numpy(color_data.astype(np.float32))
        depth_data = torch.from_numpy(depth_data.astype(np.float32))

        ret = {
            "frame_id": self.frame_ids[index],
            "c2w":  self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            "direction": self.rays_d,
        }

        return ret


class MP3DDataset(BaseDataset):
    def __init__(self, cfg, basedir, trainskip=1, 
                 downsample_factor=1, translation=0.0, 
                 sc_factor=1., crop=0):
        super(MP3DDataset, self).__init__(cfg)

        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop
        self.img_files = [""] * 20000
        self.depth_paths = [""] * 20000

        self.rays_d = None
        self.tracking_mask = None
        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)
    
    def __len__(self):
        return self.num_frames

    
    def __getitem__(self, index):
        color_path = self.img_files[index]
        depth_path = self.depth_paths[index]

        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            raise NotImplementedError()
        if self.distortion is not None:
            raise NotImplementedError()

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor

        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))

        if self.downsample_factor > 1:
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.rays_d is None:
            self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        color_data = torch.from_numpy(color_data.astype(np.float32))
        depth_data = torch.from_numpy(depth_data.astype(np.float32))

        ret = {
            "frame_id": self.frame_ids[index],
            "c2w":  self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            "direction": self.rays_d,
        }

        return ret


    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(len(self.img_files)):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w[:3, 3] *= self.sc_factor
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)

        
class NARUTODataset(MP3DDataset):
    def __init__(self, cfg, basedir, trainskip=1, 
                 downsample_factor=1, translation=0.0, 
                 sc_factor=1., crop=0):
        super(NARUTODataset, self).__init__(cfg, basedir, trainskip, 
                 downsample_factor, translation, 
                 sc_factor, crop)
