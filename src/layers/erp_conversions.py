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
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

from src.layers.backprojection import Backprojection
from src.layers.projection import Projection
from src.layers.transformation3d import Transformation3D
from src.layers.c2e import C2E
from src.layers.erp_utils import create_perspective_xyz, xyz2uv, uv2coor, uv2unitxyz

class E2P(nn.Module):
    """ Layer to convert equirectangular image to perspective image
    """
    def __init__(self,
                 fov_deg   : Tuple[float, float],
                 u_deg     : float,
                 v_deg     : float,
                 in_rot_deg: float,
                 in_hw     : Tuple[int, int],
                 out_hw    : Tuple[int, int],
                 device    : torch.device,
                 mode      : str = 'bilinear'
                 ) -> None:
        """
        Args:
            fov_deg (Tuple[float, float]): horizontal and vertical field of view in degree
            u_deg (float)                : horizon viewing angle in range [-180, 180]
            v_deg (float)                : vertical viewing angle in range [-90, 90]
            in_rot_deg (float)           : rotation angle degree along z-axis
            in_hw (Tuple[int, int])      : input ERP image size
            out_hw (Tuple[int, int])     : output perspective view size
            device (torch.device)        : device
    
        Attributes:
            coor_xy (torch.Tensor, [1,h,w,2])
            
        """
        super(E2P, self).__init__()
        self.coor_xy = create_erp_coor(fov_deg, u_deg, v_deg, in_rot_deg, in_hw, out_hw, device)
        self.mode = mode

    def forward(self, e_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e_img (torch.Tensor, [N,C,H,W]): Equirectangular image
    
        Returns:
            pers_img (torch.Tensor, [N,C,h,w]): perspective image
        """
        N, C, H, W = e_img.shape

        coor_xy = self.coor_xy.repeat(N,1,1,1)
        pers_img = F.grid_sample(e_img, coor_xy, align_corners=True, padding_mode='border', mode=self.mode)
        return pers_img


class P2E_w_pose(nn.Module):
    """ Layer to convert perspective image to equirectangular image given the correspoinding rotation
    """
    def __init__(self,
                 fov_deg   : Tuple[float, float],
                 in_hw     : Tuple[int, int],
                 out_hw    : Tuple[int, int],
                 device    : torch.device,
                 mode      : str = 'bilinear'
                 ) -> None:
        """
        Args:
            fov_deg (Tuple[float, float]): horizontal and vertical field of view in degree
            in_hw (Tuple[int, int])      : input perspective image size
            out_hw (Tuple[int, int])     : output ERP view size
            device (torch.device)        : device
            model (str)                  : warping mode
    
        Attributes:
            coor_xy (torch.Tensor, [1,h,w,2])
            
        """
        super(P2E_w_pose, self).__init__()


        height, width = out_hw
        self.height = height
        self.width = width
        
        ### Prepare ERP 3D points  ###
        # generate regular grid
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        id_coords = torch.tensor(id_coords)

        # generate homogeneous pixel coordinates
        self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                 requires_grad=False)
        xy = torch.unsqueeze(
                        torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0)
                        , 0)
        xy = torch.cat([xy, self.ones], 1)
        xy = nn.Parameter(xy, requires_grad=False)
        
        a_ = xy[:, 0:1] / (width) * (2*torch.pi) # 0->2pi
        b_ = xy[:, 1:2] / (height) * (torch.pi)  - 0.5 * torch.pi # -pi/2->pi/2
        x = -torch.sin(a_) * torch.cos(b_)
        y = torch.sin(b_)
        z = -torch.cos(a_) * torch.cos(b_)
        ones = x * 0 + 1
        self.xyz = torch.cat([x,y,z, ones], 1).to(device)

        ### layers ###
        self.transform3d = Transformation3D().to(device)
        self.projection = Projection(out_hw[0], out_hw[1])

        ### prepare intrinsics ###
        h_fov = fov_deg[0] / 180 * torch.pi
        v_fov = fov_deg[1] / 180 * torch.pi
        self.K = torch.eye(4)
        h, w = in_hw
        self.h, self.w = in_hw
        fx = w / (2 * np.tan(h_fov/2))
        fy = h / (2 * np.tan(v_fov/2))
        self.K[0,0] = fx
        self.K[1,1] = fy
        self.K[0,2] = w / 2  
        self.K[1,2] = h / 2
        self.K = self.K.to(device).unsqueeze(0)

        self.out_hw = out_hw
        
        self.mode = mode
    
    def forward(self, p_img: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pers_img (torch.Tensor, [1,C,h,w]): perspective image
            R (torch.Tensor, [1,4,4]): rotation matrix E2P
    
        Returns:
            pano (torch.Tensor, [N,C,H,W]): Equirectangular image
        """
        ### transform the sphere to perspective view ###
        pers_xyz = self.transform3d(self.xyz, R) # 1, 4, H*W

        ### project the sphere points to the perspective image view ###
        pers_uv = self.projection(pers_xyz, self.K, normalized=False)
        h, w = self.h, self.w
        pers_uv[:, :, :, 0] = pers_uv[:, :, :, 0] / (w - 1) * 2 - 1
        pers_uv[:, :, :, 1] = pers_uv[:, :, :, 1] / (h - 1) * 2 - 1
        mask = (pers_xyz.reshape(1,4,self.height, self.width)[:, 2:3] > 0) * (pers_uv[:, :, :, 0].abs()<=1) * (pers_uv[:, :, :, 1].abs()<=1)

        ### warp perspective view to ERP ###
        pano = torch.nn.functional.grid_sample(p_img, pers_uv, align_corners=True, mode=self.mode)

        ### mask out invalid projection ###
        pano *= mask
        return pano


def create_erp_coor(
        fov_deg: Tuple[float, float], 
        u_deg: float, 
        v_deg: float, 
        in_rot_deg: float,
        in_hw: Tuple[int, int],
        out_hw: Tuple[int, int], 
        device: torch.device
        ) -> torch.Tensor:
    """

    Args:
        fov_deg (tuple)      : horizontal and vertical field of view in degree
        u_deg (float)        : horizon viewing angle in range [-180, 180]
        v_deg (float)        : vertical viewing angle in range [-90, 90]
        in_rot_deg (float)   : rotation angle degree along z-axis
        in_hw (tuple)        : input ERP image size
        out_hw (tuple)       : output perspective view size
        device (torch.device): device

    Returns:
        coor_xy (torch.Tensor, [1,h,w,2]): ERP coordinate
    """
    h, w = in_hw

    ### convert from degree to radian ###
    h_fov, v_fov = fov_deg[0] * torch.pi / 180, fov_deg[1] * torch.pi / 180
    in_rot = in_rot_deg * torch.pi / 180
    u = -u_deg * torch.pi / 180
    v = v_deg * torch.pi / 180
    
    ### get perspective view XYZ ###
    xyz = create_perspective_xyz(h_fov, v_fov, u, v, in_rot, out_hw).to(device)

    ### project xyz to equirectangular map coordinate ###
    uv = xyz2uv(xyz)
    coor_xy = uv2coor(uv, (h, w))
    
    ### normalize coordinate ###
    coor_xy[:, :, 0] = coor_xy[:, :, 0] / (w - 1) * 2 - 1
    coor_xy[:, :, 1] = coor_xy[:, :, 1] / (h - 1) * 2 - 1

    ### reshape to NHW2 ###
    coor_xy = coor_xy.unsqueeze(0)
    return coor_xy


def e2p(
          e_img     : torch.Tensor,
          fov_deg   : Tuple[float, float],
          u_deg     : float,
          v_deg     : float,
          in_rot_deg: float,
          out_hw    : Tuple[int, int],
          mode      : str = 'bilinear'
        ) -> torch.Tensor: 
    """
    Args:
        e_img (torch.Tensor, [H,W,C]): Equirectangular image
        fov_deg (Tuple[float, float]): horizontal and vertical field of view in degree
        u_deg (float)                : horizon viewing angle in range [-180, 180]
        v_deg (float)                : vertical viewing angle in range [-90, 90]
        in_rot_deg (float)           : rotation angle degree along z-axis
        out_hw (Tuple[int, int])     : output perspective view image size
        mode (str)                   : interpolation mode

    Returns
        pers_img (torch.Tensor, [h,w,c]): image with range [0, 1]
    """
    ### get parameters ###
    device = e_img.device
    e_img = e_img.unsqueeze(0).permute(0,3,1,2) # N,C,H,W
    _, c, h, w = e_img.shape
    
    ### generate ERP coordinates ###
    coor_xy = create_erp_coor(fov_deg, u_deg, v_deg, in_rot_deg, (h,w), out_hw, device)

    ### warp perspective view from ERP ###
    pers_img = F.grid_sample(e_img, coor_xy, align_corners=True, padding_mode='border', mode=mode)
    pers_img = pers_img[0].permute(1,2,0)

    return pers_img


def depth2dist(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Perspective depth to distance

    Args:
        depth (torch.Tensor, [N,1,H,W]): depth map
        K (torch.Tensor, [N,4,4])      : camera intrinsics

    Returns:
        dist (torch.Tensor, [N,1,H,W]): distance map
    """
    _, _, h, w = depth.shape
    backproj = Backprojection(h, w).to(depth.device)
    inv_K = torch.inverse(K)

    pts3d = backproj(depth, inv_K, img_like_out=True)
    dist = torch.norm(pts3d[:,:3], dim=1, keepdim=True)
    return dist


class ERPDepth2Dist(nn.Module):
    """ Layer to convert equirectangular image to perspective image
    """
    def __init__(self,
                 skybox_size: int,
                 pano_hw: Tuple,
                 device    : torch.device,
                 ) -> None:
        """
        Args:
            device (torch.device)        : device
    
        """
        super(ERPDepth2Dist, self).__init__()
        pano_h, pano_w = pano_hw

        ### skybox order ###
        ### order: FRBLUD ###
        u_degs = [0, 90, -180, -90, 0, 0]
        v_degs = [0, 0, 0, 0, 90, -90]

        K = torch.eye(4).to(device)
        K[0,0] = K[0,2] = K[1,1] = K[1,2] = skybox_size/2
        self.K = K.unsqueeze(0)

        ### layers ###
        self.c2e_layer = C2E(skybox_size, pano_h, pano_w).to(device)
        self.e2p_layers = []
        for i in range(6):
            self.e2p_layers.append(
                E2P(
                    fov_deg=(90, 90),
                    u_deg=u_degs[i],
                    v_deg=v_degs[i],
                    in_rot_deg=0,
                    in_hw=(pano_h, pano_w),
                    out_hw=(skybox_size, skybox_size),
                    device=device
                    )
            )

    
    def forward(self, erp_depth):
        """ Convert ERP depth map to ERP radial distance map

        Args:
            erp_depth (torch.Tensor, [H,W]): ERP depth map

        Returns:
            erp_dist (torch.Tensor, [H,W]): ERP radial distance map
        """
        pers_dist = []
        for i in range(6):
            ### get skybox depths ###
            pers_depth = self.e2p_layers[i](erp_depth)
            
            ### convert to radial distance ###
            distance_map = depth2dist(pers_depth, self.K)
            pers_dist.append(distance_map[0,0])

        ##################################################
        ###  convert to panoramic distance
        ##################################################
        cube_dists = torch.stack(pers_dist)
        erp_dist = self.c2e_layer(cube_dists.unsqueeze(0).unsqueeze(0))
        erp_dist = erp_dist[0,0,0]
        return erp_dist

def erp_depth_to_erp_dist(
          erp_depth  : Union[np.ndarray, torch.Tensor],
          device     : str = 'cuda',
          skybox_size: int = 1024
        ) -> torch.Tensor: 
    """ Convert ERP depth map to ERP radial distance map

    Args:
        erp_depth (Union[np.ndarray, torch.Tensor], [H,W]): ERP depth map

    Returns:
        erp_dist (Union[np.ndarray, torch.Tensor], [H,W]): ERP radial distance map
    """
    h = w = skybox_size
    pano_h, pano_w = erp_depth.shape

    ### initialize layers ###
    c2e_layer = C2E(skybox_size, pano_h, pano_w).to(device)

    ### convert depth to torch tensor ###
    if type(erp_depth) is np.ndarray:
        erp_depth = torch.from_numpy(erp_depth).unsqueeze(2).to(device).float()
    
    erp_dist = erp_depth.clone() # H,W,1

    ### order: FRBLUD ###
    u_degs = [0, 90, -180, -90, 0, 0]
    v_degs = [0, 0, 0, 0, 90, -90]
    
    pers_dist = []
    K = torch.eye(4).to(device)
    K[0,0] = K[0,2] = K[1,1] = K[1,2] = skybox_size/2
    K = K.unsqueeze(0)
    for i in range(6):
        ### get skybox depths ###
        pers_depth = e2p(erp_dist, [90, 90], u_degs[i], v_degs[i], 0, [h, w])
        
        ### convert to radial distance ###
        pers_depth = pers_depth.unsqueeze(0).permute(0,3,1,2) # 1,1,H,W
        distance_map = depth2dist(pers_depth, K)
        pers_dist.append(distance_map[0,0])

    ##################################################
    ###  convert to panoramic distance
    ##################################################
    cube_dists = torch.stack(pers_dist)
    erp_dist = c2e_layer(cube_dists.unsqueeze(0).unsqueeze(0))
    erp_dist = erp_dist[0,0,0]
    return erp_dist


##################################################
### Testing
##################################################
# fov_deg = (120, 90)
# u_deg = 0
# v_deg = 0
# out_hw = (512, 512)
# in_rot_deg = 0

# import cv2
# import numpy as np
# img = "data/replica_sim/apartment_1_dyn/multiview_3/13/reference/panorama/000000.png"
# pano = cv2.imread(img)
# h,w,_ = pano.shape
# device = torch.device('cuda')
# pano_tensor = torch.from_numpy(pano).float().to(device)
# e2p1 = E2P(fov_deg, u_deg, v_deg, in_rot_deg, (h,w), out_hw, device)
# pers1 = e2p1(pano_tensor.permute(2,0,1).unsqueeze(0))[0].permute(1,2,0).cpu().numpy()
# pers2 = e2p(pano_tensor, fov_deg, u_deg, v_deg, in_rot_deg, out_hw).cpu().numpy()

# from third_party.py360convert.py360convert.e2p import e2p_gpu
# from third_party.py360convert.py360convert.e2p import e2p as e2p_np
# pers_gpu_np = e2p_gpu(pano, fov_deg, u_deg, v_deg, out_hw)*255.0
# pers_np = e2p_np(pano, fov_deg, u_deg, v_deg, out_hw)

# # cv2.imwrite("pers1.png", pers1.astype(np.uint8))
# # cv2.imwrite("pers2.png", pers2.astype(np.uint8))
# # cv2.imwrite("pers3.png", pers_np.astype(np.uint8))

# diff1 = np.abs(pers1 - pers_np).max()
# diff2 = np.abs(pers2 - pers_np).max()
# diff3 = np.abs(pers_gpu_np - pers_np).max()
# print("diff1: ", diff1)
# print("diff2: ", diff2)
# print("diff3: ", diff3)


### P2E_w_pose ###
# p2e = P2E_w_pose(
#     (90, 90),
#     (512, 512),
#     (1024, 2048),
#     'cuda'
# )
# p_img = torch.ones(1,1,512,512).cuda()
# R = torch.eye(4).unsqueeze(0).cuda()
# pano = p2e(p_img, R)
# import matplotlib.pyplot as plt
# vis = pano[0,0].detach().cpu().numpy()
# plt.imshow(vis)
# plt.show()