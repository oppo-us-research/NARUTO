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


import torch
import numpy as np
from typing import Union, Tuple

import os, sys
sys.path.append(os.getcwd())

# from third_party.py360convert.py360convert.utils import equirect_uvgrid
# from third_party.py360convert.py360convert.utils import rotation_matrix as rotation_matrix_np
# from third_party.py360convert.py360convert.utils import erp2p_pose
# from third_party.py360convert.py360convert.utils import xyzpers
# from third_party.py360convert.py360convert.utils import persxyz
# from third_party.py360convert.py360convert.utils import xyz2uv as xyz2uv_np
# from third_party.py360convert.py360convert.utils import uv2unitxyz as uv2unitxyz_np
# from third_party.py360convert.py360convert.utils import uv2coor as uv2coor_np
# from third_party.py360convert.py360convert.utils import coor2uv as coor2uv_np

''' missed functions:
- xyzcube
- equirect_facetype
- sample_equirec
- cube_list2h
- cube_h2dict
- cube_dict2h
- cube_h2dice
- cube_dice2h
'''

def erp_uvgrid(h: int, 
               w: int, 
               rev_y: bool = False
               ) -> torch.Tensor:
    """ create uv grid for equirectangular map

    Args:
        h (int): output grid height
        w (int): output grid width
        rev_y (bool): reverse y direction so Y: Down

    Returns:
        uv (torch.Tensor, [H,W,2]): uv grid 
    """
    u = torch.linspace(-torch.pi, torch.pi, steps=w, dtype=torch.float)

    if rev_y:
        v = torch.linspace(-torch.pi, torch.pi, steps=h, dtype=torch.float) / 2
    else:
        v = torch.linspace(torch.pi, -torch.pi, steps=h, dtype=torch.float) / 2

    uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1)
    return uv


def rotation_matrix(rad: float, 
                    ax: torch.Tensor,
                    ) -> torch.Tensor:
    """create rotation matrix from rotation degree and axis

    Args:
        rad (float): rotation angle in rad
        ax (torch.Tensor, [3]): rotation axis
    
    Returns:
        R (torch.Tensor, [3,3]): rotation matrix
    """
    ### convert to Tensor ###
    rad = torch.tensor([rad])
    # ax = torch.tensor(ax)
    assert len(ax.shape) == 1 and ax.shape[0] == 3

    ### normalize axis vector ###
    ax = ax / torch.sqrt((ax**2).sum())
    
    R = torch.diag(torch.cos(rad).repeat(3))
    R = R + torch.outer(ax, ax) * (1.0 - torch.cos(rad))

    ax = ax * torch.sin(rad)
    R = R + torch.tensor([[0, -ax[2].item(), ax[1].item()],
                      [ax[2].item(), 0, -ax[0].item()],
                      [-ax[1].item(), ax[0].item(), 0]])

    return R


def erp2pers_pose(u_deg: float, 
                  v_deg: float,
                  in_rot_deg: float
                  ) -> torch.Tensor:
    """equirectangular to perspective pose

    Args:
        u_deg (float): horizontal degree for the center
        v_deg (float): vertical degree for the center
        in_rot_deg (float): rotation angle degree along z-axis

    Returns:
        pose (torch.Tensor, [4,4]): ERP2P pose

    """
    u = -u_deg * torch.pi / 180
    v = v_deg * torch.pi / 180
    in_rot = in_rot_deg * torch.pi / 180

    Rx = rotation_matrix(v, torch.tensor([1., 0., 0.]))
    Ry = rotation_matrix(u, torch.tensor([0., 1., 0.]))
    in_rot_ax = (torch.tensor([[0., 0., 1.0]]) @ Rx @ Ry)[0]
    Ri = rotation_matrix(in_rot, in_rot_ax)
    R = Rx @ Ry @ Ri

    pose = torch.eye(4)
    pose[:3, :3] = R

    return pose


def create_perspective_xyz(h_fov: float, 
                           v_fov: float, 
                           u: float, 
                           v: float, 
                           in_rot: float,
                           out_hw: Tuple[int, int], 
                           ) -> torch.Tensor:
    """create 3D point cloud (on unit sphere surface) for perspective viewpoint in the ERP coordinate system

    Args:
        h_fov (float): horizontal FoV in radian
        v_fov (float): vertical FoV in radian
        u (float): horizontal degree (radian) for the center
        v (float): vertical degree (radian) for the center
        in_rot (float): rotation angle (radian) along z-axis
        out_hw (tuple): output size [H,W]
    
    Returns:
        out (torch.Tensor, [H,W,3]): 3D point clouds
    """
    ### convert to torch.Tensor ###
    h_fov = torch.tensor(h_fov)
    v_fov = torch.tensor(v_fov)
    u = torch.tensor(u)
    v = torch.tensor(v)
    in_rot = torch.tensor(in_rot)

    ### intialize point cloud ###
    out = torch.ones((*out_hw, 3))

    ### Get unit plane ###
    x_max = torch.tan(h_fov / 2)
    y_max = torch.tan(v_fov / 2)
    x_rng = torch.linspace(-x_max, x_max, steps=out_hw[1])
    y_rng = torch.linspace(-y_max, y_max, steps=out_hw[0])
    out[:, :, :2] = torch.stack(torch.meshgrid(x_rng, -y_rng, indexing='xy'), -1)

    ### create corresponding rotation matrix ###
    Rx = rotation_matrix(v, torch.tensor([1., 0., 0.]))
    Ry = rotation_matrix(u, torch.tensor([0., 1., 0.]))
    in_rot_ax = (torch.tensor([[0., 0., 1.0]]) @ Rx @ Ry)[0]
    Ri = rotation_matrix(in_rot, in_rot_ax)

    ### transform point clouds ###
    out = out @ Rx @ Ry @ Ri
    return out


def rotate_3d_pts_to_perspective(xyz: torch.Tensor, 
                                 u: float, 
                                 v: float, 
                                 in_rot: float,
                                 ) -> torch.Tensor:
    """Transform spherical 3D points (xyz) to perspective viewpoint, defined by [u,v,in_rot]

    Args:
        xyz (torch.Tensor, [h,w,3]): 3D points
        u (float): center location (horizontal) in radian
        v (float): center location (vertical) in radian
        in_rot (float): in-rotation in radian
    
    Returns:
        out (torch.Tensor, [hw, 3, 1]): transformed 3D points
    """
    ### convert to Tensor ###
    device = xyz.device
    u = torch.tensor(u)
    v = torch.tensor(v)
    in_rot = torch.tensor(in_rot)

    ### create rotation matrice ###
    Rx = rotation_matrix(v, torch.tensor([1., 0., 0.]))
    Ry = rotation_matrix(u, torch.tensor([0., 1., 0.]))
    in_rot_ax = (torch.tensor([[0., 0., 1.0]]) @ Rx @ Ry)[0]
    Ri = rotation_matrix(in_rot, in_rot_ax)
    R = Rx @ Ry @ Ri
    R = torch.inverse(R).to(device)

    ### transform points ###
    xyz = xyz.reshape(-1, 3)
    out = xyz @ R
    return out.reshape(-1, 3, 1)


def xyz2uv(xyz: torch.Tensor) -> torch.Tensor:
    '''convert 3d points (xyz) to ERP uv (angle in radian)
    
    Args:
        xyz (torch.Tensor, [H,W,3]): 3d point cloud
    
    Returns:
        uv (torch.Tensor, [H,W,2]): ERP uv (angle in radian)
    '''
    x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
    try:
        u = torch.arctan2(x, z)
    except:
        device = x.device
        u = torch.from_numpy(np.arctan2(x.cpu().numpy(), z.cpu().numpy())).to(device)
    c = torch.sqrt(x**2 + z**2)
    try:
        v = torch.arctan2(y, c)
    except:
        device = y.device
        v = torch.from_numpy(np.arctan2(y.cpu().numpy(), c.cpu().numpy())).to(device)
    return torch.stack([u, v], dim=-1)


def uv2unitxyz(uv: torch.Tensor) -> torch.Tensor:
    """ convert ERP uv (angle in radian) to 3D ponit cloud of unit sphere

    Args:
        uv (torch.Tensor, [H,W,2]): ERP uv

    Returns:
        xyz (torch.Tensor, [H,W,3]): point cloud
    """
    u, v = uv[:, :, 0], uv[:, :, 1]
    y = torch.sin(v)
    c = torch.cos(v)
    x = c * torch.sin(u)
    z = c * torch.cos(u)

    return torch.stack([x, y, z], dim=-1)


def uv2coor(uv: torch.Tensor,
            hw: Tuple[int, int]
            ) -> torch.Tensor:
    """ convert ERP uv (angle in radian) to 2D image coordinates

    Args:
        uv (torch.Tensor, [h,w,2]): ERP uv
        hw (tuple): ERP size
    
    Returns:
        coorxy (torch.Tensor, [H,W,2]): 2D image coordinates, [x,y], ranging [0, H-1] or [0, W-1]
    """
    h, w = hw
    u, v = uv[:, :, 0], uv[:, :, 1]

    coor_x = (u / (2 * torch.pi) + 0.5) * w - 0.5
    coor_y = (-v / torch.pi + 0.5) * h - 0.5

    return torch.stack([coor_x, coor_y], dim=-1)


def coor2uv(coorxy: torch.Tensor) -> torch.Tensor:
    ''' convert ERP uv (angle in radian) to 2D image coordinates

    Args:
        coorxy (torch.Tensor, [h,w,2]): 2D image coordinates, [x,y]
    
    Returns:
        uv (torch.Tensor, [h,w,2]): ERP uv
    '''
    h, w, _ = coorxy.shape
    coor_x, coor_y = coorxy[:, :, 0], coorxy[:, :, 1]

    u = ((coor_x + 0.5) / w - 0.5) * 2 * torch.pi
    v = -((coor_y + 0.5) / h - 0.5) * torch.pi

    return torch.stack([u, v], dim=-1)


def projection(xyz: torch.Tensor, 
               h_fov: float, 
               v_fov: float, 
               w: int, 
               h: int, 
               out_hw: Tuple[int, int], 
               normalize: bool = False, 
               return_mask: bool = False
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project a set of 3D points to 2D, given the horizontal field of view, vertical field of view, and image size.

    Args:
        xyz (torch.tensor, [N,3,1]): 3D points
        h_fov (float): horizontal FoV
        v_fov (float): vertical FoV
        w (int): image width of perspective view
        h (int): image height of perspective view
        out_hw (tuple): output size
        normalize (bool): normalize to [-1,1] if True
        return_mask (bool): return valid projection mask if True
    
    Returns:
        uv (torch.tensor, [1,H,W,2]): sampling grid
        mask (torch.tensor, [H,W,1]): projection mask
    """
    ### create intrinsics ###
    K = torch.eye(3)
    fx = w / (2 * np.tan(h_fov/2))
    fy = h / (2 * np.tan(v_fov/2))
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = w / 2  
    K[1,2] = h / 2

    ### project to 2D ###
    uv = K @ xyz
    uv = uv.reshape(out_hw[0], out_hw[1], 3)
    uv[:, :, 0] /= uv[:, :, 2]
    uv[:, :, 1] /= uv[:, :, 2]
    if normalize:
        uv[:, :, 0] = uv[:, :, 0] / (w - 1) * 2 - 1
        uv[:, :, 1] = uv[:, :, 1] / (h - 1) * 2 - 1

    ### filter out invalid projections ###
    mask = (uv[:, :, 2] > 0) * (uv[:, :, 0].abs()<=1) * (uv[:, :, 1].abs()<=1)
    mask = mask.unsqueeze(2)
    uv = mask * uv
    
    uv = uv[:, :, :2].unsqueeze(0)
    if return_mask:
        return uv, mask
    else:
        return uv

##################################################
### Testing
##################################################
# ## erp_uvgrid ###
# torch_data = erp_uvgrid(100, 100).numpy()
# np_data = equirect_uvgrid(100, 100)
# diff = np.abs(torch_data - np_data).mean()
# print("diff: ", diff)

# ## rotation_matrix ###
# rad = 0.2
# ax = [1,0.5,1.2]
# torch_data = rotation_matrix(torch.tensor([rad]), ax)
# np_data = rotation_matrix_np(rad, ax)
# diff = np.abs(torch_data - np_data).mean()
# print("diff: ", diff)

## erp2p_pose ###
# u = 90
# v = 30
# in_rot = 20
# torch_data = erp2pers_pose(u, v, in_rot)
# np_data = erp2p_pose(u, v, np.deg2rad(in_rot))
# diff = np.abs(torch_data - np_data).mean()
# print("diff: ", diff)

# ## create_perspective_xyz ###
# h_fov = 0.1
# v_fov = 0.2
# u = 0.3
# v = 0.4
# out_hw = (10,10)
# in_rot = 0.
# torch_data = create_perspective_xyz(h_fov, v_fov, u, v, in_rot, out_hw)
# np_data = xyzpers(h_fov, v_fov, u, v, out_hw, in_rot)
# diff = np.abs(torch_data - np_data).mean()
# print("diff: ", diff)

# ## rotate_3d_pts_to_perspective ###
# xyz = torch.ones((10,10,3)).cuda()
# u = 0.1
# v = 0.2
# in_rot = 0.3
# torch_data = rotate_3d_pts_to_perspective(xyz, u, v, in_rot).cpu()
# np_data = persxyz(xyz.cpu(), u, v, in_rot)
# diff = np.abs(torch_data - np_data).mean()
# print("diff: ", diff)

# ## xyz2uv ###
# xyz = torch.ones((5,10,3))
# xyz[..., 1] *= 2
# xyz[..., 2] *= 3
# torch_data = xyz2uv(xyz)
# np_data = xyz2uv_np(xyz)
# diff = np.abs(torch_data - np_data).mean()
# print("diff: ", diff)

# ## uv2unitxyz ###
# uv = torch.ones((5,10,2))
# uv[..., 0] *= 2
# uv[..., 1] *= 3
# torch_data = uv2unitxyz(uv)
# np_data = uv2unitxyz_np(uv)
# diff = np.abs(torch_data - np_data).mean()
# print("diff: ", diff)

# ## uv2coor ###
# uv = torch.ones((5,10,2))
# uv[..., 0] *= 2
# uv[..., 1] *= 3
# torch_data = uv2coor(uv)
# np_data = uv2coor_np(uv, 5, 10)
# diff = np.abs(torch_data - np_data).mean()
# print("diff: ", diff)

