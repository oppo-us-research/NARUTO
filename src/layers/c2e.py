"""
MIT License

Copyright (c) 2019 sunset

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


''' source code: https://github.com/sunset1995/py360convert 
We made simple modification of py360convert version to accept PyTorch data.
'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnFunc

from . import c2e_utils as utils

def cubemap_formatting(cubemap, cube_format='dice'):
    """

    Args:
        cubemap 

    Returns:
        cube_faces (np.array, [6,H,W,3])
        
    Attributes:
        
    """
    if cube_format == 'horizon':
        pass
    elif cube_format == 'list':
        cubemap = utils.cube_list2h(cubemap)
    elif cube_format == 'dict':
        cubemap = utils.cube_dict2h(cubemap)
    elif cube_format == 'dice':
        cubemap = utils.cube_dice2h(cubemap)
    else:
        raise NotImplementedError('unknown cube_format')
    assert len(cubemap.shape) == 3
    assert cubemap.shape[0] * 6 == cubemap.shape[1]

    cube_faces = np.stack(np.split(cubemap, 6, 1), 0)
    
    return cube_faces


class C2E(nn.Module):
    """Layer to convert cubemap into equirectangular panorama

    """
    def __init__(self, face_w, h, w):
        """
        Args:
            face_w (int): input cubemap width
            h (int): panorama height
            w (int): panorama width
        
        Attributes:
            grid (torch.tensor, [1, 1, H, W, 3])
        """
        super(C2E, self).__init__()

        ### generate grid ###
        uv = utils.equirect_uvgrid(h, w)
        u, v = np.split(uv, 2, axis=-1)
        u = u[..., 0]
        v = v[..., 0]

        # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
        tp = utils.equirect_facetype(h, w)
        coor_x = np.zeros((h, w))
        coor_y = np.zeros((h, w))

        for i in range(4):
            mask = (tp == i)
            coor_x[mask] = 0.5 * np.tan(u[mask] - np.pi * i / 2)
            coor_y[mask] = -0.5 * np.tan(v[mask]) / np.cos(u[mask] - np.pi * i / 2)

        mask = (tp == 4)
        c = 0.5 * np.tan(np.pi / 2 - v[mask])
        coor_x[mask] = c * np.sin(u[mask])
        coor_y[mask] = c * np.cos(u[mask])

        mask = (tp == 5)
        c = 0.5 * np.tan(np.pi / 2 - np.abs(v[mask]))
        coor_x[mask] = c * np.sin(u[mask])
        coor_y[mask] = -c * np.cos(u[mask])

        ### Final renormalize ###
        coor_x = (np.clip(coor_x, -0.5, 0.5) + 0.5) * (face_w - 1)
        coor_y = (np.clip(coor_y, -0.5, 0.5) + 0.5) * (face_w - 1)

        grid = torch.stack(
            [
                torch.from_numpy(coor_x),
                torch.from_numpy(coor_y),
                torch.from_numpy(tp), 
            ]
        )
        grid = torch.permute(grid, (1, 2, 0)).unsqueeze(0).unsqueeze(0)
        grid[..., 2] = grid[..., 2] / 5 * 2 - 1
        grid[..., 1] = grid[..., 1] / (face_w - 1) * 2 - 1
        grid[..., 0] = grid[..., 0] / (face_w - 1) * 2 - 1
        self.grid = nn.Parameter(grid.float(), requires_grad=False)

    def forward(self, cube_faces, mode='nearest'):
        """
        Args:
            cube_faces (torch.tensor, [N,C,6,s,s]): where D-dim includes 0F 1R 2B 3L 4U 5D
        Returns:
            pano (torch.tensor, [N,C,D,H,W]): panorama
        """
        batch_size = cube_faces.shape[0]
        pano = nnFunc.grid_sample(cube_faces, self.grid.repeat(batch_size,1,1,1,1), align_corners=True, mode=mode)
        return pano
