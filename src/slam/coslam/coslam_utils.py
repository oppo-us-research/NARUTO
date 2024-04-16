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


import marching_cubes as mcubes
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import trimesh

from third_parties.coslam.utils import getVoxels, get_batch_query_fn

@torch.no_grad()
def query_point_sdf(query_fn, query_pts, config, bounding_box):
    """
    Args:
        query_fn (): sdf query function
        query_pts (torch.Tensor, [N,K,3])
        config (): 
        bounding_box ():
    Returns:
        sdf (torch.Tensor, [N,K,1])
    """
    query_pts = (query_pts - bounding_box[:, 0]) / (bounding_box[:, 1] - bounding_box[:, 0])
    embed = query_fn(query_pts, embed=True)
    embed = torch.sum(torch.abs(embed), dim=2)
    
    if config['decoder']['pred_uncert'] or config['decoder']['uncert_grid']:
        sdf = query_fn(query_pts, embed=False, return_uncert=True)
        sdf, uncert = sdf[..., 0], sdf[..., 1]
    else:
        sdf = query_fn(query_pts, embed=False)
    
    return sdf

@torch.no_grad()
def get_map_volumes(query_fn, bounding_box, voxel_size):
    """ get map data (Uncertainty Volume and SDF volume)

    Args:
        query_fn                           : query function
        bounding_box (torch.Tensor, [3, 2]): bounding box
        voxel_size (float)                 : voxel size

    Returns:
        map_vols (List):
        - uncert_vol (torch.Tensor, [X,Y,Z]): Uncertainty Volume
        - sdf_vol (torch.Tensor, [X,Y,Z])   : SDF Volume
    """
    ##################################################
    ### get query pts
    ##################################################
    x_min, y_min, z_min = bounding_box[:, 0]
    x_max, y_max, z_max = bounding_box[:, 1]

    tx, ty, tz = getVoxels(x_max, x_min, y_max, y_min, z_max, z_min, voxel_size)
    query_pts = torch.stack(torch.meshgrid(tx, ty, tz, indexing='ij'), -1).to(torch.float32).to(bounding_box.device)

    query_pts = (query_pts - bounding_box[:, 0]) / (bounding_box[:, 1] - bounding_box[:, 0])

    ##################################################
    ### query SDF and Uncertainty
    ##################################################
    embed = query_fn(query_pts, embed=True)
    embed = torch.sum(torch.abs(embed), dim=3)
    
    sdf = query_fn(query_pts, embed=False, return_uncert=True)
    sdf, uncert = sdf[..., 0], sdf[..., 1]
    uncert_map = torch.nn.functional.softplus(uncert) + 0.01

    ### only consider uncertainty on surface ###
    mask = (sdf >= 0) * (sdf < 0.5) 
    uncert_map[torch.logical_not(mask)] = 0

    return [uncert_map.cpu().numpy().copy(), sdf.cpu().numpy().copy()]


@torch.no_grad()
def extract_mesh(
        query_fn, 
        config, 
        bounding_box, 
        marching_cube_bound=None, 
        color_func = None, 
        voxel_size=None, 
        resolution=None, 
        isolevel=0.0, 
        scene_name='', 
        mesh_savepath='',
        render_uncert=True,
        ) -> trimesh.Trimesh:
    '''
    Extracts mesh from the scene model using marching cubes (Adapted from NeuralRGBD)
    '''
    # Query network on dense 3d grid of points
    if marching_cube_bound is None:
        marching_cube_bound = bounding_box

    x_min, y_min, z_min = marching_cube_bound[:, 0]
    x_max, y_max, z_max = marching_cube_bound[:, 1]

    tx, ty, tz = getVoxels(x_max, x_min, y_max, y_min, z_max, z_min, voxel_size, resolution)
    query_pts = torch.stack(torch.meshgrid(tx, ty, tz, indexing='ij'), -1).to(torch.float32)

    
    sh = query_pts.shape
    flat = query_pts.reshape([-1, 3])
    bounding_box_cpu = bounding_box.cpu()

    if config['grid']['tcnn_encoding']:
        flat = (flat - bounding_box_cpu[:, 0]) / (bounding_box_cpu[:, 1] - bounding_box_cpu[:, 0])

    fn = get_batch_query_fn(query_fn, device=bounding_box.device)

    chunk = 1024 * 64
    raw = [fn(flat, i, i + chunk).cpu().data.numpy() for i in range(0, flat.shape[0], chunk)]
    
    raw = np.concatenate(raw, 0).astype(np.float32)
    raw = np.reshape(raw, list(sh[:-1]) + [-1])
    

    print('Running Marching Cubes')
    vertices, triangles = mcubes.marching_cubes(raw.squeeze(), isolevel, truncation=3.0)
    print('done', vertices.shape, triangles.shape)

    # normalize vertex positions
    vertices[:, :3] /= np.array([[tx.shape[0] - 1, ty.shape[0] - 1, tz.shape[0] - 1]])

    # Rescale and translate
    tx = tx.cpu().data.numpy()
    ty = ty.cpu().data.numpy()
    tz = tz.cpu().data.numpy()
    
    scale = np.array([tx[-1] - tx[0], ty[-1] - ty[0], tz[-1] - tz[0]])
    offset = np.array([tx[0], ty[0], tz[0]])
    vertices[:, :3] = scale[np.newaxis, :] * vertices[:, :3] + offset

    # Transform to metric units
    vertices[:, :3] = vertices[:, :3] / config['data']['sc_factor'] - config['data']['translation']


    if color_func is not None and not config['mesh']['render_color']:
        if config['grid']['tcnn_encoding']:
            vert_flat = (torch.from_numpy(vertices).to(bounding_box) - bounding_box[:, 0]) / (bounding_box[:, 1] - bounding_box[:, 0])


        fn_color = get_batch_query_fn(color_func, 1)

        chunk = 1024 * 64
        raw = [fn_color(vert_flat,  i, i + chunk).cpu().data.numpy() for i in range(0, vert_flat.shape[0], chunk)]

        sh = vert_flat.shape
        
        raw = np.concatenate(raw, 0).astype(np.float32)
        color = np.reshape(raw, list(sh[:-1]) + [-1])
        mesh = trimesh.Trimesh(vertices, triangles, process=False, vertex_colors=color)
    
    elif color_func is not None and config['mesh']['render_color']:
        print('rendering surface color')
        mesh = trimesh.Trimesh(vertices, triangles, process=False)
        vertex_normals = torch.from_numpy(mesh.vertex_normals)
        fn_color = get_batch_query_fn(color_func, 2, device=bounding_box.device)
        raw = [fn_color(torch.from_numpy(vertices), vertex_normals,  i, i + chunk).cpu().data.numpy() for i in range(0, vertices.shape[0], chunk)]

        sh = vertex_normals.shape
        
        raw = np.concatenate(raw, 0).astype(np.float32)
        color = np.reshape(raw, list(sh[:-1]) + [-1])
        mesh = trimesh.Trimesh(vertices, triangles, process=False, vertex_colors=color)
    
    elif render_uncert:
        print('rendering surface uncertainty')
        if config['grid']['tcnn_encoding']:
            vert_flat = (torch.from_numpy(vertices).to(bounding_box) - bounding_box[:, 0]) / (bounding_box[:, 1] - bounding_box[:, 0])
        
        ### get query function for uncertainty ###
        fn_uncert = lambda f, i0, i1: query_fn(f[i0:i1, None, :].to(bounding_box.device), return_uncert=True)[:, 0, 1] # query function for uncertainty

        ### get uncertainty ###
        raw_uncert = [fn_uncert(vert_flat,  i, i + chunk).cpu().data.numpy() for i in range(0, vert_flat.shape[0], chunk)]
        sh = vert_flat.shape
        raw_uncert = np.concatenate(raw_uncert, 0).astype(np.float32)

        ### colorize mesh with uncertainty ###
        ## relative uncertainty ##
        uncert_normalized = (raw_uncert - raw_uncert.min()) / (raw_uncert.max() - raw_uncert.min())
        ## absolute uncertainty ##
        # uncert_normalized = np.clip(raw_uncert, 0, 3) / 3
        colormap = plt.get_cmap('jet')
        uncert_colored = colormap(uncert_normalized.flatten())[:, :3]  # Discard alpha channel

        color = np.reshape(uncert_colored, list(sh[:-1]) + [-1])
        mesh = trimesh.Trimesh(vertices, triangles, process=False, vertex_colors=color)

    else:
        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles, process=False)

    
    os.makedirs(os.path.split(mesh_savepath)[0], exist_ok=True)
    mesh.export(mesh_savepath)

    print('Mesh saved')
    return mesh