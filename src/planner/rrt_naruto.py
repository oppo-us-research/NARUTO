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
from tqdm import tqdm
import torch
import torch.nn.functional as nnFunc
from typing import Tuple, List

from src.planner.rrt import Node, RRT, trilinear_interpolation, query_sdf_np, is_collision_free



class RRTNaruto(RRT):
    def __init__(self, 
                 bbox              : np.ndarray,
                 voxel_size        : float,
                 max_iter          : int = None,
                 step_size         : float = 1.,
                 maxz              : int = None,
                 z_levels          : List = None,
                 step_amplifier    : int = 1,
                 collision_thre    : float = 0.5,
                 margin            : int = 0,
                 device            : str='cuda',
                 enable_eval       : bool = False,
                 enable_direct_line: bool = True
                 ):
        """ 
        Args:
            bbox (np.ndarray, [3,2]) : bounding box for the space. Unit: meter
            voxel_size (float)       : voxel size
            max_iter (int)           : maximum base number of iteration for generating RRT nodes
            step_size (float)        : rrt step size. Unit             : voxel
            maxz (int)               : maximum z level
            z_levels (List)          : Z levels. Unit                  : voxel. Min and Max level
            step_amplifier           : multiplication factor for step_size in generating nodes. aiming to generate more nodes in each step
            collision_thre (float)   : collision threshold. Unit       : voxel
            margin (float)           : safe volume boundary margin
            device (str)             : device
            enable_eval (bool)       : enable evaluation, including timing
            enable_direct_line (bool): enable direct line attempt
        
        Attributes:
            collision_thre (float)                   : collision threshold. Unit: voxel
            _device (str)                            : device
            step_amplifier                           : multiplication factor for step_size in generating nodes. aiming to generate more nodes in each step
            step_size (float)                        : rrt step size. Unit      : voxel
            max_iter (int)                           : maximum base number of iteration for generating RRT nodes
            x/y/z_range (List)                       : x/y/z range
            full_x/y/z_range (List)                  : full x/y/z range
            points (torch.Tensor, [X*Y*Z, 3])        : full grid points
            reachable_3d_mask (torch.Tensor, [X,Y,Z]): reachable 3D mask
        """
        super(RRTNaruto, self).__init__(
                bbox           = bbox,
                voxel_size     = voxel_size,
                max_iter       = max_iter,
                step_size      = step_size,
                maxz           = maxz,
                z_levels       = z_levels,
                step_amplifier = step_amplifier,
                collision_thre = collision_thre,
                margin         = margin,
                device         = device,
                enable_eval    = enable_eval
            )
        self.enable_direct_line = enable_direct_line

    def extend_tree_straight(self) -> bool:
        """ Extend tree with preferbly the straight line connecting goal and latest node

        Returns:
            is_target_reached (bool): is target reached. if the distance is less than step size 

        Attributes:
            nodes (List): RRT nodes
            nodes_tensor (torch.Tensor, [N, 3]): RRT nodes as torch.Tensor
        """
        num_collision_free_step, _ = is_collision_free(
                            self.goal._xyz_arr, 
                            self.nodes[-1]._xyz_arr,
                            self.sdf_map,
                            self.step_size,
                            )
        # if complete_free:
        if num_collision_free_step > 0:
            x_diff = self.goal.x - self.nodes[-1].x
            y_diff = self.goal.y - self.nodes[-1].y
            z_diff = self.goal.z - self.nodes[-1].z
            distance = np.linalg.norm([x_diff, y_diff, z_diff])
            cur_nearest_node = nearest_node = self.nodes[-1]
            for i in range(num_collision_free_step):
                x_new2 = nearest_node.x + (x_diff / distance) * min(self.step_size * (i + 1), distance)
                y_new2 = nearest_node.y + (y_diff / distance) * min(self.step_size * (i + 1), distance)
                z_new2 = nearest_node.z + (z_diff / distance) * min(self.step_size * (i + 1), distance)
                new_node = Node(x_new2, y_new2, z_new2)
                new_node.parent = cur_nearest_node
                self.nodes.append(new_node)
                self.nodes_tensor = torch.cat([self.nodes_tensor, new_node.get_xyz()])

                ### update current_node ###
                cur_nearest_node = new_node
            
            ### return True if target is reached ###
            if torch.norm(self.nodes_tensor[-1] - self.goal.get_xyz(), dim=1) < self.step_size:
                return True
            else:
                return False
        else:
            return False

    def extend_tree(self, full_range: bool = False) -> int:
        """ Extend tree. Note that as gorwing RRT can be time-consuming.
        We speed up the growing process by adding multiple points per step: 
            We make the step bigger using a multiplier and add all consecutive points that don't collide into the tree.
        
        Args: 
            full_range (bool): sample points from full range
        
        Returns:
            num_collision_free_step (int): number of collision free steps
        
        Attributes:
            nodes (List): RRT nodes
            nodes_tensor (torch.Tensor, [N, 3]): RRT nodes as torch.Tensor
        """
        ##################################################
        ### genreate random point and find the nearest node from RRT
        ##################################################
        random_point = self.generate_random_point(full_range)
        nearest_node = self.find_nearest_node(random_point)

        ##################################################
        ### add new nodes
        ##################################################
        ### compute the random point to be added ###
        diff = random_point - nearest_node._xyz_arr
        distance = np.linalg.norm(diff)
        if distance > self.step_size * self.step_amplifier:
            new_node_arr = nearest_node._xyz_arr + diff / distance * min(self.step_size * self.step_amplifier, distance)
        else:
            new_node_arr = random_point
        new_node = Node(*new_node_arr)

        ### detetermine how many interpolated points are collision-free ###
        num_collision_free_step, _ = is_collision_free(nearest_node._xyz_arr, new_node._xyz_arr, self.sdf_map, step_size=self.step_size, collision_thre=self.collision_thre)

        ## add new collision-free points into the nodes ##
        if num_collision_free_step > 0:
            diff = new_node._xyz_arr - nearest_node._xyz_arr
            distance = np.linalg.norm(diff)
            cur_nearest_node = nearest_node
            for i in range(num_collision_free_step):
                new_node_arr = nearest_node._xyz_arr + diff / distance * min(self.step_size * (i+1), distance)

                new_node = Node(*new_node_arr)
                new_node.parent = cur_nearest_node
                self.nodes.append(new_node)
                self.nodes_tensor = torch.cat([self.nodes_tensor, new_node.get_xyz()])

                ### update nearest_node ###
                cur_nearest_node = new_node
        
        return num_collision_free_step

    def run(self) -> bool:
        """ RRT planning

        Returns:
            target_reachable (bool): is target reachable

        Attributes:
            goal (Node): update goal.parent
        """
        ### RRT planning ###
        for step in tqdm(range(self.max_iter), desc='RRT planning: '):
            self.rrt_iter += 1

            if self.enable_direct_line:
                ### add straightlines if possible ###
                target_reached = self.extend_tree_straight()
                if target_reached:
                    break
                else:
                    ### add random point/line ###
                    num_new_nodes = self.extend_tree()
                
                ### early break if newly added line points can reach target ###
                if num_new_nodes > 0:
                    dist_new_nodes_to_goal = torch.norm(self.nodes_tensor[-num_new_nodes:] - self.goal.get_xyz(), dim=1)
                    min_dist = torch.min(dist_new_nodes_to_goal).item()
                    if min_dist < self.step_size:
                        break
            else:
                num_new_nodes = target_reached = self.extend_tree()
                ### early break if newly added line points can reach target ###
                if num_new_nodes > 0:
                    dist_new_nodes_to_goal = torch.norm(self.nodes_tensor[-num_new_nodes:] - self.goal.get_xyz(), dim=1)
                    min_dist = torch.min(dist_new_nodes_to_goal).item()
                    if min_dist < self.step_size:
                        break
        
        ### check if RRT sucess ###
        last = self.find_nearest_node(self.goal._xyz_arr)
        dist_last2goal = np.linalg.norm(last._xyz_arr - self.goal._xyz_arr) # voxel unit
        if dist_last2goal > self.step_size:
            target_reachable = False
        else:
            target_reachable = True
        self.goal.parent = last
        return target_reachable

    ##################################################
    ### unused functions
    ##################################################

    def query_sdf(self, sdf_grid_in: torch.Tensor, points_in: torch.Tensor) -> torch.Tensor:
        """ Query SDF grid with trilinear interpolation
    
        Args:
            sdf_grid (torch.Tensor, [H,W,D]): sdf grid
                   /
                  Z
                 /
                |-- Y --
                X
                |
            points (torch.Tensor, [K, 3]): query points
    
        Returns:
            sdf (torch.Tensor, [K]): query sdf
    
        """
        ### convert array to tensor ###
        H,W,D = sdf_grid_in.shape
        sdf_grid = sdf_grid_in.unsqueeze(0).unsqueeze(0) # 1,1,X,Y,Z
        points = points_in.unsqueeze(0).unsqueeze(0).unsqueeze(0) # 1,1,1,K,3

        ### normalize points ###
        points[:, :, :, :, 0] = points[:, :, :, :, 0] / (H-1) * 2 - 1 #X
        points[:, :, :, :, 1] = points[:, :, :, :, 1] / (W-1) * 2 - 1 #Y
        points[:, :, :, :, 2] = points[:, :, :, :, 2] / (D-1) * 2 - 1 #Z

        ##################################################
        ### grid_sample requires Data N,C,D,H,W; provided data is 1,1,X,Y,Z 
        ### grid_sample requires query_point to be [u,v,w] where u->W; v->H, w->D 
        ### rearrange query point to match the requirement 
        ##################################################
        points = points[:,:,:,:,[2,1,0]]

        ### trilinear interpolation ###
        sdf = nnFunc.grid_sample(sdf_grid, points, align_corners=True) # 1,1,1,1,K
        sdf = sdf[0,0,0,0]#.detach().cpu().numpy()

        return sdf

    def is_collision_free_gpu(self, node1: Node, node2: Node) -> Tuple:
        """ check if collision free between pa and pb using sdf values in between

        Args:
            node1 (Node): node 1
            node2 (Node): node 2
        
        Returns:
            Tuple: num_collision_free, complete_free
                - num_collision_free (int): number of collision-free points in between
                - complete_free (bool): is pa->pb completely collision free
        """
        ### interpolate points in-between ###
        pa = node1.get_xyz()
        pb = node2.get_xyz()
        points_x = torch.linspace(pa[0,0], pb[0,0], steps=int(torch.ceil(torch.norm(pb - pa) / (self.step_size / 5))) + 1, device=pa.device).float()
        points_y = torch.linspace(pa[0,1], pb[0,1], steps=int(torch.ceil(torch.norm(pb - pa) / (self.step_size / 5))) + 1, device=pa.device).float()
        points_z = torch.linspace(pa[0,2], pb[0,2], steps=int(torch.ceil(torch.norm(pb - pa) / (self.step_size / 5))) + 1, device=pa.device).float()
        points = torch.stack([points_x, points_y, points_z], 1)
        points_sdf = self.query_sdf(torch.from_numpy(self.sdf_map).cuda().float(), points)

        collision_check = (points_sdf > self.collision_thre) * 1.
        """
        FIXME: there can be potential issue!
        if agent moves to a location where simulator doesn't give collision but incorrect sdf map give collision
        Agent can be trapped and stayed without moving out
        """
        if collision_check.sum() == len(collision_check):
            num_collision_free = max((len(collision_check) - 1) // 5, 1)
            complete_free = True
        else:
            num_collision_free = (torch.argmax(1-collision_check).item() - 1) // 5
            complete_free = False

        return num_collision_free, complete_free
