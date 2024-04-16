import numpy as np
import torch
from tqdm import tqdm
from typing import List, Tuple

from src.planner.path_planner import PathPlanner
from src.utils.general_utils import InfoPrinter

from third_parties.coslam.utils import getVoxels


def trilinear_interpolation(voxel_grid: np.ndarray, point: np.ndarray) -> np.ndarray:
    """ trilinear interpolation

    Args:
        voxel_grid (np.ndarray, [H,W,D]): voxel grid
        point (np.ndarray, [3])         : query point

    Returns
        interpolated_value (float): interpolated value
    """
    # Get the dimensions of the voxel grid
    H, W, D = voxel_grid.shape
    
    # Extract the coordinates of the point
    x, y, z = point
    
    # Check if the point is within the valid range
    if (x < 0 or x > H - 1 or
        y < 0 or y > W - 1 or
        z < 0 or z > D - 1):
        return None  # Point is outside the grid
    
    # Calculate the integer and fractional parts of the coordinates
    x0, x1 = int(x), int(x) + 1
    y0, y1 = int(y), int(y) + 1
    z0, z1 = int(z), int(z) + 1
    
    dx, dy, dz = x - x0, y - y0, z - z0
    
    # Perform trilinear interpolation
    c000 = voxel_grid[x0, y0, z0]
    c001 = voxel_grid[x0, y0, z1]
    c010 = voxel_grid[x0, y1, z0]
    c011 = voxel_grid[x0, y1, z1]
    c100 = voxel_grid[x1, y0, z0]
    c101 = voxel_grid[x1, y0, z1]
    c110 = voxel_grid[x1, y1, z0]
    c111 = voxel_grid[x1, y1, z1]
    
    interpolated_value = (1 - dx) * (1 - dy) * (1 - dz) * c000 + \
                        (1 - dx) * (1 - dy) * dz * c001 + \
                        (1 - dx) * dy * (1 - dz) * c010 + \
                        (1 - dx) * dy * dz * c011 + \
                        dx * (1 - dy) * (1 - dz) * c100 + \
                        dx * (1 - dy) * dz * c101 + \
                        dx * dy * (1 - dz) * c110 + \
                        dx * dy * dz * c111
    
    return interpolated_value


def query_sdf_np(sdf_grid, points):
    """ Query sdf values (numpy implementation)

    Args:
        sdf_grid (np.ndarray, [H,W,D]): sdf grid
        points (np.ndarray, [N, 3])   : query points

    Returns
        sdf (np.ndarray, [N]): queried SDF values
    """
    sdf = np.array([trilinear_interpolation(sdf_grid, point) for point in points])
    return sdf


def is_collision_free(
                      pa            : np.ndarray,
                      pb            : np.ndarray,
                      sdf_map       : np.ndarray,
                      step_size     : float = 1,
                      collision_thre: float = 0.5
                    ) -> Tuple : 
    """ check if collision free between pa and pb using sdf values in between

    Args:
        pa (np.ndarray, [3])         : point A
        pb (np.ndarray, [3])         : point B
        sdf_map (np.ndarray, [X,Y,Z]): SDF volume
        step_size (float)            : rrt step size
        collision_thre (float)       : collision threshold. Unit: voxel
    
    Returns:
        Tuple: num_collision_free, complete_free
            - num_collision_free (int): number of collision-free points in between
            - complete_free (bool): is pa->pb completely collision free
    """
    ### sample points in between with a step < rrt_step_size/5 ###
    points = np.linspace(pa, pb, num=int(np.ceil(np.linalg.norm(pb - pa) / (step_size / 5))) + 1)
    points_sdf = query_sdf_np(sdf_map, points)

    ### check collision ###
    collision_check = (points_sdf > collision_thre)
    """
    FIXME: there can be potential issue!
    if agent moves to a location where simulator doesn't give collision but incorrect sdf map give collision
    Agent can be trapped and stayed without moving out
    """

    ### get number of collision-free points ###
    if collision_check.sum() == len(collision_check):
        num_collision_free = max((len(collision_check) - 1) // 5, 1)
        complete_free = True
    else:
        num_collision_free = (np.argmax(~collision_check) - 1) // 5
        complete_free = False
    return num_collision_free, complete_free


class Node: 
    def __init__(self, x: float, y: float, z: float, device='cuda'):
        """ RRT Node object

        Args:
            x (float)   : node x
            y (float)   : node y
            z (float)   : node z
            device (str): device to store node xyz as torch.Tensor

        Attributes:
            x (float)                 : node x
            y (float)                 : node y
            z (float)                 : node z
            _xyz_arr (np.ndarray, [3]): node xyz
            _xyz (torch.Tensor, [3])  : node xyz
            parent (Node)             : parent node
            _device (str)             : device to store node xyz as torch.Tensor
        """
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self._device = device
        self._xyz = torch.tensor([self.x, self.y, self.z]).reshape(1,3).to(self._device).float()
        self._xyz_arr = np.asarray([self.x, self.y, self.z])

    def get_xyz(self):
        return self._xyz


class RRT(PathPlanner):
    def __init__(self, 
                 bbox          : np.ndarray,
                 voxel_size    : float,
                 max_iter      : int = None,
                 step_size     : float = 1.,
                 maxz          : int = None,
                 z_levels      : List = None,
                 step_amplifier: int = 1,
                 collision_thre: float = 0.5,
                 margin        : int = 0,
                 device        : str='cuda',
                 enable_eval   : bool = False,
                 ):
        """ 
        Args:
            bbox (np.ndarray, [3,2]): bounding box for the space. Unit: meter
            voxel_size (float)      : voxel size
            max_iter (int)          : maximum base number of iteration for generating RRT nodes
            step_size (float)       : rrt step size. Unit: voxel
            maxz (int)              : maximum z level
            z_levels (List)         : Z levels. Unit: voxel. Min and Max level
            step_amplifier          : multiplication factor for step_size in generating nodes. aiming to generate more nodes in each step
            collision_thre (float)  : collision threshold. Unit: voxel
            margin (float)          : safe volume boundary margin
            device (str)            : device
            enable_eval (bool)      : enable evaluation, including timing
        
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
            eval_results (Dict)                      : update evaluation results
        """
        ### load arguments ###
        self.collision_thre = collision_thre
        self._device = device
        self.step_amplifier = step_amplifier
        self.step_size = step_size
        self.enable_eval = enable_eval

        ### compute volume size ###
        vol_shape = self.compute_volume_size(bbox, voxel_size)
        
        ### load parameters ###
        self.max_iter = max_iter if max_iter is not None else \
                            torch.prod(torch.tensor(vol_shape)).item() # approximate the maximum sample number by the volume size

        ### compute sampling range ###
        self.x_range = [margin, vol_shape[0] - 1 - margin]  # safe margin from boundary
        self.y_range = [margin, vol_shape[1] - 1 - margin]
        self.z_range = [margin, min(vol_shape[2] - 1 - margin, maxz)] if z_levels is None else z_levels
        self.full_x_range = [0, vol_shape[0] - 1]
        self.full_y_range = [0, vol_shape[1] - 1]
        self.full_z_range = [0, vol_shape[2] - 1]
        
        ### create regular grid for computing traversability mask when necessary ###
        sdf_x_range = torch.arange(0, vol_shape[0])
        sdf_y_range = torch.arange(0, vol_shape[1])
        sdf_z_range = torch.arange(0, vol_shape[2])
        x, y, z = torch.meshgrid(sdf_x_range, sdf_y_range, sdf_z_range, indexing="ij")
        self.points = torch.cat([x.reshape(-1, 1), 
                            y.reshape(-1, 1), 
                            z.reshape(-1, 1)], dim=1).to(self._device).float()
        self.reachable_3d_mask = torch.ones(*vol_shape).to(self._device).float()

        ### evaluation metrics ###
        self.eval_results = {
            "time (ms)": [],
            "node_num": [],
            "rrt_iter": [],
        }

    def compute_volume_size(self, bbox: np.ndarray, voxel_size: float) -> torch.Size:
        """ compute volume size (Unit: voxel)
    
        Args:
            bbox (np.ndarray, [3, 2]): bounding box corners
            voxel_size (float)       : voxel size
    
        Returns:
            vol_shape (torch.Size, [3]): volume shape
        """
        x_min, y_min, z_min = bbox[:, 0]
        x_max, y_max, z_max = bbox[:, 1]

        tx, ty, tz = getVoxels(x_max, x_min, y_max, y_min, z_max, z_min, voxel_size)
        query_pts = torch.stack(torch.meshgrid(tx, ty, tz, indexing='ij'), -1)
        vol_shape = query_pts.shape[:3]
        return vol_shape

    def start_new_plan(self, 
                       start  : np.ndarray,
                       goal   : np.ndarray,
                       sdf_map: np.ndarray
                       ) -> None:
        """ initialize a new planning request 
    
        Args:
            start (np.ndarray, [3])      : start location. Unit: voxel
            goal (np.ndarray, [3])       : start location. Unit: voxel
            sdf_map (np.ndarray, [X,Y,Z]): SDF volume.
    
        Attributes:
            start (Node)                 : start location
            goal (Node)                  : goal location
            nodes (List)                 : RRT nodes
            nodes_tensor (torch.Tensor)  : RRT nodes as tensor
            sdf_map (np.ndarray, [X,Y,Z]): SDF volume.
            rrt_iter (int)               : rrt iteration
            
        """
        self.start = Node(*start)
        self.goal = Node(*goal)

        self.nodes = [self.start]
        self.nodes_tensor = torch.cat([self.start.get_xyz()], dim=0).to(self._device)

        self.sdf_map = sdf_map

        self.rrt_iter = 0

    def generate_random_point(self, full_range: bool = False) -> np.ndarray:
        """ Generate random point.

        Args:  
            full_range (bool): generate random points within full range. otherwise in the predefined range

        Returns:
            xyz (np.ndarray, [3]): random point location
        """
        if full_range:
            x = np.random.uniform(self.full_x_range[0], self.full_x_range[1])
            y = np.random.uniform(self.full_y_range[0], self.full_y_range[1])
            z = np.random.uniform(self.full_z_range[0], self.full_z_range[1])
        else:
            x = np.random.uniform(self.x_range[0], self.x_range[1])
            y = np.random.uniform(self.y_range[0], self.y_range[1])
            z = np.random.uniform(self.z_range[0], self.z_range[1])
        xyz = np.array([x, y, z])
        return xyz

    def find_nearest_node(self, point: np.ndarray) -> Node:
        """ find the nearest node from the tree

        Args:
            point (np.ndarray, [3]): point coordinate
        
        Returns:
            Node: nearest node to the point
        """
        distances = torch.norm(torch.from_numpy(point).reshape(1,3).to(self._device) - self.nodes_tensor, dim=1, keepdim=True)
        nearest_node_index = torch.argmin(distances).item()
    
        return self.nodes[nearest_node_index]

    def extend_tree(self, full_range: bool = False) -> None:
        """ Extend tree.
        
        Args: 
            full_range (bool): sample points from full range
        
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
        if distance > self.step_size:
            new_node_arr = nearest_node._xyz_arr + diff / distance * min(self.step_size, distance)
        else:
            new_node_arr = random_point
        new_node = Node(*new_node_arr)

        ### detetermine collision-free ###
        _, complete_free = is_collision_free(nearest_node._xyz_arr, new_node._xyz_arr, self.sdf_map, step_size=self.step_size, collision_thre=self.collision_thre)

        ### add new point into the tree ###
        if complete_free:
            new_node.parent = nearest_node
            self.nodes.append(new_node)
            self.nodes_tensor = torch.cat([self.nodes_tensor, new_node.get_xyz()])
        
    def run_full(self) -> None:
        """ Run RRT planning with max number of iterations
        """
        ### RRT planning ###
        for _ in tqdm(range(self.max_iter), desc='RRT planning: '):
            self.extend_tree(full_range=True)
    
    def run(self) -> bool:
        """ RRT planning

        Returns:
            target_reachable (bool): is target reachable

        Attributes:
            goal (Node): update goal.parent
        """
        for rrt_iter in tqdm(range(self.max_iter), desc='RRT planning: '):
            self.rrt_iter += 1

            ### add straightlines if possible ###
            self.extend_tree()
            target_reachable = torch.norm(self.nodes_tensor[-1] - self.goal.get_xyz(), dim=1) < self.step_size
            if target_reachable:
                return True
        return False
                
    def find_path(self) -> List[Node]:
        """ find path

        Returns:
            path (List): planned path
        """
        path = [self.goal]
        current_node = self.goal
        while current_node.parent is not None:
            path.append(current_node.parent)
            current_node = current_node.parent
        return path
    
    def get_reachable_mask(self) -> np.ndarray:
        """get reachable/traversability mask for all voxels

        Returns:    
            reachable_3d_mask (np.ndarray, [X,Y,Z]): reachable mask
        """
        ##################################################
        ### compute distance between full grid points and 
        ###     RRT nodes. If distance is larger than 
        ###     step size, it is considered non-reachable.
        ##################################################
        batch_size = 1000
        num_repeat = self.points.shape[0] // batch_size
        invalid_mask = []
        ### first N-1 batches ###
        for i in range(num_repeat):
            points2nodes = (self.points[i*batch_size:(i+1)*batch_size].reshape(-1, 1, 3) - self.nodes_tensor.reshape(1, -1, 3))
            dist = torch.norm(points2nodes, dim=2)
            min_dist, min_dist_idx = torch.min(dist, dim=1)
            invalid_pt_idx = torch.where(min_dist > self.step_size)[0]
            invalid_mask.append(i * batch_size + invalid_pt_idx)
        
        ### last batch ###
        i += 1
        points2nodes = (self.points[(i)*batch_size:].reshape(-1, 1, 3) - self.nodes_tensor.reshape(1, -1, 3))
        dist = torch.norm(points2nodes, dim=2)
        min_dist, min_dist_idx = torch.min(dist, dim=1)
        invalid_pt_idx = torch.where(min_dist > self.step_size)[0]
        invalid_mask.append(i * batch_size + invalid_pt_idx)

        ### gather all invalid masks ###
        invalid_mask = torch.cat(invalid_mask)

        ##################################################
        ### update reachable mask
        ##################################################
        ### update reachable_mask ###
        reachable_3d_mask = self.update_reachable_mask(invalid_mask)

        ### convert to numpy ###
        reachable_3d_mask = reachable_3d_mask.detach().cpu().numpy()

        return reachable_3d_mask
    
    def update_reachable_mask(self, invalid_mask: torch.Tensor) -> torch.Tensor:
        """ update reachable mask by making non-reachable voxel to be zero
    
        Args:
            reachable_3d_mask (torch.Tensor, [X,Y,Z]): reachable mask
        """
        h, w, d = self.reachable_3d_mask.shape
        reachable_3d_mask = self.reachable_3d_mask.reshape(-1).clone()
        reachable_3d_mask[invalid_mask] = 0
        reachable_3d_mask = reachable_3d_mask.reshape(h, w, d)
        return reachable_3d_mask
    
    def update_eval(self, 
                    is_valid_planning: bool,
                    time             : float,
                    path             : List[Node]
                    ) -> None:
        """update evaluation result
    
        Args:
            is_valid_planning: is planning valid/sucess
            time             : overall planning time
            path             : planned path
    
        Attributes:
            eval_results (Dict): update evaluation results
            
        """
        if not(is_valid_planning):
            return 
        
        ### compute distance ###
        start, end = path[-1], path[0]
        dist = np.linalg.norm(start._xyz_arr - end._xyz_arr)

        ### duration ###
        self.eval_results['time (ms)'].append(time * 1000)

        ### node_num ###
        self.eval_results['node_num'].append(len(self.nodes))

        ### step_num ###
        self.eval_results['rrt_iter'].append(self.rrt_iter)

    def print_eval_result(self, info_printer: InfoPrinter) -> None:
        """ print average evaluation results

        Args:
            info_printer: information printer
        """
        info_printer("Running RRT Evaluation.")
        for key, val in self.eval_results.items():
            key_str = info_printer.adjust_string_length(20, key)
            avg_val = np.mean(np.asarray(val))
            info_printer(f"{key_str}: {avg_val:.2f}")
