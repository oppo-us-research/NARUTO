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
import torch
from typing import Dict, List, Tuple

from src.planner.planner import Planner, compute_camera_pose
from src.utils.general_utils import InfoPrinter
from src.planner.rotation_planning import rotation_planning
from src.planner.rrt_naruto import Node, is_collision_free
from src.data.pose_loader import habitat_pose_conversion

class NarutoPlanner(Planner):
    def __init__(self, 
                 main_cfg    : mmengine.Config,
                 info_printer: InfoPrinter,
                 ) -> None:
        """
        Args:
            main_cfg (mmengine.Config): Configuration
            info_printer (InfoPrinter): information printer

        Attributes:
            state (str): planner state
        """
        super(NarutoPlanner, self).__init__(main_cfg, info_printer)
        
        ### initialize planner state ###
        self.state = "staying"

    def init_local_planner(self) -> None:
        """ initialize local planner 
    
        Attributes:
            local_planner (RRTNaruto)
            
        """
        if self.planner_cfg.local_planner_method == 'RRTNaruto':
            from src.planner.rrt_naruto import RRTNaruto
            self.local_planner = RRTNaruto(
                bbox               = self.bbox,
                voxel_size         = self.voxel_size,
                max_iter           = self.planner_cfg.get("rrt_max_iter", None),
                step_size          = self.planner_cfg.rrt_step_size,
                maxz               = self.planner_cfg.rrt_maxz,
                z_levels           = self.planner_cfg.get("rrt_z_levels", None),
                step_amplifier     = self.planner_cfg.get("rrt_step_amplifier", 1),
                collision_thre     = self.planner_cfg.get("collision_thre", 0.05) / self.voxel_size, # Unit: xovel
                enable_eval        = self.planner_cfg.get("enable_eval", False),
                enable_direct_line = self.planner_cfg.get("enable_direct_line", True),
            )
        elif self.planner_cfg.local_planner_method == 'RRT':
            from src.planner.rrt import RRT
            self.local_planner = RRT(
                bbox           = self.bbox,
                voxel_size     = self.voxel_size,
                max_iter       = self.planner_cfg.get("rrt_max_iter", None),
                step_size      = self.planner_cfg.rrt_step_size,
                maxz           = self.planner_cfg.rrt_maxz,
                z_levels       = self.planner_cfg.get("rrt_z_levels", None),
                step_amplifier = self.planner_cfg.get("rrt_step_amplifier", 1),
                collision_thre = self.planner_cfg.get("collision_thre", 0.05) / self.voxel_size, # Unit: xovel
                enable_eval    = self.planner_cfg.get("enable_eval", False)
            )
        return

    def init_data(self, bbox: List) -> None:
        """initialize data for naruto planner
    
        Args:
            bbox (List, [3,2]): bounding box corners coordinates
    
        Attributes:
            gs_z_levels (List, [N])                  : Goal Space Z-levels. if not provided, unitformly samples from Z range.
            voxel_size (float)                       : voxel size
            bbox (np.ndarray, [3,2])                 : bounding box corners coordinates
            Nx/Ny/Nz (int)                           : bounding box sizes
            gs_x/y/z_range (torch.Tensor, [X/Y/Z])   : goal space X/Y/Z levels
            goal_space_pts (torch.Tensor, [X*Y*Z, 3]): goal space candidate locations. Unit: voxel
        """
        self.path = None
        self.lookat_tgts = None

        ### load config data ###
        self.gs_z_levels = self.planner_cfg.get("gs_z_levels", [5, 11, 17])
        self.voxel_size = self.main_cfg.planner.voxel_size # 0.1

        ### bounding box ###
        self.bbox = np.asarray(bbox)

        ## bounding box size (Unit: voxel) ##
        Nx = round((bbox[0][1] - bbox[0][0]) / self.voxel_size + 0.0005) + 1
        Ny = round((bbox[1][1] - bbox[1][0]) / self.voxel_size + 0.0005) + 1
        Nz = round((bbox[2][1] - bbox[2][0]) / self.voxel_size + 0.0005) + 1
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz

        ### Goal Space ###
        self.gs_x_range = torch.arange(0, self.Nx, 2)
        self.gs_y_range = torch.arange(0, self.Ny, 2) 
        if self.gs_z_levels is None:
            ### generate z-levels from 0.5 meter to bounding box's maximum Z-level ###
            self.gs_z_range = torch.arange( 
                        int(1/self.voxel_size), 
                        self.Nz, 
                        int(1/self.voxel_size))  
        else:
            self.gs_z_range = torch.tensor(self.gs_z_levels)
        self.gs_x, self.gs_y, self.gs_z = torch.meshgrid(self.gs_x_range, self.gs_y_range, self.gs_z_range, indexing="ij")
        self.goal_space_pts = torch.cat([self.gs_x.reshape(-1, 1), 
                                         self.gs_y.reshape(-1, 1), 
                                         self.gs_z.reshape(-1, 1)], dim=1).cuda().float()

    def main(self, 
             uncert_sdf_vols: List,
             cur_pose       : np.ndarray,
             is_new_vols    : bool
             ) -> torch.Tensor:
        """ Naruto Planner main function
    
        Args:
            uncert_sdf_vols (List)      : Uncertainty Volume and SDF Volume
                - uncert_vol (np.ndarray, [X,Y,Z]): uncertainty volume
                - sdf_vol (np.ndarray, [X,Y,Z])   : SDF volume
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
            is_new_vols (bool)          : is uncert_sdf_vols new optimized volumes
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system
        """
        self.update_state(uncert_sdf_vols[1], cur_pose, is_new_vols)
        self.info_printer(f"Current state: {self.state}", self.step, self.__class__.__name__)
        new_pose = self.compute_next_state_pose(cur_pose, uncert_sdf_vols)
        new_pose = torch.from_numpy(new_pose).float()
        return new_pose

    def update_state(self, 
                     sdf_vol    : np.ndarray,
                     cur_pose   : np.ndarray,
                     is_new_vols: bool
                     ) -> None:
        """ update state machine for the planner
    
        Args:
            sdf_vol (np.ndarray, [X,Y,Z]): SDF volume
            cur_pose (np.ndarray, [4,4]) : current pose. Format: camera-to-world, RUB system
            is_new_vols (bool)           : is uncert_sdf_vols new optimized volumes
    
        Attributes:
            state (str): planner state
        """
        ##################################################
        ### planning
        ##################################################
        if self.state == "planning":
            is_goal_reachable = self.check_goal_reachable()
            self.state = "rotationPlanningAtStart" if is_goal_reachable else "staying"
        ##################################################
        ### rotation planning at start point
        ##################################################
        elif self.state == "rotationPlanningAtStart":
            self.state = "rotatingAtStart"
        ##################################################
        ### rotating at start point
        ##################################################
        elif self.state == "rotatingAtStart":
            is_rotation_done = self.check_rotation_done()
            self.state = "movingToGoal" if is_rotation_done else "rotatingAtStart"
        ##################################################
        ### moving to goal
        ##################################################
        elif self.state == "movingToGoal":
            is_goal_reached = self.check_goal_reached()
            if is_goal_reached:
                self.state = "rotationPlanningAtGoal"
            else:
                next_pt_node = self.path[-1]
                next_pt_loc = self.vox2loc(next_pt_node._xyz_arr)
                is_collision_detected = self.detect_collision_v2(
                                            sdf_vol     = sdf_vol,
                                            cur_pose    = cur_pose,
                                            next_pt_loc = next_pt_loc
                                            )
                if is_collision_detected:
                    self.state = "staying" 
                else:
                    self.state = "movingToGoal"
        ##################################################
        ### rotation planning at goal location
        ##################################################
        elif self.state == "rotationPlanningAtGoal":
            self.state = "rotatingAtGoal"
        ##################################################
        ### rotating at goal location
        ##################################################
        elif self.state == "rotatingAtGoal":
            is_rotation_done = self.check_rotation_done()
            self.state = "planning" if is_rotation_done else "rotatingAtGoal"
        ##################################################
        ### staying at current location
        ##################################################
        elif self.state == "staying":
            is_new_map_received = self.check_new_map_received(is_new_vols)
            self.state = "planning" if is_new_map_received else "staying"
    
    def compute_next_state_pose(self, 
                                cur_pose       : np.ndarray,
                                uncert_sdf_vols: List
                                ) -> np.ndarray:
        """ compute next state pose
    
        Args:
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world
            path (List)                 : each element is a Node. [GoalNode, ..., CurrentNode]
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world

        Attributes:
            traversability_mask (np.ndarray, [X,Y,Z]): valid goal space mask. get updated in self.uncertainty_aware_planning_v2()
            is_goal_reachable (bool)                 : is goal reachable
            lookat_tgts (List)                       : uncertaint target observation locations to lookat. each element is (np.ndarray, [3])
            path (List)                              : each element is a Node. [GoalNode, ..., CurrentNode]
        """
        ##################################################
        ### planning
        ##################################################
        if self.state == "planning":
            planning_out = self.uncertainty_aware_planning_v2(uncert_sdf_vols, cur_pose)
            self.is_goal_reachable = planning_out['is_goal_reachable']
            self.lookat_tgts = planning_out['lookat_tgts']
            self.path = planning_out['path']
            new_pose = cur_pose.copy()
        ##################################################
        ### rotation planning at start location
        ##################################################
        elif self.state == "rotationPlanningAtStart":
            new_pose = self.rotation_planning_at_start(cur_pose, self.lookat_tgts[0])
        ##################################################
        ### rotating at start location
        ##################################################
        elif self.state == "rotatingAtStart":
            new_pose = self.rotating_at_start(cur_pose)
        ##################################################
        ### moving to goal 
        ##################################################
        elif self.state == "movingToGoal":
            next_node = self.path[-1]
            new_pose = self.moving_to_goal(cur_pose, self.lookat_tgts[0], next_node)
            self.path.pop(-1)
        ##################################################
        ### rotation planning at goal location
        ##################################################
        elif self.state == "rotationPlanningAtGoal":
            new_pose = self.rotation_planning_at_goal(cur_pose, self.lookat_tgts)
        ##################################################
        ### rotating at goal location
        ##################################################
        elif self.state == "rotatingAtGoal":
            new_pose = self.rotating_at_start(cur_pose)
        ##################################################
        ### staying at current location
        ##################################################
        elif self.state == "staying":
            new_pose = cur_pose.copy()
        else:
            raise NotImplementedError

        return new_pose
  
    def compute_traversability_mask(self,
                                    sdf: np.ndarray,
                                    pose: np.ndarray,
                                    ) -> np.ndarray:
        """ generate a temporaty traversability 3D mask based on the current map
    
        Args:
            sdf (np.ndarray, [X,Y,Z]): SDF Volume
            pose (np.ndarray, [4,4]) : current pose. Format: camera-to-world, RUB system
    
        Returns:
            traversability_mask (np.ndarray, [X,Y,Z]): valid goal space mask
        """
        cur_vxl = self.loc2vox(pose[:3, 3])
        self.local_planner.start_new_plan(
            start = cur_vxl, 
            goal = np.zeros((3)), # dummy target,
            sdf_map = sdf
            )
        self.local_planner.run_full()
        traversability_mask = self.local_planner.get_reachable_mask()
        return traversability_mask

    def uncertainty_aware_planning_v2(self,
                                      uncert_sdf_vols: List,
                                      cur_pose       : np.ndarray,
                                    ) -> Dict:
        """ running Uncertainty Aware Planning
    
        Args:
            uncert_sdf_vols (List)      : Uncertainty Volume and SDF Volume
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
    
        Returns:
            Dict: planning output
                - path (List)             : each element is a Node. [GoalNode, ..., CurrentNode]
                - is_goal_reachable (bool): is goal reachable
                - lookat_tgts (List)      : uncertaint target observation locations to lookat
                    each element is (np.ndarray, [3])
    
        Attributes:
            traversability_mask (np.ndarray, [H,W,D]): traversability mask may be updated 
            
        """
        ### Initialize Data and flags
        uncert_vol, sdf_vol = uncert_sdf_vols
        if self.step == 0:
            self.traversability_mask = np.ones_like(uncert_vol)

        ### FIlter Uncertainty Volume based on traversability
        if self.planner_cfg.enable_uncert_filtering:
            uncert_vol = uncert_vol * self.traversability_mask

        ### uncertainty aggregation ###
        valid_uncert_aggre, uncert_aggre_outputs = self.uncertainty_aggregation_v2(
                                [uncert_vol, sdf_vol], 
                                force_running=self.planner_cfg.force_uncert_aggre
                                )
        
        if not(valid_uncert_aggre) and self.planner_cfg.enable_uncert_filtering:
            ##################################################
            ### Compute traversability and filter out invalid
            ###     goals if they are not reachable.
            ### Perform uncertainty aggregation again for 
            ###     valid uncertain target observations.
            ##################################################
            self.traversability_mask = self.compute_traversability_mask(
                sdf = uncert_sdf_vols[1],
                pose = cur_pose,
            )
            uncert_vol = uncert_vol * self.traversability_mask

            valid_uncert_aggre, uncert_aggre_outputs = self.uncertainty_aggregation_v2(
                    [uncert_vol, sdf_vol], 
                    force_running=self.planner_cfg.force_uncert_aggre
                    )
        
        ### goal search ###
        goal_vxl, lookat_tgts = self.goal_search_v2(uncert_aggre_outputs)

        ### RRT ###
        if self.planner_cfg.enable_eval:
            self.timer.start("path_planning", self.__class__.__name__)
        path, is_goal_reachable, traversability_mask = self.path_planning_v2(
                    sdf_vol  = sdf_vol,
                    cur_pose = cur_pose,
                    goal_vxl = goal_vxl,
                )
        if self.planner_cfg.enable_eval:
            self.timer.end("path_planning")
            self.local_planner.update_eval(
                is_valid_planning = is_goal_reachable,
                time              = self.timer.get_last_timing("path_planning"),
                path              = path
            )
            self.local_planner.print_eval_result(self.info_printer)
        if traversability_mask is not None:
            self.traversability_mask = traversability_mask

        ### gather output data ###
        out = dict(
            path                = path,
            is_goal_reachable = is_goal_reachable,
            lookat_tgts         = lookat_tgts,
        )
        return out

    def path_planning_v2(self, 
                         sdf_vol : np.ndarray,
                         cur_pose: np.ndarray,
                         goal_vxl: np.ndarray
                         ) -> Tuple:
        """ Path planning
    
        Args:
            sdf_vol (np.ndarray, [X,Y,Z]): SDF volume
            cur_pose (np.ndarray, [4,4]) : current pose. Format: camera-to-world, RUB system
            goal_vxl (np.ndarray, [3])   : goal location. Unit : voxel
    
        Returns:
            Tuple: path planning output
                - path (List)             : each element is a Node. [GoalNode, ..., CurrentNode]
                - is_goal_reachable (bool): is goal reachable
                - traversability_mask (np.ndarray, [X,Y,Z]): traversability mask 
        """
        ### Force initial SDF to be empty space ###
        if self.step == 0:
            sdf_vol = sdf_vol * 0. + 100.
        
        ## run local path planner ##
        cur_vxl = self.loc2vox(cur_pose[:3, 3])
        self.local_planner.start_new_plan(
            start = cur_vxl,
            goal = goal_vxl,
            sdf_map = sdf_vol
        )
        target_reachable_1st_run = self.local_planner.run()
        
        ##################################################
        ### If target is non-reachable, run RRT one more 
        ###   time and obtain traversability masks.
        ##################################################
        if not(target_reachable_1st_run):
            ### run RRT one more time to increase density ###
            self.info_printer("Run RRT second time to increase RRT node density.",
                self.step, self.__class__.__name__
            )
            is_goal_reachable = self.local_planner.run()

            if not(is_goal_reachable):
                ### get traversability mask ###
                self.info_printer("Update observation traversability mask.",
                    self.step, self.__class__.__name__
                )
                traversability_mask = self.local_planner.get_reachable_mask()
            else:
                traversability_mask = None
        else:
            traversability_mask = None
            is_goal_reachable = True

        ### find path ###
        path = self.local_planner.find_path()

        return path, is_goal_reachable, traversability_mask

    def goal_search_v2(self, uncert_aggre_outputs: Dict) -> Tuple:
        """ goal search based on uncertainty predictions
    
        Args:
            uncert_aggre_outputs (Dict): uncertainty aggregation outputs
                - gs_aggre_uncert (torch.Tensor, [X,Y,Z])         : goal space aggregated uncertainty.
                - topk_uncert_vxl (torch.Tensor, [k, 3])          : sampled top-k uncertain target observations. Unit: voxel
                - gs_uncert_collections (torch.Tensor, [X*Y*Z, k]): individual uncertainties for each goal space candidate
    
        Returns:
            Tuple: 
                - goal_vxl (np.ndarray, [3]): goal location. Unit: voxel
                - lookat_tgt_locs (List): uncertaint target observation locations to lookat 
                    each element is (np.ndarray, [3])
        """
        ### load data ###
        gs_aggre_uncerts = uncert_aggre_outputs['gs_aggre_uncerts'].cpu().numpy()
        topk_uncert_vxl = uncert_aggre_outputs['topk_uncert_vxl']
        gs_uncert_collections = uncert_aggre_outputs['gs_uncert_collections']

        ### identify goal location in voxel space ###
        max_uncert_vxl = np.argpartition(gs_aggre_uncerts, -1, axis=None)[-1]
        max_uncert_vxl = np.unravel_index(max_uncert_vxl, gs_aggre_uncerts.shape)
        goal_vxl = np.array([self.gs_x_range[max_uncert_vxl[0]],
                             self.gs_y_range[max_uncert_vxl[1]],
                             self.gs_z_range[max_uncert_vxl[2]]])
        
        ##################################################
        ### compute look-at locations by
        ###     finding the top-k uncertain observations from 
        ###     the computed goal location,
        ###     and choosing the most uncertain observation
        ###     as the lookat target while moving the goal
        ##################################################
        uncert_obs_from_goal = gs_uncert_collections.reshape([
                                    self.gs_x_range.shape[0], 
                                    self.gs_y_range.shape[0], 
                                    self.gs_z_range.shape[0], 
                                    -1])[max_uncert_vxl]
        lookat_target_vals, lookat_tgt_vxls = uncert_obs_from_goal.topk(k=self.planner_cfg.obs_per_goal, largest=True)  # indices of top k largest uncertain observations in most 
        lookat_tgt_vxls = lookat_tgt_vxls[:max((lookat_target_vals > 0).sum(), 1)]  # only consider target voxels with non-zero uncertainty
        lookat_tgt_vxls = topk_uncert_vxl[lookat_tgt_vxls].cpu().numpy()

        lookat_tgt_locs = []
        for lookat_tgt_vxl in lookat_tgt_vxls:
            lookat_tgt_locs.append(self.vox2loc(lookat_tgt_vxl))
            
        # lookat_tgt_loc = lookat_tgt_locs.pop(0) 
        return goal_vxl, lookat_tgt_locs

    def detect_collision_v2(self,
                            sdf_vol    : np.ndarray,
                            cur_pose   : np.ndarray,
                            next_pt_loc: np.ndarray
                            ) -> bool:
        """ detect collision with the following strategies
        (1) SDF map: querying SDF values between current location and planned next location
        (2) Simulation (Closest Dist): 
                Assume an agent's safety boundary. check the closest distance from next location, 
                mimicing the situation that the agent cannot reach the next location due to the safety boundary
        (3) Simulation (Invalid ERP):
                As some scenes (e.g. MP3D) do not have watertight mesh, the agent tends to going outside the scene.
                We determine this by checking the ratio of invalid equirectangular distance measurement.
    
        Args:
            sdf_vol (np.ndarray, [X,Y,Z]): SDF volume
            cur_pose (np.ndarray, [4,4]) : current pose. Format     : camera-to-world, RUB system
            next_pt_loc (np.ndarray, [3]): next point location. Unit: meter
            
        Returns:
            is_collision_detected (bool): is collision detected
        """
        ##################################################
        ### Run simulation at next location
        ##################################################
        if self.main_cfg.general.dataset in ['MP3D', "Replica", "NARUTO"]:
            ### compute next-state pose ###
            next_c2w_coslam = cur_pose.copy()
            next_c2w_coslam[:3, 3] = next_pt_loc
            next_c2w_sim = next_c2w_coslam.copy()

            ### simulate ERP depth at next-state pose ###
            _, _, _, erp_depth = self.sim.simulate(next_c2w_sim, return_erp=True, no_print=True)
            
            dist_closest = erp_depth.min()
            invalid_region_ratio = (erp_depth>1e6).sum() / (erp_depth.shape[0] * erp_depth.shape[1]) # invalid depths are set as large values
        else:
            raise NotImplementedError

        ##################################################
        ### check collision from SDF
        ##################################################
        cur_pt_vxl = self.loc2vox(cur_pose[:3, 3])
        next_pt_vxl = self.loc2vox(next_pt_loc)
        _, sdf_collision_free = is_collision_free(
                                next_pt_vxl, 
                                cur_pt_vxl, 
                                sdf_vol, 
                                step_size=self.planner_cfg.rrt_step_size,
                                )

        ##################################################
        ### collision detection
        ##################################################
        ''' We additionally introduce a detection mechanism using closest dist to the scene.
        dist_closest < 0.05 assumes that agent has a safety sphere boundary of 5cm radius. 
        We simply use the simulation result to inform whether the agent can or cannot move to the next state.
        This will be more robust, especially in MP3D scene (where GTs are with lower quality), but it is optional.
        '''
        invalid_region_ratio_thre = self.planner_cfg.get("invalid_region_ratio_thre", 0.2) 
        if self.main_cfg.general.dataset == 'Replica':
            is_collision_detected = not(sdf_collision_free)
            # is_collision_detected = dist_closest < self.planner_cfg.collision_dist_thre or invalid_region_ratio > invalid_region_ratio_thre or not(sdf_collision_free)
        elif self.main_cfg.general.dataset == 'MP3D':
            is_collision_detected = invalid_region_ratio > invalid_region_ratio_thre or not(sdf_collision_free)
            # is_collision_detected = dist_closest < self.planner_cfg.collision_dist_thre or invalid_region_ratio > invalid_region_ratio_thre or not(sdf_collision_free)
        elif self.main_cfg.general.dataset == 'NARUTO':
            # is_collision_detected = not(sdf_collision_free)
            is_collision_detected = dist_closest < self.planner_cfg.collision_dist_thre or invalid_region_ratio > invalid_region_ratio_thre or not(sdf_collision_free)
        else:
            raise NotImplementedError
        
        ### Collision Detected ###
        if is_collision_detected:
            self.info_printer("Collision Detected!", 
                          self.step, self.__class__.__name__)   
            self.info_printer(f"    Invalid region ratio: {invalid_region_ratio:.3f}", 
                            self.step, self.__class__.__name__)
            self.info_printer(f"    SDF collision free: {sdf_collision_free}", 
                            self.step, self.__class__.__name__)
            self.info_printer(f"    Observation distance: {dist_closest*100:.3f}cm", 
                            self.step, self.__class__.__name__)
        return is_collision_detected

    @torch.no_grad()
    def uncertainty_aggregation_v2(self, 
                                   uncert_sdf_vols: List,
                                   force_running  : bool = False
                                   ) -> Tuple[bool, Dict]: 
        """ Uncertainty Aggregation in Goal Space for Goal Search

        Args:
            uncert_sdf_vols (List)  : Uncertainty Volume and SDF Volume
                - uncert_vol (np.ndarray, [X,Y,Z]): uncertainty volume
                - sdf_vol (np.ndarray, [X,Y,Z])   : SDF volume
            force_running (bool)    : force running even goal space is invalid
        
        Returns:
            Tuple [bool, Dict]: goal_space_valid, outputs
                - outputs includes
                    - gs_aggre_uncert (torch.Tensor, [X,Y,Z]): goal space aggregated uncertainty.
                    - topk_uncert_vxl (torch.Tensor, [k, 3]): sampled top-k uncertain target observations. Unit: voxel
                    - gs_uncert_collections (torch.Tensor, [X*Y*Z, k]): individual uncertainties for each goal space candidate 
        """
        ##################################################
        ### Remove the predicted empty space from consideration
        ##################################################
        sdf = uncert_sdf_vols[1]
        uncert = uncert_sdf_vols[0]
        
        ##################################################
        ### get top-k uncertainty target points
        ##################################################
        top_k_subset = self.planner_cfg.uncert_top_k_subset
        top_k = self.planner_cfg.uncert_top_k
        topk_uncert_vxl = np.argpartition(uncert, -top_k, axis=None)[-top_k_subset:]
        topk_uncert_vxl = np.unravel_index(topk_uncert_vxl, uncert.shape)
        topk_uncert_vxl = np.column_stack(topk_uncert_vxl)
        topk_uncert_vxl = torch.from_numpy(topk_uncert_vxl).cuda().float()

        ### move volumes to TorchTensor and cuda device ###
        uncert = torch.from_numpy(uncert).cuda()
        sdf = torch.from_numpy(sdf).cuda()

        ##################################################
        ### only consider goal_space_pts within max sensing distance
        ##################################################
        goal_space_pts = self.goal_space_pts[:,None,:].repeat([1,top_k_subset,1])  # [x*y*z, k, 3]
        view_vec = goal_space_pts - topk_uncert_vxl  # [x*y*z, k, 3]
        dist = torch.norm(view_vec, dim=2)  # [x*y*z, k]
        min_sensing_dist = self.planner_cfg.gs_sensing_range[0] / self.voxel_size
        max_sensing_dist = self.planner_cfg.gs_sensing_range[1] / self.voxel_size
        dist_mask = (dist < max_sensing_dist ) * (dist > min_sensing_dist)  # we assume best view distance is between 0.5m and 2m
        if dist_mask.sum() == 0:
            self.info_printer(
                "   Warning! All high uncertainty points are far from Goal Space points",
                self.step,
                self.__class__.__name__ 
                )
        valid_mask = torch.ones_like(goal_space_pts[:, :, 0]) == 1.
        valid_mask = valid_mask * dist_mask
        
        ##################################################
        ### remove goal points that are not safe to go (close to sdf surface)
        ##################################################
        safe_sdf = self.planner_cfg.safe_sdf
        unsafe_tgt = ((self.gs_x<1) + (self.gs_x+1>=self.Nx) + \
                      (self.gs_y<1) + (self.gs_y+1>=self.Ny) + \
                      (self.gs_z<1) + (self.gs_z+1>=self.Nz)).cuda()  # FIXME(HY): newly added
        unsafe_tgt += (sdf[self.gs_x,self.gs_y,self.gs_z]<safe_sdf) + \
                      (sdf[(self.gs_x+1).clamp(0,self.Nx-1),self.gs_y,self.gs_z]<safe_sdf) + \
                      (sdf[(self.gs_x-1).clamp(0,self.Nx-1),self.gs_y,self.gs_z]<safe_sdf)
        unsafe_tgt += (sdf[self.gs_x,(self.gs_y+1).clamp(0,self.Ny-1),self.gs_z]<safe_sdf) + \
                      (sdf[self.gs_x,(self.gs_y-1).clamp(0,self.Ny-1),self.gs_z]<safe_sdf)
        unsafe_tgt += (sdf[self.gs_x,self.gs_y,(self.gs_z+1).clamp(0,self.Nz-1)]<safe_sdf) + \
                      (sdf[self.gs_x,self.gs_y,(self.gs_z-1).clamp(0,self.Nz-1)]<safe_sdf)
        unsafe_tgt = unsafe_tgt.reshape(-1)
        valid_mask[unsafe_tgt,:] = False  # remove points that are not safe to go

        ##################################################
        ### visibility check
        ##################################################
        near_view_vec = view_vec[valid_mask]  # [x*y*z*near, 3]
        t_values = torch.linspace(0, 1, 30, device=near_view_vec.device)
        vis_test_points = goal_space_pts[valid_mask][..., None] - t_values * near_view_vec[..., None]  # [x*y*z*near, 3, 20]
        vis_test_points = vis_test_points.permute(0, 2, 1).long()  # [x*y*z*near, 20, 3]
        vis_test_sdf = sdf[vis_test_points[:,:,0], vis_test_points[:,:,1], vis_test_points[:,:,2]]  # [x*y*z*near, 20]
        ## Visibility check is valid only if sampled points are all with positive sdf ##
        vis_test_sdf, _ = vis_test_sdf.min(dim=1) 
        visible_mask = vis_test_sdf > 0
        if visible_mask.sum() == 0:
            self.info_printer(
                "   Warning! No visible uncertainty points.",
                self.step,
                self.__class__.__name__ 
                )
        
        ### warning message ###
        valid_mask = valid_mask.masked_scatter(valid_mask.clone(), visible_mask)
        if valid_mask.sum() == 0:
            self.info_printer(
                "   Warning! No valid uncertainty points. either too far or non-visible",
                self.step,
                self.__class__.__name__ 
                )
            invalid_goal_space = True
        else:
            invalid_goal_space = False

        ##################################################
        ### aggregate uncertainty
        ##################################################
        topk_uncert_vxl = topk_uncert_vxl.long()
        k_uncerts = uncert[topk_uncert_vxl[:,0], topk_uncert_vxl[:,1], topk_uncert_vxl[:,2]]  # [k]
        k_uncerts = k_uncerts[None, :].repeat(goal_space_pts.shape[0], 1) # [x*y*z, k]
        gs_uncert_collections = torch.zeros_like(k_uncerts)
        gs_uncert_collections[valid_mask] = k_uncerts[valid_mask]
        gs_aggre_uncerts = gs_uncert_collections.sum(dim=1)
        gs_aggre_uncerts = gs_aggre_uncerts.reshape([self.gs_x_range.shape[0], self.gs_y_range.shape[0], self.gs_z_range.shape[0]])

        ### warning message ###
        if torch.count_nonzero(gs_aggre_uncerts) == 0:
            self.info_printer(
                "   Warning! No valid goal space. Going to filter search space. ",
                self.step,
                self.__class__.__name__ 
                )

        ##################################################
        ### gather outputs
        ##################################################
        outputs = {
            'gs_aggre_uncerts': gs_aggre_uncerts, 
            'topk_uncert_vxl': topk_uncert_vxl, 
            'gs_uncert_collections': gs_uncert_collections, 
        }

        if invalid_goal_space:
            if force_running:
                return True, outputs
            else:
                return False, {}
        else:
            return True, outputs

    def rotating_at_current_loc(self, cur_pose: np.ndarray) -> np.ndarray:
        """ rotating at the current location using the rotations in self.rots. 
    
        Args:
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system
        """
        rot = self.rots.pop(0)
        new_pose = cur_pose.copy()
        new_pose[:3, :3] = rot
        return new_pose
    
    def rotating_at_start(self, cur_pose: np.ndarray) -> np.ndarray:
        """ rotating at the starting location using the rotations in self.rots. 
    
        Args:
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system
        """
        return self.rotating_at_current_loc(cur_pose)
    
    def rotating_at_goal(self, cur_pose: np.ndarray) -> np.ndarray:
        """ rotating at the goal location using the rotations in self.rots. 
    
        Args:
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system
        """
        return self.rotating_at_current_loc(cur_pose)

    def rotation_planning_at_start(self, 
                                   cur_pose  : np.ndarray,
                                   lookat_loc: np.ndarray
                                   ) -> np.ndarray:
        """ perform rotation planning
    
        Args:
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
            lookat_loc (np.ndarray, [3]): look-at location
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system

        Attributes:
            rots (List): planned rotations. each element is (np.ndarray, [3,3])
        """
        rot = compute_camera_pose(cur_pose[:3, 3], lookat_loc, up_dir=self.planner_cfg.up_dir)
        self.rots = rotation_planning(cur_pose[:3, :3], [rot], self.planner_cfg.max_rot_deg)

        new_pose = cur_pose.copy()
        return new_pose

    def rotation_planning_at_goal(self,
                                  cur_pose   : np.ndarray,
                                  lookat_locs: np.ndarray
                                  ) -> np.ndarray:
        """ perform rotation planning
    
        Args:
            cur_pose    (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
            lookat_locs (List)             : look-at locations. Each element is (np.ndarray, [3])
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system
        
        Attributes:
            rots (List): planned rotations. each element is (np.ndarray, [3,3])
        """
        self.rots = []
        for lookat_loc in lookat_locs:
            rot = compute_camera_pose(cur_pose[:3, 3], lookat_loc, up_dir=self.planner_cfg.up_dir)
            self.rots.append(rot)
        self.rots = rotation_planning(cur_pose[:3, :3], self.rots, self.planner_cfg.max_rot_deg)
        
        new_pose = cur_pose.copy()
        return new_pose

    def moving_to_goal(self, 
                       cur_pose    : np.ndarray,
                       lookat_loc  : np.ndarray,
                       next_pt_node: Node,
                       ) -> np.ndarray:
        """ moving to goal while looking at lookat_loc
    
        Args:
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
            lookat_loc (np.ndarray, [3]): look-at location
            next_pt_node (Node)         : next_pt node
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system
        """
        next_loc = self.vox2loc(next_pt_node._xyz_arr)
        rot = compute_camera_pose(next_loc, lookat_loc, up_dir=self.planner_cfg.up_dir)

        new_pose = cur_pose.copy()
        new_pose[:3, :3] = rot
        new_pose[:3, 3] = next_loc
        return new_pose
    
    def check_goal_reachable(self) -> bool:
        """ Check if goal is reachable
        Returns:
            is_goal_reachable (bool): is goal reachable
        """
        is_goal_reachable = self.is_goal_reachable
        return is_goal_reachable

    def check_rotation_done(self) -> bool:
        """
        Returns:
            is_rotation_done (bool): is rotation done        
        """
        is_rotation_done = len(self.rots) == 0
        return is_rotation_done
    
    def check_goal_reached(self) -> bool:
        """ check if goal is reached
        Returns:
            is_goal_reached (bool): is goal reached
            
        """
        is_goal_reached = len(self.path) == 0
        return is_goal_reached
    
    def check_new_map_received(self, is_new_vols):
        """ check if new map is received.
    
        Args:
            is_new_vols (bool): is new volumes/map received

        Returns:
            is_new_vols (bool): is new volumes/map received        
            
        """
        return is_new_vols
    