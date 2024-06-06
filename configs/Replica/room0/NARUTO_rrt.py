import numpy as np
import os

_base_ = "../../default.py"

##################################################
### NARUTO (General)
##################################################
general = dict(
    dataset = "Replica",
    scene = "room0",
    num_iter = 2000,
)

##################################################
### Directories
##################################################
dirs = dict(
    data_dir = "data/",
    result_dir = "results/",
    cfg_dir = os.path.join("configs", general['dataset'], general['scene'])
)


##################################################
### Simulator
##################################################
if _base_.sim["method"] == "habitat":
    _base_.sim.update(
        habitat_cfg = os.path.join(dirs['cfg_dir'], "habitat.py")
    )

##################################################
### SLAM
##################################################
if _base_.slam["method"] == "coslam":
    _base_.slam.update(
        room_cfg        = f"{dirs['cfg_dir']}/coslam.yaml",     # Co-SLAM room configuration
        enable_active_planning = True,                                 # enable/disable active planning
        active_ray      = True,                                 # enable/disable active ray sampling
        
        use_traj_pose = False,                                   # use pre-defined trajectory pose
        SLAMData_dir = os.path.join(                            # SLAM Data directory (for passive mapping or pre-defined trajectory pose)
            dirs["data_dir"], 
            "Replica", general['scene']
            ),
        
        start_c2w = np.eye(4)
    )

##################################################
### Planner
##################################################
planner = dict(
    ### RRT ###
    local_planner_method = "RRT",             # RRT method
    up_dir = np.array([0, 0, 1]), # up direction for planning pose
)

### NARUTO Planner ###
if planner["local_planner_method"] == "RRT":
    planner.update(
        rrt_step_size = _base_.planner['step_size'] / _base_.planner['voxel_size'], # Unit: voxel
        rrt_step_amplifier = 10,                    # rrt step amplifier to fast expansion
        rrt_maxz = 100,                             # Maximum Z-level to limit the RRT nodes. Unit: voxel
        rrt_max_iter = None,                        # maximum iterations for RRT
        rrt_z_levels = None,                        # Z levels for sampling RRT nodes. Unit: voxel. Min and Max level
    )


##################################################
### Visualization
##################################################
visualizer = dict(
    vis_rgbd        = True,                             # visualize RGB-D

    ### mesh related ###
    mesh_vis_freq = 500,                                # mesh save frequency
)

