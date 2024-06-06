import numpy as np
import os

_base_ = "../../default.py"

##################################################
### NARUTO (General)
##################################################
general = dict(
    dataset = "Replica",
    scene = "room2",
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
        room_cfg        = f"{dirs['cfg_dir']}/coslam.yaml",   # Co-SLAM room configuration
        enable_active_planning = True,                             # enable/disable active planning
        active_ray      = True,                             # enable/disable active ray sampling
        
        use_traj_pose = False,                          # use pre-defined trajectory pose
        SLAMData_dir = os.path.join(                    # SLAM Data directory (for passive mapping or pre-defined trajectory pose)
            dirs["data_dir"], 
            "Replica", general['scene']
            ),
        
        start_c2w = np.eye(4)
    )

##################################################
### Planner
##################################################
planner = dict(
    up_dir = np.array([0, 0, 1]), # up direction for planning pose
)

##################################################
### Visualization
##################################################
visualizer = dict(
    vis_rgbd        = True,                             # visualize RGB-D

    ### mesh related ###
    mesh_vis_freq = 500,                                # mesh save frequency
)

