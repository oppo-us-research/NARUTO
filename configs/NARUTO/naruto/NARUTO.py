import numpy as np
import os

_base_ = "../../default.py"

##################################################
### NARUTO (General)
##################################################
general = dict(
    dataset = "NARUTO",
    scene = "naruto",
    num_iter = 5000,
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
        voxel_size = 0.02,                               # Voxel size for Uncertainty/SDF volume. Unit: meter
        
        use_traj_pose = False,                          # use pre-defined trajectory pose
        SLAMData_dir = None,
        
        start_c2w = np.array([
            [ 1,  0,  0,  0],
            [ 0,  0,  -1,  -2.4],
            [ 0,  1,  0,  0],
            [ 0,  0,  0,  1]]).astype(np.float32)
            )

##################################################
### Planner
##################################################
planner = dict(
    up_dir = np.array([0, 0, 1]),
    voxel_size = 0.02,       # Uncertainty Volume voxel size. Unit: meter
)

##################################################
### Visualization
##################################################
visualizer = dict(
    vis_rgbd        = True,                             # visualize RGB-D

    ### mesh related ###
    mesh_vis_freq = 500,                                # mesh save frequency
    enable_all_vis       = True,                   # enable comprehensive visualization data
    save_mesh_voxel_size = 0.02,                    # voxel size to save mesh for visualization
)

