import os
import numpy as np

##################################################
### NARUTO (General)
##################################################
general = dict(
    seed     = 0,                                       # random seed
    dataset  = "Replica",                               # dataset name
    scene    = "office0",                               # scene name
    num_iter = 2000,                                    # number of optimization iteration
)

##################################################
### Directories
##################################################
dirs = dict(
    data_dir   = "data/",                               # data directory
    result_dir = "results/",                            # result directory
    cfg_dir    = os.path.join(                          # configuration directory
        "configs", 
        general['dataset'], 
        general['scene']
        )
)

##################################################
### Simulator
##################################################
sim = dict(
    method = "habitat"                                  # simulator method
)

## HabitatSim ##
if sim["method"] == "habitat":
    sim.update(
        habitat_cfg = os.path.join(                     # HabitatSim configuration
            dirs['cfg_dir'], 
            "habitat.py"
            ),
    )

##################################################
### SLAM Model
##################################################
slam = dict(
    method="coslam"                                     # SLAM backbone method
)

if slam["method"] == "coslam":
    slam.update(
        room_cfg = f"{dirs['cfg_dir']}/coslam.yaml",    # Co-SLAM room configuration
        enable_active_planning = True,                  # enable/disable active planning
        enable_active_ray = True,                       # enable/disable active ray sampling
        
        use_traj_pose = False,                          # use pre-defined trajectory pose
        SLAMData_dir = os.path.join(                    # SLAM Data directory (for passive mapping or pre-defined trajectory pose)
            dirs["data_dir"], 
            "Replica", general['scene']
            ),

        start_c2w = np.eye(4),                          # initial camera pose

        ### volumes ###
        voxel_size = 0.1,                               # Voxel size for Uncertainty/SDF volume. Unit: meter
    )

    ### Active Ray Sampling ###
    if slam.get('active_ray', True):
        slam.update(
            act_ray_oversample_mul = 4,                 # oversampling multiplier
            act_ray_num_uncert_sample = 500,            # number of unceratinty samples for replacing original samples
        )


##################################################
### Planner
##################################################
planner = dict(
    method        = "naruto",                           # planner method
    enable_timing = False,                              # enable timing message
)

### NARUTO Planner ###
if planner["method"] == "naruto":
    planner.update(
        step_size = 0.1,                                # step size. Unit: meter

        ### Uncertainty Volume ###
        voxel_size = slam.get("voxel_size", 0.1),       # Uncertainty Volume voxel size. Unit: meter
        
        ### Goal Space (Uncertainty Aggregation) ###
        uncert_top_k        = 4000,                     # number of top-k uncertainty to be considered in Goal Space
        uncert_top_k_subset = 300,                      # subset number of top-k uncertainty to be considered in Goal Space. Choose randomly from uncert_top_k to avoid Uncertainty Point Concentration
        gs_sensing_range    = [0.5, 2],                 # meter. goal space sensing range
        safe_sdf            = 0.8,                      # Unit: voxel
        force_uncert_aggre  = False,                    # force running uncertainty aggregation
        gs_z_levels         = None,                     # goal space z levels

        ### path planning ###
        obs_per_goal            = 10,                   # maximum uncertain observation per goal
        enable_uncert_filtering = True,                 # filter uncertainty volume according to the traversability
        up_dir                  = np.array([0, 0, 1]), # up direction for planning pose

        ### RRT ###
        local_planner_method = "RRTNaruto",             # RRT method

        ### Collision ###
        invalid_region_ratio_thre = 0.5,                # invalid region ratio threshold by checking ERP
        collision_dist_thre       = 0.05,               # Unit: meter

        ### Rotation planning ###
        max_rot_deg = 10,                               # degree

    )

    if planner["local_planner_method"] == "RRTNaruto":
        planner.update(
            rrt_step_size = planner['step_size'] / planner['voxel_size'], # Unit: voxel
            rrt_step_amplifier = 10,                    # rrt step amplifier to fast expansion
            rrt_maxz = 100,                             # Maximum Z-level to limit the RRT nodes. Unit: voxel
            rrt_max_iter = None,                        # maximum iterations for RRT
            rrt_z_levels = None,                        # Z levels for sampling RRT nodes. Unit: voxel. Min and Max level
            enable_eval = False,                        # enable RRT evaluation
            enable_direct_line = True,                  # enable direct connection attempt
        )

##################################################
### Visualization
##################################################
visualizer = dict(
    vis_rgbd        = True,                             # visualize RGB-D
    
    ### mesh related ###
    mesh_vis_freq = 500,                                # mesh save frequency

    ### comprehensive visualizer ###
    method = "naruto"                                   # comprehensive visualizer method
)

if visualizer["method"] == "naruto":
    visualizer.update(
        enable_all_vis       = False,                   # enable comprehensive visualization data
        save_rgbd            = True,                    # save RGB-D data
        save_pose            = True,                    # save pose
        save_planning_path   = True,                    # save planning path
        save_lookat_tgts     = True,                    # save uncertain target observation locations
        save_state           = True,                    # save planner state
        save_color_mesh      = True,                    # save colored mesh
        save_uncert_mesh     = True,                    # save uncertainty mesh
        save_mesh_freq       = 5,                       # frequency to save mesh for visualization
        save_mesh_voxel_size = 0.05,                    # voxel size to save mesh for visualization
    )