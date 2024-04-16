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


import os
import sys
sys.path.append(os.getcwd())

from src.data.pose_loader import habitat_pose_conversion, PoseLoader
from src.naruto.cfg_loader import argument_parsing, load_cfg
from src.planner import init_planner
from src.slam import init_SLAM_model
from src.simulator import init_simulator
from src.utils.timer import Timer
from src.utils.general_utils import fix_random_seed, InfoPrinter, update_module_step
from src.visualization import init_visualizer


if __name__ == "__main__":
    info_printer = InfoPrinter("NARUTO")
    timer = Timer()

    ##################################################
    ### argument parsing and load configuration
    ##################################################
    info_printer("Parsing arguments...", 0, "Initialization")
    args = argument_parsing()
    info_printer("Loading configuration...", 0, "Initialization")
    main_cfg = load_cfg(args)
    info_printer.update_total_step(main_cfg.general.num_iter)
    info_printer.update_scene(main_cfg.general.dataset + " - " + main_cfg.general.scene)

    ##################################################
    ### Fix random seed
    ##################################################
    info_printer("Fix random seed...", 0, "Initialization")
    fix_random_seed(main_cfg.general.seed)
    
    ##################################################
    ### initialize simulator
    ##################################################
    sim = init_simulator(main_cfg, info_printer)

    ##################################################
    ### initialize SLAM module
    ##################################################
    slam = init_SLAM_model(main_cfg, info_printer)

    ##################################################
    ### initialize planning module
    ##################################################
    planner = init_planner(main_cfg, info_printer)
    planner.update_sim(sim)
    planner.init_data(slam.config['mapping']['bound'])
    planner.init_local_planner()

    ##################################################
    ### initialize visualizer
    ##################################################
    visualizer = init_visualizer(main_cfg, info_printer)

    ##################################################
    ### Run NARUTO
    ##################################################
    ## load initial pose ##
    pose_loader = PoseLoader(main_cfg)
    c2w_slam = pose_loader.load_init_pose()

    for i in range(main_cfg.general.num_iter):
        ##################################################
        ### update module infomation (e.g. step)
        ##################################################
        update_module_step(i, [sim, slam, planner, visualizer])

        ##################################################
        ### load pose and transform pose
        ##################################################
        c2w_slam = pose_loader.update_pose(c2w_slam, i)
        c2w_sim = c2w_slam.cpu().numpy().copy()
        
        ##################################################
        ### Simulation
        ##################################################
        timer.start("Simulation", "General")
        color, depth = sim.simulate(c2w_sim)
        if main_cfg.visualizer.vis_rgbd:
            visualizer.visualize_rgbd(color, depth, slam.config["cam"]["depth_trunc"])
        timer.end("Simulation")
        
        ##################################################
        ### save data for comprehensive visualization
        ##################################################
        if main_cfg.visualizer.enable_all_vis:
            visualizer.main(slam, planner, color, depth, c2w_slam)

        ##################################################
        ### Mapping optimization
        ##################################################
        timer.start("SLAM", "General")
        new_uncert_sdf_vols = slam.online_recon_step(i, color, depth, c2w_slam)
        timer.end("SLAM")
        
        ##################################################
        ### Active Planning
        ##################################################
        if main_cfg.slam.enable_active_planning:
            timer.start("Planning", "General")
            ### update map volumes ###
            if new_uncert_sdf_vols is not None:
                uncert_sdf = new_uncert_sdf_vols
                is_new_vols = True
            else:
                is_new_vols = False
            c2w_slam = planner.main(
                uncert_sdf, 
                c2w_slam.cpu().numpy(), 
                is_new_vols
                )
            timer.end("Planning")

    ##################################################
    ### Save Final Mesh and Checkpoint
    ##################################################
    slam.save_mesh(main_cfg.general.num_iter, voxel_size=slam.config['mesh']['voxel_final'], suffix='_final')
    slam.save_ckpt(main_cfg.general.num_iter, suffix="_final")

    ##################################################
    ### Runtime Analysis
    ##################################################
    timer.time_analysis()
