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


import cv2
import habitat_sim
from habitat_sim.utils import viz_utils as vut
import mmengine
import numpy as np
import os
import quaternion
import shutil
from tqdm import tqdm
from typing import List, Dict, Tuple, Union
import _magnum

def init_camera_spec(
        cam_cfg    : mmengine.ConfigDict,
        cam_id     : str,
        orientation: List[float],
        fov        : Union[float, Tuple[float, float]] = None,
        cam_subtype: str = "pinhole",
        cam_type   : str = "color",
        ) -> habitat_sim.CameraSensorSpec:
    """ Initialize Camera Spec

    Args:
        cam_cfg (mmengine.ConfigDict): Camera configurations
        cam_id (str)                 : camera ID
        orientation (List[float])    : camera orientation (rotation)
        fov (Union[float, Tuple])    : camera FoV. [fov_vertical, fov_horizontal] if Tuple is provided
        cam_type (str)               : camera type
            - color
            - depth
        cam_subtype (str)            : camera type
            - pinhole
            - erp

    Returns:
        cam_spec (habitat_sim.CameraSensorSpec): camera sensor spec
    """
    cam_spec = {
        'pinhole': habitat_sim.CameraSensorSpec(),
        'erp': habitat_sim.EquirectangularSensorSpec()
    }[cam_subtype]
    cam_spec.uuid = "{}_{}".format(cam_subtype, cam_type)
    cam_spec.uuid += "_{}".format(cam_id) if cam_id is not None else ""
    cam_spec.sensor_subtype = {
        'pinhole': habitat_sim.SensorSubType.PINHOLE,
        'erp': habitat_sim.SensorSubType.EQUIRECTANGULAR,
    }[cam_subtype]
    cam_spec.sensor_type = {
        'color': habitat_sim.SensorType.COLOR,
        'depth': habitat_sim.SensorType.DEPTH,
    }[cam_type]
    cam_spec.resolution = cam_cfg.resolution_hw
    cam_spec.position = [0.0, 0.0, 0.0]
    cam_spec.orientation = orientation
    if fov:
        if type(fov) == float:
            cam_spec.hfov = fov
        else:
            cam_spec.vfov = fov[0] # height
            cam_spec.hfov = fov[1] # width
    return cam_spec


def setup_pinhole_cams(cam_cfg: mmengine.ConfigDict) -> List[habitat_sim.CameraSensorSpec]:
    """ Setup PinHole cameras

    Args:
        cam_cfg (mmengine.ConfigDict): PinHole camera configurations

    Returns:
        cam_specs (List[habitat_sim.CameraSensorSpec]): camera specs
    """
    
    ### initialize orientation ###
    orientations = {}
    if cam_cfg.orientation_type == 'skybox':
        orientations = {
            "front": [0.0, 0.0, 0.0],      # front
            "back" : [0.0, np.pi, 0.0],    # back
            "left" : [0.0, np.pi/2, 0.0],  # left
            "right": [0.0, -np.pi/2, 0.0], # right
            "up"   : [np.pi/2, 0.0, 0.0],  # up
            "down" : [-np.pi/2, 0.0, 0.0], # down
        }
    elif "horizontal" in cam_cfg.orientation_type:
        delta_rot = 2 * np.pi / cam_cfg.horizontal.num_rot
        for i in range(cam_cfg.horizontal.num_rot):
            orientations["{:03}".format(np.round(np.rad2deg(delta_rot*i)))] = [0.0, delta_rot*i, 0.0]
    elif cam_cfg.orientation_type == "horizontal+UpDown":
        orientations["up"] = [np.pi/2, 0.0, 0.0],  # up
        orientations["down"] = [-np.pi/2, 0.0, 0.0], # down
    else:
        raise NotImplementedError
    
    ### setup camera specs ###
    sensor_specs = []
    for ori_key, orientation in orientations.items():
        ### RGBA Camera ###
        cam_spec = init_camera_spec(
                    cam_cfg     = cam_cfg,
                    cam_id      = ori_key,
                    orientation = orientation,
                    fov         = cam_cfg.fov,
                    cam_subtype = "pinhole",
                    cam_type    = "color"
            )
        sensor_specs.append(cam_spec)
        
        ### Depth camera ###
        cam_spec = init_camera_spec(
                    cam_cfg     = cam_cfg,
                    cam_id      = ori_key,
                    orientation = orientation,
                    fov         = cam_cfg.fov,
                    cam_subtype = "pinhole",
                    cam_type    = "depth"
            )
        sensor_specs.append(cam_spec)

    return sensor_specs


def setup_equirectangular_cams(cam_cfg: mmengine.ConfigDict) -> List[habitat_sim.CameraSensorSpec]:
    """ Setup Equirectangular cameras

    Args:
        cam_cfg (mmengine.ConfigDict): Equirectangular camera configurations

    Returns:
        cam_specs (List[habitat_sim.CameraSensorSpec]): camera specs
    """
    sensor_specs = []
    ### RGBA Camera ###
    if 'color' in cam_cfg.cam_type:
        cam_spec = init_camera_spec(
                    cam_cfg     = cam_cfg,
                    cam_id      = None,
                    orientation = [0.0, 0.0, 0.0],
                    cam_subtype = "erp",
                    cam_type    = "color"
        )
        sensor_specs.append(cam_spec)
    
    ### Depth camera ###
    if 'depth' in cam_cfg.cam_type:
        cam_spec = init_camera_spec(
                    cam_cfg     = cam_cfg,
                    cam_id      = None,
                    orientation = [0.0, 0.0, 0.0],
                    cam_subtype = "erp",
                    cam_type    = "depth"
        )
        sensor_specs.append(cam_spec)
    return sensor_specs


def make_configuration(cfg: mmengine.Config) -> habitat_sim.simulator.Configuration:
    """ Create Simulation Configuration

    Args:
        cfg (mmengine.Config): configuration

    Returns:
        sim_cfg (habitat_sim.Configuration): Habitat Simulator Configuration
    """
    ### simulator configuration ###
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = cfg.simulator.scene_id
    assert os.path.exists(backend_cfg.scene_id)
    backend_cfg.enable_physics = cfg.simulator.physics.enable

    ##################################################
    ### Add camera sensor spects
    ##################################################
    sensor_specs = []
    if 'pinhole' in cfg.camera and cfg.camera.pinhole.enable:
        sensor_specs += setup_pinhole_cams(cfg.camera.pinhole)
    
    if 'equirectangular' in cfg.camera and cfg.camera.equirectangular.enable:
        sensor_specs += setup_equirectangular_cams(cfg.camera.equirectangular)
    
    assert len(sensor_specs) > 0, "No sensor spec is added!"

    ##################################################
    ### agent configuration
    ##################################################
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def init_multiview_locations(r: float, num_sample: int) -> Dict[int, np.ndarray]:
    """ initialize multiview locations

    Args:
        r (float)       : radius
        num_sample (int): number of samples

    Returns:
        locations (Dict[int, np.ndarray]): multiview locations
    """
    cnt = 0
    locations = {}
    for delta1 in np.linspace(-r,r,num_sample):
        for delta2 in np.linspace(-r,r,num_sample):
            for delta3 in np.linspace(-r,r,num_sample):
                locations[cnt] = np.array([delta1, delta2, delta3])
                cnt += 1
    return locations


def get_pose_matrix_from_agent_state(agent_state: habitat_sim.agent.agent.AgentState) -> np.ndarray:
    """ Get Pose Matrix from Agent State

    Args:
        agent_state (habitat_sim.Agentstate): agent state

    Returns:
        pose (np.ndarray, [4,4]): w2c pose
    """
    pose = np.eye(4)
    pose[:3, :3] = quaternion.as_rotation_matrix(agent_state.rotation)
    pose[:3, 3] = agent_state.position
    return pose


def place_agent(
        sim      : habitat_sim.Simulator,
        agent_cfg: mmengine.ConfigDict,
        ) -> _magnum.Matrix4:
    """ Place agent
    
    Args:
        sim (habitat_sim.Simulator)    : Simulator
        agent_cfg (mmengine.ConfigDict): agent configuration
    
    Returns:
        agent_pose (_magnum.Matrix4): world-to-camera pose
    """
    ### get agent position shift ###
    if 'multiview_sim' in agent_cfg:
        mv_sim_cfg = agent_cfg.multiview_sim
        mv_shifts = init_multiview_locations(mv_sim_cfg.radius, mv_sim_cfg.num_sample)
        position_shift = mv_shifts[mv_sim_cfg.location_idx]
    else:
        position_shift = np.array([0., 0., 0.])

    ### initialize agent pose ###
    agent_state = habitat_sim.AgentState()
    agent_state.position = list(agent_cfg.position)
    agent_state.rotation = quaternion.from_rotation_vector(agent_cfg.rotation)
    agent_w2c = get_pose_matrix_from_agent_state(agent_state)

    ### shift the agent to multiview location  ###
    ### T_{c1->w} @ T_{c2->c1} -> T_{c2->w} ###
    shift_transform = np.eye(4)
    shift_transform[:3, 3] = position_shift 
    agent_w2c = agent_w2c @ shift_transform 

    ### shift the agent to right (for stereo) ###
    if 'is_right' in agent_cfg and agent_cfg.is_right:
        shift_transform = np.eye(4)
        shift_transform[:3, 3] = np.array([agent_cfg.right_shift, 0, 0])
        agent_w2c = agent_w2c @ shift_transform 

    ### update Agent position ###
    agent_state.position = list(agent_w2c[:3, 3])

    ### initialize Agent ###
    agent = sim.initialize_agent(0, agent_state)
    return agent.scene_node.transformation_matrix()


def SixDOFPose2Mat(state: habitat_sim.SixDOFPose) -> np.ndarray:
    """

    Args:
        state (habitat_sim.SixDOFPose): camera-to-world
        
    Returns:
        pose (np.ndarray, [4,4]): camera to world
        
    """

    ''' 
        coordinate system transformation for camera pose

        current system: [X: right; Y: up; Z: backward] 
        desired system: [X: right; Y: down; Z: forward]

        let the desired pose be T_w'c' (camera'-to-world')
        the current pose be T_wc

        T_w'c'  = T_w'w @ T_wc @ T_cc' 
                = T_r @ T_wc @ (T_r)^-1
                where T_r = [
                    1, 0, 0,  0
                    0, -1, 0, 0
                    0, 0, -1, 0
                    0, 0, 0,  1
                ]

    '''
    pose = np.eye(4)
    pose[:3, 3] = state.position
    pose[:3,:3] = quaternion.as_rotation_matrix(state.rotation)

    T = np.eye(4)
    T[1,1] = -1
    T[2,2] = -1

    pose = T @ (pose @ np.linalg.inv(T))
    return pose


def simulate_objects(
        sim      : habitat_sim.Simulator,
        obj_cfg  : mmengine.ConfigDict,
        agent_cfg: mmengine.ConfigDict,
    ) -> None:
    """ Simulate objects

    Args:
        sim (habitat_sim.Simulator)    : Simulator
        obj_cfg (mmengine.ConfigDict)  : object configuration
        agent_cfg (mmengine.ConfigDict): agent configuration
    
    """
    ##################################################
    ### initialize object managers
    ##################################################
    ### get the physics object attributes manager ###
    obj_templates_mgr = sim.get_object_template_manager()

    ### get the rigid object manager ###
    rigid_obj_mgr = sim.get_rigid_object_manager()

    ##################################################
    ### initialize objects
    ##################################################
    ### load some object templates from configuration files ###
    template_ids = {}
    for obj_temp in obj_cfg.obj_templates:
        template_ids[obj_temp] = obj_templates_mgr.load_configs(
            str(os.path.join(obj_cfg.obj_dir, obj_temp))
            )[0]
    
    ### initialize objects ###
    location_profiles = obj_cfg.obj_profiles.location if 'location' in obj_cfg.obj_profiles else None
    velocity_profiles = obj_cfg.obj_profiles.velocity if 'velocity' in obj_cfg.obj_profiles else None
    angular_velocity_profiles = obj_cfg.obj_profiles.angular_velocity if 'angular_velocity' in obj_cfg.obj_profiles else None
    rotation_profiles = obj_cfg.obj_profiles.rotation if 'rotation' in obj_cfg.obj_profiles else None

    j = 0
    for i, obj_temp in enumerate(obj_cfg.obj_templates):
        num_obj = obj_cfg.num_objs[i]
        for _ in range(num_obj):
            ### add objs to the scene, returns the object ###
            obj = rigid_obj_mgr.add_object_by_template_id(template_ids[obj_temp])

            ### put object in front of the front camera ###
            if location_profiles:
                ## get front camera pose @ center location  ##
                ff_camera_c2w = np.eye(4)
                front_cam_rot = quaternion.from_rotation_vector(agent_cfg.rotation)
                ff_camera_c2w[:3, :3] = quaternion.as_rotation_matrix(front_cam_rot)
                ff_camera_c2w[:3, 3] = agent_cfg.position

                ## get position of obj (camera coordiante system) ##
                loc_cam = np.ones((4,1))
                loc_cam[:3, 0] = np.asarray(location_profiles[j])
                
                ## get position of obj (world coordinate system) ##
                loc_world = ff_camera_c2w @ loc_cam
                obj.translation = loc_world.reshape(4)[:3]
            else:
                obj.translation = np.zeros(3)
            
            ### convert velocity so it is w.r.t front camera ###
            if velocity_profiles:
                vel_cam = np.asarray(velocity_profiles[j]).astype(np.float32)
                vel_world = ff_camera_c2w[:3, :3] @ vel_cam
                obj.linear_velocity = vel_world
            else:
                obj.linear_velocity = np.zeros(3)

            ### convert angular_velocity so it is w.r.t front camear ###
            if angular_velocity_profiles:
                vel_cam = np.asarray(angular_velocity_profiles[j]).astype(np.float32)
                vel_world = ff_camera_c2w[:3, :3] @ vel_cam
                obj.angular_velocity = vel_world
            else:
                obj.angular_velocity = np.zeros(3)

            ### object rotation ###
            if rotation_profiles:
                obj.rotate_local(_magnum.Rad(np.deg2rad(rotation_profiles[j][0])), _magnum.Vector3(np.asarray(rotation_profiles[j][1:]).astype(np.float32)))
            else:
                obj.rotate_local(_magnum.Rad(0.), _magnum.Vector3(np.array([1., 0., 0.])))
            j += 1


def get_next_spiral_pose(
            agent     : habitat_sim.Agent,
            timestamp : float,
            spiral_rad: float,
            move_dir  : str = 'x'
        ) -> habitat_sim.SixDOFPose:
    """ get spiral pose

    Args:
        agent (habitat_sim.Agent): Agent
        timestamp (float)        : timestamp [0, 1]
        spiral_rad (float)       : spiral motion radius
        move_dir (str)           : move direction

    Returns:
        new_state (habitat_sim.SixDOFPose): new state
    """
    new_state = habitat_sim.agent.AgentState()
    new_state.position = agent.initial_state.position.copy()
    new_state.rotation = agent.initial_state.rotation.copy()
    if move_dir == 'x':
        new_state.position[1] += spiral_rad * np.cos(((timestamp) * 360.0)/180.*np.pi)
        new_state.position[2] += spiral_rad * np.sin(((timestamp) * 360.0)/180.*np.pi)
        new_state.position[0] += spiral_rad * timestamp
    elif move_dir == 'y':
        new_state.position[2] += spiral_rad * np.cos(((timestamp) * 360.0)/180.*np.pi)
        new_state.position[0] += spiral_rad * np.sin(((timestamp) * 360.0)/180.*np.pi)
        new_state.position[1] += spiral_rad * timestamp
    return new_state


def get_hori_orientation(
            init_rot: quaternion.quaternion,
            num_rot : int,
            i       : int
        ) -> quaternion.quaternion:
    """ get rotation (quaternion) along y-axis, i.e. horizontal rotation over 360 deg

    Args:
        init_rot (quaternion.quaternion): initial rotation
        num_rot (int)                   : number of rotations
        i (int)                         : current rotation index

    Returns:
        rot (quaternion.quaternion): current rotation
        
    """
    euler = quaternion.as_euler_angles(init_rot)
    delta_rot = 2*np.pi/num_rot
    euler[1] += delta_rot*i
    rot = quaternion.from_euler_angles(euler)
    return rot


def agent_motion_simulation(
            agent: habitat_sim.Agent,
            agent_cfg: mmengine.ConfigDict,
            cnt : int,
            num_frames: int
    ) -> habitat_sim.agent.AgentState: 
    """

    Args:
        agent (habitat_sim.agent.Agent): agent
        agent_cfg (mmengine.ConfigDict): agent configuration
        cnt (int)                      : current frame index
        num_frames (int)               : total number of frames
    
    Returns:
        next_state (habitat_sim.agent.AgentState): next state
    """
    ### initialize parameters ###
    shift_radius = agent_cfg.motion_profile.radius
    motion_type = agent_cfg.motion_profile.motion_type
    timestamp = cnt / num_frames

    ### initialize AgentState ###
    next_state = habitat_sim.agent.AgentState()

    if motion_type == 'stationary':
        next_state = agent.get_state()

    elif motion_type == 'random':
        ### random state for testing scenes ###
        next_state.position = agent.initial_state.position + (np.random.rand(3) * (2*shift_radius) - shift_radius)
        next_state.rotation = agent.initial_state.rotation.copy()
    
    elif motion_type == 'spiral_forward':
        move_dir = agent_cfg.motion_profile.spiral_move_dir
        next_state = get_next_spiral_pose(agent, timestamp, shift_radius, move_dir)
    
    elif motion_type == 'spiral_forward_with_hori_rot':
        move_dir = agent_cfg.motion_profile.spiral_move_dir
        next_state = get_next_spiral_pose(agent, timestamp, shift_radius, move_dir)
        next_state.rotation = get_hori_orientation(
                    agent.initial_state.rotation.copy(), 
                    num_frames, 
                    cnt
        )
    elif motion_type == 'forward':
        next_state.position = agent.get_state().position + np.array([0.0, 0.0, - timestamp * shift_radius])
    elif motion_type == 'predefined':
        traj_txt = agent_cfg.motion_profile.predefined_traj_txt
        with open(traj_txt, 'r') as f:
            ### read pose ###
            line = f.readlines()[cnt]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4) # RDF, c2w

            ### convert pose from RDF to RUB ###
            T = np.eye(4)
            T[1,1] = -1.
            T[2,2] = -1.
            c2w = T @  ( c2w @ np.linalg.inv(T) ) # RUB, c2w

            ### convert pose from matrix to quaternion ###
            qut = quaternion.from_rotation_matrix(c2w[:3, :3])
            trans = c2w[:3, 3]
            next_state.position = trans
            next_state.rotation = qut
    else:
        raise NotImplementedError

    return next_state


def simulate(
        sim: habitat_sim.Simulator,
        cfg: mmengine.Config,
    ) -> Tuple[List, List]:
    """ Run Simulation

    Args:
        sim (habitat_sim.Simulator): Simulator
        cfg (mmengine.Config)      : configuration
    
    Returns:
        observations (List[Dict]): sensor observations
        poses (List[Dict])       : sensor poses
    """
    agent_cfg = cfg.agent
    sim_cfg = cfg.simulator
    dt = sim_cfg.duration
    FPS = sim_cfg.FPS
    num_frames = dt * FPS

    ### simulate dt seconds at 30Hz to the nearest fixed timestep ###
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    poses = []
    start_time = sim.get_world_time()
    cnt = 0
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / FPS)

        ### simulate agent motion ###
        next_state = agent_motion_simulation(sim.agents[0], agent_cfg, cnt, num_frames)
        sim.agents[0].set_state(next_state)

        ### get frames ###
        obs = sim.get_sensor_observations()
        observations.append(obs)

        ### get poses ###
        pose = {}
        for sensor in sim._sensors.keys():
            pose_6dof = sim._sensors[sensor]._agent.get_state().sensor_states[sensor]
            pose_mat = SixDOFPose2Mat(pose_6dof)
            pose.update({
                sensor: pose_mat
            })
        poses.append(pose)
        
        cnt += 1

    return observations, poses


def get_pinhole_intrinsic(sim: habitat_sim.Simulator) -> np.ndarray:
    """ get Pinhole camera intrinsics

    Args:
        sim (habitat_sim.Simulator)

    Returns:
        K (np.ndarray, [3,3]): pinhole camera intrinsics        

    """
    pinhole_sensor = [i for i in sim.agents[0]._sensors.keys()][0]
    K_gl = sim.agents[0]._sensors[pinhole_sensor].render_camera.projection_matrix
    h, w = sim._Simulator__sensors[0][pinhole_sensor]._spec.resolution
    K = np.array([
        [w*K_gl[0,0]/2, 0, (w-1)/2],
        [0, h*K_gl[1,1]/2, (h-1)/2],
        [0, 0, 1]
        ]
    )
    return K


def save_sim_video(
        observations: List[Dict],
        obs_name    : str,
        output_dir  : str,
        fps         : int
        ) -> None:
    """ save simulation video

    Args:
        observations (List[Dict]): observations
        obs_name (str)           : observation name
        output_dir (str)         : output directory
        fps (int)                : camera FPS
        
    """
    obs_type = obs_name.split("_")[1]
    vut.make_video(
        observations,
        obs_name,
        obs_type,
        os.path.join(output_dir, 'videos', obs_name),
        open_vid=False,
        fps=fps,
    )


def save_color_frames(
        observations: List[Dict],
        obs_name    : str,
        output_dir  : str,
        frame_suffix: str = '.png',
    ) -> None:
    """save simulation color frames

    Args:
        observations (List[Dict]): observations
        obs_name (str)           : observation name
        output_dir (str)         : output directory
        frame_suffix (str)       : frame suffix [.png, .jpg]
        
    """
    for cnt, obs in tqdm(enumerate(observations), total=len(observations), desc='Saving [{}]: '.format(obs_name)):
        img_path = os.path.join(output_dir, obs_name, '{:06}{}'.format(cnt, frame_suffix))
        img = cv2.cvtColor(obs[obs_name], cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img)


def save_depth_frames(
        observations: List[Dict],
        obs_name    : str,
        output_dir  : str,
        depth_png_scale: float = 6553.5,
    ) -> None:
    """save simulation depth frames

    Args:
        observations (List[Dict]): observations
        obs_name (str)           : observation name
        output_dir (str)         : output directory
        depth_png_scale (float)  : depth png scaling factor
        
    """
    for cnt, obs in tqdm(enumerate(observations), total=len(observations), desc='Saving [{}]: '.format(obs_name)):
        depth = obs[obs_name]
        img_path = os.path.join(output_dir, obs_name, '{:06}.png'.format(cnt))
        depth = np.clip((depth * depth_png_scale), 0, 65535).astype(np.uint16)
        cv2.imwrite(img_path, depth)


def save_sensor_poses(
        poses     : List[Dict],
        sensor    : str,
        output_dir: str,
    ) -> None:
    """save simulation depth frames

    Args:
        poses (List[Dict]): sensor poses (sensor-to-world)
        sensor (str)      : sensor name
        output_dir (str)  : output directory
        
    """
    for cnt, pose in tqdm(enumerate(poses), total=len(poses), desc='Saving [{}] poses: '.format(sensor)):
        txt_path = os.path.join(output_dir, sensor+"_pose", '{:06}.txt'.format(cnt))
        np.savetxt(txt_path, pose[sensor])


def save_observations(
        observations: List[Dict],
        poses       : List[Dict],
        output_dir  : str,
        fps         : int,
        out_cfg     : mmengine.ConfigDict,
        K_pinhole   : np.ndarray = None,
        ) -> None:
    """

    Args:
        observations (List[Dict])    : observations
        poses (List[Dict])           : sensor poses (sensor-to-world)
        output_dir (str)             : output directory
        fps (int)                    : camera FPS
        out_cfg (mmengine.ConfigDict): output configuration
        K_pinhole (np.ndarray, [3,3]): camera intrinsics for pinhole camera
    """
    ### clear old result output ###
    if out_cfg.clear_old_output and os.path.isdir(output_dir):
        if out_cfg.force_clear:
            is_del = True
        else:
            is_del_ans = input("Are you sure to delete [{}]? [Y/N]".format(output_dir))
            is_del = True if is_del_ans == "Y" else False
        if is_del:
            shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
        
    ### save output video ###
    if out_cfg.save_video and len(observations) > 0:
        for obs_name in observations[0].keys():
            os.makedirs(os.path.join(output_dir, 'videos'), exist_ok=True)
            save_sim_video(observations, obs_name, output_dir, fps)
    
    ### save output frames ###
    if out_cfg.save_frame and len(observations) > 0:
        ### save images ###
        for obs_name in observations[0].keys():
            os.makedirs(os.path.join(output_dir, obs_name), exist_ok=True)
            if 'color' in obs_name:
                save_color_frames(observations, obs_name, output_dir, out_cfg.frame_suffix)
            elif 'depth' in obs_name:
                save_depth_frames(observations, obs_name, output_dir, out_cfg.depth_png_scale)
            else:
                raise NotImplementedError
    
    ### save camera intrinsics ###
    if out_cfg.save_K and K_pinhole is not None:
        txt_path = os.path.join(output_dir, "pinhole_intrinsics.txt")
        np.savetxt(txt_path, K_pinhole)
    
    ### save poses ###
    if out_cfg.save_pose and len(poses) > 0:
        for sensor in poses[0].keys():
            os.makedirs(os.path.join(output_dir, sensor+"_pose"), exist_ok=True)
            save_sensor_poses(poses, sensor, output_dir)
