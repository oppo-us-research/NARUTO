# <img src="assets/naruto.png" width="30" height="30"> <img src="assets/naruto.png" width="30" height="30"> <img src="assets/naruto.png" width="30" height="30">  NARUTO <img src="assets/naruto.png" width="30" height="30"> <img src="assets/naruto.png" width="30" height="30"> <img src="assets/naruto.png" width="30" height="30"> 

# Neural Active Reconstruction from Uncertain Target Observations (CVPR-2024)

<a href='https://oppo-us-research.github.io/NARUTO-website/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/pdf/2402.18771.pdf'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<!-- [![YouTube](https://badges.aleen42.com/src/youtube.svg)](TODO) -->
<!-- <a href='TODO'><img src='assets/poster_badge.png' width=78 height=21></a> -->


This is an official implementation of the paper "NARUTO: Neural Active Reconstruction from Uncertain Target Observations". 

[__*Ziyue Feng*__](https://ziyue.cool/)<sup>\*&dagger;1,2</sup>, 
[__*Huangying Zhan*__](https://huangying-zhan.github.io/)<sup>\*&dagger;&ddagger;1</sup>, 
[Zheng Chen](https://scholar.google.com/citations?user=X6MkScIAAAAJ&hl=en)<sup>&dagger;1,3</sup>, 
[Qingan Yan](https://yanqingan.github.io/)<sup>1</sup>, 
<br>
[Xiangyu Xu](https://scholar.google.com/citations?user=U7TNIYYAAAAJ&hl=en)<sup>1</sup>, 
[Changjiang Cai](https://www.changjiangcai.com/)<sup>1</sup>, 
[Bing Li](https://www.clemson.edu/cecas/departments/automotive-engineering/people/li.html)<sup>2</sup>, 
[Qilun Zhu](https://www.clemson.edu/cecas/departments/automotive-engineering/people/qzhu.html)<sup>2</sup>, 
[Yi Xu](https://www.linkedin.com/in/yi-xu-42654823/)<sup>1</sup> 

<sup>1</sup> OPPO US Research Center, 
<sup>2</sup> Clemson University 
<sup>3</sup> Indiana University 

<sup>*</sup> Co-first authors with equal contribution </br>
<sup>&dagger;</sup> Work done as an intern at OPPO US Research Center </br>
<sup>&ddagger;</sup> Corresponding author </br>


## Table of Contents
- [📁 &nbsp; Repo Structure](#repo-structure)
- [🛠️ &nbsp; Installation](#installation) 
- [💿 &nbsp; Dataset Preparation](#dataset-preparation)
- [🏃‍♂️ &nbsp;Running NARUTO](#running_naruto)
- [🔎 &nbsp; Evaluation](#evaluation)
- [🎨 &nbsp; Run on Customized Scenes](#run-on-customized-scenes)
- [📜 &nbsp; License](#license)
- [🤝 &nbsp; Acknowledgement](#acknowledgement)
- [📖 &nbsp; Citation](#citation)

<h2 id="repo-structure"> 📁 Repo Structure  </h2>

```
# Main directory
├── NARUTO (ROOT)
│   ├── assets                                      # README assets
│   ├── configs                                     # experiment configs
│   ├── data                                        # dataset
│   │   └── MP3D                                    # Matterport3D for Habitat data
│   │   └── mp3d_data                               # Matterport3D raw Dataset
│   │   └── Replica                                 # Replica SLAM Dataset
│   │   └── replica_v1                              # Replica Dataset v1
│   ├── envs                                        # environment installation 
│   ├── results                                     # experiment results
│   ├── scripts                                     # scripts
│   │   └── data                                    # data related scripts
│   │   └── evaluation                              # evaluation related scripts
│   │   └── installation                            # installation related scripts
│   │   └── naruto                                  # running NARUTO scripts
│   ├── src                                         # source code
│   │   └── data                                    # data code
│   │   └── evaluation                              # evaluation code
│   │   └── layers                                  # pytorch layers
│   │   └── naruto                                  # NARUTO framework code
│   │   └── planning                                # planning code
│   │   └── simulator                               # simulator code
│   │   └── slam                                    # SLAM code
│   │   └── utils                                   # utility code
│   │   └── visualization                           # visualization code
│   ├── third_parties                               # third_parties
│   │   └── coslam                                  # CoSLAM 
│   │   └── habitat_sim                             # habitat-sim
│   │   └── neural_slam_eval                        # evaluation tool


# Data structure
├── data                                            # dataset dir
│   ├── MP3D                                        # Matterport3D data
│   │   └── v1
│   │       └── tasks
│   │           └── mp3d_habitat
│   │               ├── 1LXtFkjw3qL
│   │               └── ...
│   ├── replica_v1                                  # Replica-Dataset
│   │   └── apartment_0
│   │       └── habitat
│   │           └── replicaSDK_stage.stage_config.json
│   │   └── ...
│   ├── Replica                                     # Replica SLAM Dataset
│   │   └── office_0
│   │   └── ...

# Configuration structure
├── configs                                         # configuration dir
│   ├── default.py                                  # Default overall configuration
│   ├── MP3D                                        # Matterport3D 
│   │   └── mp3d_coslam.yaml                        # CoSLAM default configuration for MP3D
│   │   └── {SCENE}
│   │       └── {EXP}.py                            # experiment-specific overall configuraiton, inherit from default.py
│   │       └── coslam.yaml                         # scene-specific CoSLAM configuration, inherit from mp3d_coslam.yaml
│   │       └── habitat.py                          # scene-specific HabitatSim configuration
│   │   └── ...
│   ├── Replica                                     # Replica
│   │   └── ...

NOTE: default.py/{EXP}.py has the highest priority that can override configurations in other config files (e.g. mp3d_coslam.yaml/coslam.yaml, habitat.py)

# Result structure
├── results                                         # result dir
│   ├── MP3D                                        # Matterport3D result
│   │   └── GdvgFV5R1Z5
│   │       └── {EXPERIMENT_SETUP}
│   │           └── run_{COUNT}
│   │               └── eval_result.txt
│   │               └── coslam
│   │                   └── checkpoint
│   │                   └── mesh
│   │   └── ...
│   ├── Replica                                     # Replica result
│   │   └── office_0
│   │       └── {EXPERIMENT_SETUP}
│   │           └── run_{COUNT}
│   │               └── eval_result.txt
│   │               └── coslam
│   │                   └── checkpoint
│   │                   └── mesh
│   │   └── ...
```

<h2 id="installation"> 🛠️ Installation </h2>

### Install NARUTO

```
# Clone the repo with the required third parties.
git clone --recursive https://github.com/oppo-us-research/NARUTO.git

# We assume ROOT as the project directory.
cd NARUTO
ROOT=${PWD}
```

In this repo, we provide two types of environement installations: Docker and Anaconda.

Users can optionally install one of them. The installation process includes: 

- installation of [Habitat-Sim](https://github.com/facebookresearch/habitat-sim), where we install our updated [Habitat-Sim](https://github.com/Huangying-Zhan/habitat-sim), where the geometry compilation is updated.  

- installation of [Co-SLAM](https://github.com/HengyiWang/Co-SLAM), which is used as our mapping backbone.

- installation of other packages required to run NARUTO.

### <img src="assets/docker_logo.png" width="20" height="20"> [Optional 1] Docker Environment

Follow the steps to install the Docker environment: 
```
# Build Docker image
bash scripts/installation/docker_env/build.sh

# Run Docker container
bash scripts/installation/docker_env/run.sh

# Activate conda env in Docker Env
source activate naruto

# Install tinycudann as required in Co-SLAM
# We try to include this installation while building Docker but it fails somehow. 
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### <img src="assets/anaconda_logo.png" width="15" height="15"> [Optional 2] Conda Environment

Follow the steps to install the conda environment

```
# Build conda environment
bash scripts/installation/conda_env/build.sh

# Activate conda env
source activate naruto
```

<h2 id="dataset-preparation"> 💿 Dataset Preparation   </h2>

### <img src="assets/meta_logo.png" width="25" height="15"> Replica Data

Follow the steps to download [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset/tree/main).
```
# Download Replica data and save as data/replica_v1.
# This process can take a while.
bash scripts/data/replica_download.sh data/replica_v1

# Once the donwload is completed, create modified Habitat Simulator configs that adjust the coordinate system direction.
# P.S. we adjust the config so the coordinates matches with the mesh coordinates.
bash scripts/data/replica_update.sh data/replica_v1
```

**[Optional]** We also use Replica Data (for SLAM) for some tasks, e.g. passive mapping / initialize the starting position.

```
# Download Replica (SLAM) Data and save as data/Replica
bash scripts/data/replica_slam_download.sh
```

### <img src="assets/matterport_logo.png" width="15" height="15"> Matterport3D


To download Matterport3D dataset, please refer to the instruction in [Matterport3D](https://niessner.github.io/Matterport/).

The download script is not included here as there is a __Term of Use agreement__ for using Matterport3D data. 

However, our method **does not** require the full Matterport3D dataset. 
Users can download the data related to the task **habitat** only.

```
# Example use of the Matterport3D download script:
python download_mp.py -o data/MP3D --task_data habitat

# Unzip data
cd data/MP3D/v1/
unzip mp3d_habitat.zip
rm mp3d_habitat.zip
cd ${ROOT}
```

<h2 id="running_naruto"> <img src="assets/running_naruto.gif" width="30" height="30"> Running NARUTO </h2>

Here we provide the script to run the full NARUTO system described in the paper. 
This script also includes the upcoming [evaluation process](#evaluation).
We also provide a [flowchart](assets/naruto_planner_flowchart.png) to assist users to better understand the logic flow of the NARUTO planner. 

```
# Run Replica 
bash scripts/naruto/run_replica.sh {SceneName/all} {NUM_TRIAL} {EXP_NAME} {ENABLE_VIS}

# Run MP3D 
bash scripts/naruto/run_mp3d.sh {SceneName/all} {NUM_TRIAL} {EXP_NAME} {ENABLE_VIS}

# examples
bash scripts/naruto/run_replica.sh office0 1 NARUTO 1
bash scripts/naruto/run_mp3d.sh gZ6f7yhEvPG 1 NARUTO 0
bash scripts/naruto/run_replica.sh all 5 NARUTO 0
```


<h2 id="evaluation"> 🔎 Evaluation  </h2>

We evaluate the reconstruction using the following metrics with a threshold of 5cm: 

- Accuracy (cm)
- Completion (cm)
- Completion ratio (%) 

We also compute the mean absolute distance, MAD (cm), between the estimated SDF distance on all vertices from the ground truth mesh. 

In line with methodologies employed in previous studies [65, 66], we refine the predicted mesh by removing unobserved regions and noisy points that are within the camera frustum but external to the
target scene, utilizing a mesh culling technique. Refer to [65] for a detailed explanation of the mesh culling process

### Quantitative Evaluation

```
# Evaluate Replica result
bash scripts/evaluation/eval_replica.sh {SceneName/all} {TrialIndex} {IterationToBeEval}
bash scripts/evaluation/eval_mp3d.sh {SceneName/all} {TrialIndex} {IterationToBeEval}

# Examples
bash scripts/evaluation/eval_replica.sh office0 0 2000
bash scripts/evaluation/eval_mp3d.sh gZ6f7yhEvPG 0 5000
```

### Qualitative Evaluation

```
# Draw trajectory in the scene mesh
bash scripts/evaluation/visualize_traj.sh {MESH_FILE} {TRAJ_DIR} {CAM_VIEW} {OUT_DIR}

# examples
bash scripts/evaluation/visualize_traj.sh \
results/Replica/office0/NARUTO/run_0/coslam/mesh/mesh_2000_final_cull_occlusion.ply \
results/Replica/office0/NARUTO/run_0/visualization/pose \
src/visualization/default_camera_view.json \
results/Replica/office0/NARUTO/run_0/visualization/traj_vis_view_1
```

<h2 id="run-on-customized-scenes"> 🎨 Run on Customized Scenes  </h2>

```
1. Prepare 3D model (e.g. ply/glb files)
2. Use Blender to adjust the coordinate system. (currently we are using RDF in Blender, when it is properly adjusted. We should be looking at the model, and the model is standing)
3. Delete all other unnecessary objects in the `Scene Collection`. 
4. Scale and Translate the object to a proper location and size.
5. Add a cube (shift+A) as the bounding box. Scaling cube with negative scales!
6. Export the model as glb
7. Create configuration files as listed in `configs/NARUTO/hokage_room`.
8. RUNNING NARUTO!
```

<h2 id="license"> 📜 License  </h2>

NARUTO is licensed under [MIT licence](LICENSE). For the third parties, please refer to their license. 

- [CoSLAM](https://github.com/HengyiWang/Co-SLAM/blob/main/LICENSE): Apache-2.0 License
- [HabitatSim](https://github.com/facebookresearch/habitat-sim/blob/main/LICENSE): MIT License
- [neural-slam-eval](https://github.com/HengyiWang/Co-SLAM/blob/main/LICENSE): Apache-2.0 License


<h2 id="acknowledgement"> 🤝 Acknowledgement  </h2>

We sincerely thank the owners of the following open source projects, which are used by our released codes: [CoSLAM](https://github.com/HengyiWang/Co-SLAM), [HabitatSim](https://github.com/facebookresearch/habitat-sim).



<h2 id="citation"> 📖 Citation  </h2>

```
@article{feng2024naruto,
  title={NARUTO: Neural Active Reconstruction from Uncertain Target Observations},
  author={Feng, Ziyue and Zhan, Huangying and Chen, Zheng and Yan, Qingan and Xu, Xiangyu and Cai, Changjiang and Li, Bing and Zhu, Qilun and Xu, Yi},
  journal={arXiv preprint arXiv:2402.18771},
  year={2024}
}
```
