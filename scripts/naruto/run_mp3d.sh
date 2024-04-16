#!/bin/bash
##################################################
### This script is to run the full NARUTO system 
### (active planning and active ray sampling) 
###  on the MP3D dataset.
##################################################

# Input arguments
scene=${1:-gZ6f7yhEvPG}
num_run=${2:-1}
EXP=${3:-NARUTO} # config in configs/{DATASET}/{scene}/{EXP}.py will be loaded
ENABLE_VIS=${4:-0}

export CUDA_VISIBLE_DEVICES=0
PROJ_DIR=${PWD}
DATASET=MP3D
RESULT_DIR=${PROJ_DIR}/results

##################################################
### Random Seed
###     also used to initialize agent pose 
###     from indexing a set of predefined pose/traj 
##################################################
seeds=(0 1224 4869 8964 1000)
seeds=("${seeds[@]:0:$num_run}")

##################################################
### Scenes
###     choose one or all of the scenes
##################################################
scenes=( GdvgFV5R1Z5 gZ6f7yhEvPG HxpKQynjfin pLe4wQe7qrG YmJkqBEsHnH )
# Check if the input argument is 'all'
if [ "$scene" == "all" ]; then
    selected_scenes=${scenes[@]} # Copy all scenes
else
    selected_scenes=($scene) # Assign the matching scene
fi

##################################################
### Main
###     Run for selected scenes for N trials
##################################################
for scene in $selected_scenes
do
    for i in "${!seeds[@]}"; do
        seed=${seeds[$i]}

        ### create result folder ###
        result_dir=${RESULT_DIR}/${DATASET}/$scene/${EXP}/run_${i}
        mkdir -p ${result_dir}

        ### run experiment ###
        CFG=configs/${DATASET}/${scene}/${EXP}.py
        python src/naruto/main.py --cfg ${CFG} --seed ${seed} --result_dir ${result_dir} --enable_vis ${ENABLE_VIS}

        ### evaluation ###
        bash scripts/evaluation/eval_mp3d.sh ${scene} ${i} 5000 ${EXP}
    done
done
