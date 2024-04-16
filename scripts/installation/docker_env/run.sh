#!/bin/bash

### Arguments (personalized) ###
u=${1:-hyzhan}
docker_name=${2:-naruto}
root_dir=$HOME
g=$(id -gn)
DOCKER_IMAGE=${u}/naruto:1.0

### Run Docker ###
docker run --gpus all --ipc=host \
    --name ${docker_name} \
    --rm \
    -e ROOT_DIR=${root_dir} \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -it $DOCKER_IMAGE /bin/bash \
    