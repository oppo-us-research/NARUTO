#!/bin/bash

echo "[Installation] | start building docker image ..."

### Arguments (personalized) ###
PYTHON_VERSION=3.8
USER_NAME=${1:-hyzhan}
DOCKER_TAG=${USER_NAME}/naruto:1.0

### Docker building ###
echo "Will build docker container $DOCKER_TAG ..."
docker build \
    --file envs/Dockerfile \
    --tag $DOCKER_TAG \
    --force-rm \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    --build-arg USER_NAME=$USER_NAME \
    --build-arg GROUP_NAME=$(id -gn) \
    --build-arg python=${PYTHON_VERSION} \
    .
