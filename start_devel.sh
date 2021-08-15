#!/bin/bash
xhost + && \
docker build -t deepstream-monai:ds6.0-dev . -f Dockerfile.devel && \
docker run --gpus all \
           --rm \
           -it \
           --shm-size=1g --ulimit memlock=-1 \
           -v ${PWD}:/app \
           -w /app \
           -e DISPLAY \
           deepstream-monai:ds6.0-dev
