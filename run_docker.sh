#!/bin/bash
sudo xhost +si:localuser:root
XSOCK=/tmp/.X11-unix

docker run -it --rm  \
    --net=host  \
    --privileged \
    --runtime=nvidia \
    -e DISPLAY=$DISPLAY \
    -v $XSOCK:$XSOCK \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v `pwd`/nav2D-envs:/usr/src/app/nav2D-envs \
    -v `pwd`/rlkit:/usr/src/app/rlkit \
    -v `pwd`/rl:/usr/src/app/rl \
    --shm-size 8G \
    reinforcement_planning "$@"

