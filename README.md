# tum-adlr-ss22-07

## Install
1. Install Docker following the instructions on the [link](https://docs.docker.com/engine/install/ubuntu/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (for gpu support).

3. Clone this repo

4. Build Docker Container
    ```bash
    docker build . -t reinforcement_planning
    ```

## Run (Needs nvidia-docker and the right Nvidia GPU drivers)
    ```bash
    source run_docker.sh 
    ```