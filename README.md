# tum-adlr-ss22-07

## Install

### Docker
1. Install Docker following the instructions on the [link](https://docs.docker.com/engine/install/ubuntu/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (for gpu support).

3. Clone this repo

4. Build Docker Container
    ```bash
    docker build . -t reinforcement_planning
    ```
### In host machine (with conda)
1. Create conda environment
    ```bash
    conda create -n "reinforcement_planning" python=3.8.10
    ```

2. Activate conda environmnet
    ```bash
    conda activate reinforcement_planning
    ```

3. Install dependencies
    ```bash
    python -m pip install -r requirements.txt
    ```

4. Install pytorch
    ```bash
    pip install --no-cache-dir torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
    ```

5. Install further packages
    ```bash
    python -m pip install -e nav2D-envs/
    python -m pip install -e rlkit/
    ```


## Run (Needs nvidia-docker and the right Nvidia GPU drivers)
```bash
source run_docker.sh 
```




