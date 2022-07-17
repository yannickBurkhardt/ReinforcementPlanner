import glob
import shutil
import os
from yacs.config import CfgNode as Node

def create_new_experiment_folder(folder_name: str) -> str:
    """
    Create a new experiment file name.
    """
    files = glob.glob(folder_name + "/*")
    if len(files)>0:
        max_file = max(files)
    else:
        max_file = 0
    experiment_dir = folder_name + "/{:02d}".format(int(max_file.split('/')[-1].split('_')[0])+1)
    os.mkdir(experiment_dir)
    return experiment_dir

def save_configs(config: Node, save_path: str) -> None:
    """
    Save configuration file to experiment directory.
    """
    with open(save_path + "/config.yaml", "w") as f:
        f.write(config.dump())
    shutil.copyfile("train_sac.py", save_path + "/train_sac.py")
    shutil.copyfile("../nav2D-envs/nav2D_envs/envs/nav2D_world.py", save_path + "/nav2D_world.py")