from readline import parse_and_bind
import gym
import numpy as np
import nav2D_envs
import sys
import shutil
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from configs import config
from utils import create_new_experiment_folder, save_configs

def main():
    # Create a new experiments folder to save checkpoints and logs
    experiment_dir = create_new_experiment_folder(config.TRAIN.OUT_DIR)
    print('Saving Experiment results to:', experiment_dir)

    # Copy  configuration, environment and training files to experiment directory
    save_configs(config, experiment_dir)

    # Noise
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=config.MODEL.OUNOISE_SIGMA * np.ones(2))
    # Define Environment and Model
    env = gym.make('nav2D_envs/Nav2DWorld-v0')
    model = SAC("MlpPolicy", env, train_freq=8, tensorboard_log=experiment_dir,verbose=1, action_noise=action_noise)
    print("\n Model:", model.policy)
    # Train and Save Model
    model.learn(total_timesteps=config.TRAIN.TOTAL_STEPS, log_interval=10)
    model.save(experiment_dir+"/sac_nav")

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description="Reinforcement Learning - based path planning")
    parser.add_argument("--config-file", type=str, default="configs/sac.yaml", required=True)
    parser.add_argument("--output-path", type=str, default="experiments", required=False)

    args = parser.parse_args()
    config.TRAIN.OUT_DIR = args.output_path
    print("\n Training SAC with configuration parameters: \n",config)
    config.merge_from_file(args.config_file)

    main()