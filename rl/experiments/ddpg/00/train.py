from readline import parse_and_bind
import gym
import numpy as np
import nav2D_envs
import argparse
from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from configs import config
from utils import create_new_experiment_folder, save_configs

def main():
    # Create a new experiments folder to save checkpoints and logs
    experiment_dir = create_new_experiment_folder(config.TRAIN.OUT_DIR)
    print('Saving Experiment results to:', experiment_dir)

    # Copy  configuration, environment and training files to experiment directory
    save_configs(config, experiment_dir)

    # Define Environment
    env = gym.make('nav2D_envs/Nav2DWorld-v0')
    # Noise
    if config.TRAIN.ALGORITHM != "ppo":
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=config.MODEL.OUNOISE_SIGMA * np.ones(2))
    
    #Define Model and algorithm
    if config.TRAIN.ALGORITHM == "sac":
        model = SAC("MlpPolicy", env, train_freq=8, tensorboard_log=experiment_dir,verbose=1, 
                        action_noise=action_noise, learning_rate=config.TRAIN.LEARNING_RATE)
    elif config.TRAIN.ALGORITHM == "ppo":
        model = PPO("MlpPolicy", env, tensorboard_log=experiment_dir,verbose=1, 
                        learning_rate=config.TRAIN.LEARNING_RATE)
    elif config.TRAIN.ALGORITHM == "ddpg":
        model = DDPG("MlpPolicy", env, train_freq=8, tensorboard_log=experiment_dir,verbose=1, 
                        action_noise=action_noise, learning_rate=config.TRAIN.LEARNING_RATE)
    else:
        raise ValueError("Algorithm {} not supported".format(config.TRAIN.ALGORITHM))
    print("\n Model:", model.policy)
    
    # Train and Save Model
    model.learn(total_timesteps=config.TRAIN.TOTAL_STEPS, log_interval=10)
    model.save(experiment_dir+"/nav_model")

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description="Reinforcement Learning - based path planning")
    parser.add_argument("--config-file", type=str, default="configs/sac.yaml", required=True)
    parser.add_argument("--output-path", type=str, default="experiments", required=False)

    args = parser.parse_args()
    config.merge_from_file(args.config_file)
    config.TRAIN.OUT_DIR = args.output_path + "/{}".format(config.TRAIN.ALGORITHM)

    config.ENV.DEBUG = False # Make sure we don't save any images during training
    print("\n Training {} with configuration parameters: {}\n".format(config.TRAIN.ALGORITHM, config))

    main()