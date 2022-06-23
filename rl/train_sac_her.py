import gym
import numpy as np
import nav2D_envs
import glob
import os
import sys
import shutil
from datetime import datetime

from stable_baselines3 import SAC,HerReplayBuffer
# from stable_baselines3.common.envs import BitFlippingEnv


# Get Experiment path where all files will be saved
TRAIN_OUT_DIR = 'experiments/'
files = glob.glob(TRAIN_OUT_DIR + "*")
max_file = max(files)
now = str(datetime.now()) # current date and time
experiment_dir = TRAIN_OUT_DIR + "{:02d}_{}_HER".format(int(max_file.split('/')[-1].split('_')[0])+1, now.replace(' ','').replace(':','_'))
os.mkdir(experiment_dir)
print('Saving Experiment results to:', experiment_dir)

# Copy environment and training files to experiment directory
shutil.copyfile('../nav2D-envs/nav2D_envs/envs/nav2D_world.py', experiment_dir+"/nav2D_world.py")
shutil.copyfile('./train_sac_her.py', experiment_dir+"/train_sac_her.py")

# Define HER
goal_selection_strategy = 'future'
online_sampling = True
max_episode_length = 300
replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=max_episode_length,
    )

# Define Environment and Model
env = gym.make('nav2D_envs/Nav2DWorld-v0', dict_obs_space=True)
model = SAC("MultiInputPolicy", env, train_freq=8, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs = replay_buffer_kwargs, tensorboard_log=experiment_dir,verbose=1)

# Train and Save
model.learn(total_timesteps=2000000, log_interval=10)
model.save(experiment_dir+"/her_sac_nav")

del model # remove to demonstrate saving and loading

model = SAC.load(experiment_dir+"/her_sac_nav", env=env)

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()