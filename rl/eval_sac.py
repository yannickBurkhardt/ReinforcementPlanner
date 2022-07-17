import gym
import numpy as np
import nav2D_envs
import glob
import os
import sys
import shutil
from datetime import datetime

from stable_baselines3 import SAC


# Path to model
model_path = 'experiments/06_2022-06-2812_30_53.220324/sac_nav'

# Define Environment and Model
env = gym.make('nav2D_envs/Nav2DWorld-v0')
model = SAC.load(model_path)

# Make sure zero noise
model.action_noise = None

obs = env.reset()
max_steps = 100
while True:
  for i in range(max_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(obs.shape)
    env.render()
    if done:
      break
  obs = env.reset()

