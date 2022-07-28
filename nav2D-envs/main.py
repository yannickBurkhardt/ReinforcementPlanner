import nav2D_envs
import gym
import random
import numpy as np
from configs import config

CONFIG_FILE = "/usr/src/app/rl/configs/sac.yaml"
config.merge_from_file(CONFIG_FILE)

env = gym.make('nav2D_envs/Nav2DWorld-v0')
observation, info = env.reset(seed=42, return_info=True)
action = np.zeros(2) #np.random.randint(-1,1,size=(2,))
rewards = 0.0

for _ in range(100):
    env.render(mode="human") #env.render(mode="rgb_array")
    observation, reward, done, info = env.step(action)
    #goal = observation['desired_goal']
    #pos = observation['achieved_goal'][:2]
    rewards += reward
    direction = observation[2:4]
    action = direction/np.linalg.norm(direction)*env.max_vel
    if done:
        print("\rreward: {}".format(rewards))
        rewards=0.0
        observation, info = env.reset(return_info=True)
        print(observation.shape)
        print("Done!!!")

env.close()
