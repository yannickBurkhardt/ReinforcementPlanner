import nav2D_envs
import gym
import random
import numpy as np
from configs import config
from math import atan2, pi
CONFIG_FILE = "/usr/src/app/rl/configs/sac.yaml"
config.merge_from_file(CONFIG_FILE)

env = gym.make('nav2D_envs/Nav2DWorld-v0')
observation, info = env.reset(seed=42, return_info=True)
action = np.zeros(2) #np.random.randint(-1,1,size=(2,))
rewards = 0.0

for _ in range(1000):
    env.render(mode="human") #env.render(mode="rgb_array")
    observation, reward, done, info = env.step(action)
    #goal = observation['desired_goal']
    #pos = observation['achieved_goal'][:2]
    rewards += reward
    if config.ENV.MOTION_MODEL == 'holonomic':
        direction = observation[2:4]

        action = direction/np.linalg.norm(direction)*env.max_vel
    elif config.ENV.MOTION_MODEL == 'differential':
        direction = observation[3:5]

        #pure pursuit
        omega = env.max_vel*(atan2(direction[1], direction[0]) - observation[2]*pi)/pi
        vel = np.linalg.norm(direction)*env.max_vel
        # vel= 0.0
        # omega=1.0
        action = np.array([vel, omega])
        print("\npsi: ", atan2(direction[1], direction[0]))
        print("theta: ", observation[2]*pi)
        print(omega)
        print("direction: ", direction)

    if done:
        print("\rreward: {}".format(rewards))
        rewards=0.0
        observation, info = env.reset(return_info=True)
        print(observation.shape)
        print("Done!!!")

env.close()
