import nav2D_envs
import gym
import random
import numpy as np

env = gym.make('nav2D_envs/Nav2DWorld-v0')
observation, info = env.reset(seed=42, return_info=True)
action = np.zeros(2) #np.random.randint(-1,1,size=(2,))
for _ in range(100):
    env.render(mode="human") #env.render(mode="rgb_array")
    observation, reward, done, info = env.step(action)
    #goal = observation['desired_goal']
    #pos = observation['achieved_goal'][:2]

    direction = observation[:2]
    action = direction/np.linalg.norm(direction)/256*10
    #print(observation)
    if done:
        observation, info = env.reset(return_info=True)
        action = np.zeros(2)

env.close()
