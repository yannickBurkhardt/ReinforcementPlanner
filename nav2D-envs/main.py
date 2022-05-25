import nav2D_envs
import gym
import random
import numpy as np
env = gym.make('nav2D_envs/Nav2DWorld-v0')
observation, info = env.reset(seed=42, return_info=True)
action = np.random.randint(-1,1,size=(2,))
for _ in range(100):
    env.render()
    observation, reward, done, info = env.step(action)
    pos = observation[:2] #['target']
    goal = observation[2:4] #['agent']
    direction = goal-pos
    action = np.int16(10.0/np.linalg.norm(direction) * direction)
    print(action)
    if done:
        observation, info = env.reset(return_info=True)
        print("Done!!!")

env.close()
