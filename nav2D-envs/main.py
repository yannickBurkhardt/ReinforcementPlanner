import nav2D_envs
import gym
import numpy as np

# Set up environment
env = gym.make('nav2D_envs/Nav2DWorld-v0')
observation, info = env.reset(seed=42, return_info=True)
action = np.zeros(2)

# Run test
for _ in range(100):
    env.render(mode="human")
    observation, reward, done, info = env.step(action)

    # Move agent towards goal
    direction = observation[2:4]
    action = direction/np.linalg.norm(direction)/256*10
    print(observation)

    if done:
        # Reset environment
        observation, info = env.reset(return_info=True)
        action = np.zeros(2)

env.close()
