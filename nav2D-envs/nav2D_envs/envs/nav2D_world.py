import gym
from gym import spaces
import pygame
import numpy as np


class Nav2DWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=512):
        # Optional settings
        self.difficult_scenario = False  # Sample obstacles between agent and goal
        self.variance_observation_noise = 0.01  # Set variance of Gaussian noise on observations (0 is no noise)
        self.variance_action_noise = 0.01  # Set variance of Gaussian noise on actions (0 is no noise)

        self.size = size  # The size of the square grid
        self.window_size = size  # The size of the PyGame window

        self.num_obstacles = 10  # Number of obstacles in the environment
        self.num_obs_considered = 5  # Number of the closest obstacles considered in observations
        self.max_vel = 10 / self.size  # Maximal velocity of agent
        self.agent_size = 20 / self.size  # Size of agent
        self.obs_min_size = 5 / self.size  # Minimal size of obstacles
        self.obs_max_size = 40 / self.size  # Maximal size of obstacles
        self.obstacles_size = np.zeros(self.num_obstacles)  # Size of obstacles
        self.obstacles_positions = np.zeros([self.num_obstacles, 2])  # Positions of obstacles
        self.target_location = np.array([0, 0])  # Position of the goal
        self.agent_location = np.array([0, 0])  # Position of the agent
        self.memory = 10  # Number of previous actions fed back into observations
        self.previous_actions = np.zeros((self.memory, 2))  # Array for previous actions
        self.num_steps = 0  # Number of steps executed in current episode
        self.path = []  # List to store the path of the agent

        # Lower and upper bound for obstacle positions
        low_obs_p = np.array([-2.0] * 2 * self.num_obs_considered)
        high_obs_p = np.array([2.0] * 2 * self.num_obs_considered)

        # Lower and upper bound for observation space
        # (Agent position, relative goal position, relative obstacle positions, previous actions)
        low = np.hstack((np.array([-1.0, -1.0]), np.array([-2.0, -2.0]), low_obs_p,
                         np.array([-self.max_vel] * 2 * self.memory)))
        high = np.hstack((np.array([1.0, 1.0]), np.array([2.0, 2.0]), high_obs_p,
                          np.array([-self.max_vel] * 2 * self.memory)))

        # Initialize observation space
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Initialize action space (velocity in x- and y-direction)
        self.action_space = spaces.Box(np.array([-self.max_vel] * 2), np.array([self.max_vel] * 2), dtype=np.float32)

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):

        # Calculate relative locations of target/ obstacles
        dir_goal = self.target_location - self.agent_location
        dir_obs = self.obstacles_positions - self.agent_location

        # Find the closest distance between agent and each obstacle
        dir_obs -= dir_obs / np.linalg.norm(dir_obs, axis=1)[:, np.newaxis] * \
                   (self.agent_size + self.obstacles_size)[:, np.newaxis]

        # Sort obstacle distances by norm
        sortidxs = np.argsort(np.linalg.norm(dir_obs, axis=-1))
        dir_obs_sorted = dir_obs[sortidxs, np.arange(dir_obs.shape[1])[:, None]]
        dir_obs_closest = dir_obs_sorted.T[:self.num_obs_considered]

        # Update observations (Agent position, relative goal position, relative obstacle positions, previous actions)
        observation = np.hstack((self.agent_location, dir_goal, dir_obs_closest.flatten(),
                                 self.previous_actions.flatten()))

        # Add Gaussian noise to agent position, goal position and obstacle positions
        num_noisy_elements = (4 + 2 * self.num_obs_considered)
        noise = np.zeros(observation.size)
        noise[:num_noisy_elements] = self.np_random.normal(scale=self.variance_observation_noise,
                                                           size=num_noisy_elements)
        observation += noise
        return observation

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self.agent_location - self.target_location, ord=1
            )
        }

    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random (not too close to environment boundaries)
        self.agent_location = self.np_random.uniform(-0.9, 0.9, size=2)

        # Reset path, previous actions and number of steps of episode
        self.path = [self.agent_location]
        self.previous_actions = np.zeros((self.memory, 2))
        self.num_steps = 0

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self.target_location = self.agent_location
        while np.linalg.norm(self.target_location - self.agent_location) < 2.0 * self.agent_size:
            self.target_location = self.np_random.uniform(-1.0, 1.0, size=2)

        # Sample the obstacles' radius randomly
        self.obstacles_size = self.np_random.uniform(self.obs_min_size, self.obs_max_size, size=self.num_obstacles)

        # Sample the obstacles' locations randomly until it does not coincide with the agent's location or the target
        # location
        obstacles = []
        for i in range(self.num_obstacles):
            obs_location = self.agent_location

            # Calculate line between agent and target
            target_dir = self.target_location - self.agent_location
            while ((np.linalg.norm(obs_location - self.agent_location) < 2 * (self.agent_size + self.obstacles_size[i]))
                   or (np.linalg.norm(obs_location - self.target_location) <
                       2 * (self.agent_size + self.obstacles_size[i]))):

                if self.difficult_scenario:
                    # Gaussian sampling of obstacles along line between agent and goal
                    obs_location = self.agent_location + self.np_random.uniform(0.0, 1.0, size=1) * target_dir
                    obs_location = obs_location + self.np_random.normal(scale=0.3, size=2)
                    obs_location = np.clip(obs_location, -1.0, 1.0)
                else:
                    # Uniform sampling of obstacles
                    obs_location = self.np_random.uniform(-1.0, 1.0, size=2)
            obstacles.append(obs_location)
        self.obstacles_positions = np.array(obstacles)

        # Update observation and info
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):
        # Apply noise to action
        action += self.np_random.normal(scale=self.variance_action_noise, size=action.size)

        # Update list with previous actions
        self.previous_actions[:-1] = self.previous_actions[1:]
        self.previous_actions[-1] = action

        # Update the agent's position
        self.agent_location += action

        # Append new location to agent's path
        self.path.append(self.agent_location.copy())

        # An episode is done iff the agent has reached the target or crashed into an obstacle
        arrived_goal = np.linalg.norm(self.agent_location - self.target_location) < 2 * self.agent_size

        # Check for collisions
        collision = False
        if (any(np.linalg.norm(self.agent_location - self.obstacles_positions, axis=1) < 2 * (
                self.agent_size + self.obstacles_size))):
            collision = True

        # Check if agent is out of boundaries
        if (self.agent_location[0] < -1.0 + self.agent_size or self.agent_location[0] > 1.0 - self.agent_size or
                self.agent_location[1] < -1.0 - self.agent_size or self.agent_location[1] > 1.0 - self.agent_size):
            collision = True

        # Check if max. number of steps is reached
        self.num_steps += 1
        max_steps_reached = False
        if self.num_steps >= 500:
            max_steps_reached = True

        # Sparse reward
        if arrived_goal:
            reward = 1
        else:
            reward = 0

        # Get observation, info and if episode is over
        observation = self._get_obs()
        info = self._get_info()
        done = arrived_goal or collision or max_steps_reached

        return observation, reward, done, info

    def render(self, mode="human"):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.agent_size * self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                (self.target_location + 1.0) / 2 * self.size - np.array([pix_square_size, pix_square_size]),
                (pix_square_size * 2, pix_square_size * 2),
            ),
        )
        # Now we draw the agent
        for pos in self.path:
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                ((pos + 1.0) / 2 * self.size),
                pix_square_size,
            )

        # Finally, add obstacles
        for i, obs_pos in enumerate(self.obstacles_positions):
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                ((obs_pos + 1.0) / 2 * self.size),
                self.obstacles_size[i] * self.size,
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, 50 * x),
                (self.window_size, 50 * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (50 * x, 0),
                (50 * x, self.window_size),
                width=3,
            )

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
