from gym.envs.registration import register

register(
    id="nav2D_envs/Nav2DWorld-v0",
    entry_point="nav2D_envs.envs:Nav2DWorldEnv",
)
