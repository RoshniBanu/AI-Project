from env.grid_world import GridWorldEnv
from gymnasium.utils.env_checker import check_env

env = GridWorldEnv()
check_env(env, warn=True)
print("Environment passed Gymnasium checks!")