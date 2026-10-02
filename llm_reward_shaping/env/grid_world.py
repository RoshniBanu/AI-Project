import numpy as np
import gymnasium as gym
from gymnasium import spaces

def default_reward_fn(collided, terminated, step_count):
    if terminated:
        return 1.0
    elif collided:
        return -1.0
    else:
        return 0.0

class GridWorldEnv(gym.Env):
    def __init__(self, grid_size=5, obstacles=None, goal=None, start=None,
                 max_steps=50, reward_fn=None):
        super().__init__()
        self.grid_size = grid_size
        self.obstacles = obstacles if obstacles is not None else [(1, 1), (2, 3)]
        self.goal = goal if goal is not None else (4, 4)
        self.start = start if start is not None else (0, 0)
        self.max_steps = max_steps
        self.reward_fn = reward_fn if reward_fn is not None else default_reward_fn

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=grid_size - 1, shape=(2,), dtype=np.int32
        )

        self.agent_pos = self.start
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        while True:
            rand_start = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size),
            )
            if rand_start not in self.obstacles and rand_start != self.goal:
                break

        self.agent_pos = rand_start
        self.current_step = 0
        obs = np.array(self.agent_pos, dtype=np.int32)
        info = {}
        return obs, info

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:
            x = max(0, x - 1)
        elif action == 1:
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:
            y = max(0, y - 1)
        elif action == 3:
            y = min(self.grid_size - 1, y + 1)

        new_pos = (x, y)
        collided = new_pos in self.obstacles

        if not collided:
            self.agent_pos = new_pos

        self.current_step += 1
        terminated = self.agent_pos == self.goal
        truncated = self.current_step >= self.max_steps

        reward = self.reward_fn(collided=collided, terminated=terminated,
                                 step_count=self.current_step)

        obs = np.array(self.agent_pos, dtype=np.int32)
        info = {"collided": collided}
        return obs, reward, terminated, truncated, info

    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for (ox, oy) in self.obstacles:
            grid[ox][oy] = "X"
        gx, gy = self.goal
        grid[gx][gy] = "G"
        ax, ay = self.agent_pos
        grid[ax][ay] = "A"
        for row in grid:
            print(" ".join(row))
        print()