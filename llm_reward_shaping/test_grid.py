from env.grid_world import GridWorld
import random

env = GridWorld()
env.reset()
env.render()

for step in range(10):
    action = random.choice([0, 1, 2, 3])
    pos, reward, done, info = env.step(action)
    print(f"Step {step}: action={action}, pos={pos}, reward={reward}, done={done}, info={info}")
    env.render()
    if done:
        print("Reached goal!")
        break