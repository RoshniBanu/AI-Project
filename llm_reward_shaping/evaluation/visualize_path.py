import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_agent_path(model, env, max_steps=50):
    """Runs one deterministic episode and returns the sequence of positions."""
    obs, info = env.reset()
    path = [tuple(obs)]
    done = False
    truncated = False
    steps = 0

    while not (done or truncated) and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        path.append(tuple(obs))
        steps += 1

    return path, done


def plot_grid_path(env, path, success, title="Agent Path"):
    grid_size = env.grid_size
    fig, ax = plt.subplots(figsize=(5, 5))

    # Draw grid lines
    for i in range(grid_size + 1):
        ax.axhline(i, color="gray", linewidth=0.5)
        ax.axvline(i, color="gray", linewidth=0.5)

    # Draw obstacles
    for (ox, oy) in env.obstacles:
        ax.add_patch(patches.Rectangle((oy, grid_size - 1 - ox), 1, 1, color="black"))

    # Draw goal
    gx, gy = env.goal
    ax.add_patch(patches.Rectangle((gy, grid_size - 1 - gx), 1, 1, color="green", alpha=0.6))

    # Draw path
    xs = [p[1] + 0.5 for p in path]
    ys = [grid_size - 1 - p[0] + 0.5 for p in path]
    ax.plot(xs, ys, marker="o", color="blue" if success else "red", linewidth=2, markersize=6)

    # Mark start and end
    ax.plot(xs[0], ys[0], marker="s", color="orange", markersize=12, label="Start")
    ax.plot(xs[-1], ys[-1], marker="*", color="purple", markersize=15, label="End")

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")
    ax.set_title(title + (" ✅ Success" if success else " ❌ Failed/Timeout"))
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig