# Allowed reward components and safe weight ranges
REWARD_SCHEMA = {
    "goal": {"min": 0.0, "max": 20.0},
    "obstacle": {"min": -20.0, "max": 0.0},
    "efficiency": {"min": -2.0, "max": 0.0},
}

def validate_reward_config(config):
    """
    config example:
    {"goal": 10.0, "obstacle": -5.0, "efficiency": -0.1}
    """
    if not isinstance(config, dict):
        return False, "Config must be a dictionary."

    for key in REWARD_SCHEMA:
        if key not in config:
            return False, f"Missing required component: {key}"

    for key, value in config.items():
        if key not in REWARD_SCHEMA:
            return False, f"Unknown component: {key}"
        if not isinstance(value, (int, float)):
            return False, f"Component '{key}' must be numeric."
        lo, hi = REWARD_SCHEMA[key]["min"], REWARD_SCHEMA[key]["max"]
        if not (lo <= value <= hi):
            return False, f"Component '{key}'={value} out of allowed range [{lo}, {hi}]"

    return True, "Valid"


def build_reward_fn(config):
    """
    Converts a validated config dict into a callable reward function
    matching the signature GridWorldEnv expects.
    """
    goal_reward = config["goal"]
    obstacle_penalty = config["obstacle"]
    step_penalty = config["efficiency"]

    def reward_fn(collided, terminated, step_count):
        if terminated:
            return goal_reward
        elif collided:
            return obstacle_penalty
        else:
            return step_penalty

    return reward_fn


if __name__ == "__main__":
    # Quick manual test
    manual_config = {"goal": 10.0, "obstacle": -5.0, "efficiency": -0.1}

    valid, msg = validate_reward_config(manual_config)
    print(f"Valid: {valid}, Message: {msg}")

    if valid:
        reward_fn = build_reward_fn(manual_config)
        print("Test calls:")
        print("Reached goal:", reward_fn(collided=False, terminated=True, step_count=5))
        print("Collided:", reward_fn(collided=True, terminated=False, step_count=5))
        print("Normal step:", reward_fn(collided=False, terminated=False, step_count=5))

    # Test an invalid config
    bad_config = {"goal": 10.0, "obstacle": -5.0, "efficiency": 5.0}  # efficiency out of range
    valid, msg = validate_reward_config(bad_config)
    print(f"\nBad config -> Valid: {valid}, Message: {msg}")