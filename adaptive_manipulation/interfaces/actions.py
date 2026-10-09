"""Map human controls to the same normalized incremental actions as SAC."""

def normalized_gamepad_action(axes, gripper_goal, targets, increments, resetting=False):
    """Convert controller requests to SAC commands without any position jumps."""
    import numpy as np
    from adaptive_manipulation.tasks.reach import GRIPPER_OPEN
    if resetting:
        desired = np.array([0., 0., 0., GRIPPER_OPEN])
        return np.clip((desired-targets)/increments, -1, 1).astype(np.float32)
    return np.array([*axes, np.clip((gripper_goal-targets[3])/increments[3], -1, 1)], dtype=np.float32)
