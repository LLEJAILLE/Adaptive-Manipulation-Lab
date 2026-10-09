"""Gymnasium adapter; the manual scenario engine remains the source of rewards."""

import copy
import math
import random
import time

import gymnasium as gym
import mujoco
import numpy as np

from adaptive_manipulation.simulation.environment import Environment
from adaptive_manipulation.tasks.reach import GRIPPER_OPEN, MAX_DURATION


class ManipulationEnv(gym.Env):
    metadata = {"render_modes": ["human", "machine"], "render_fps": 50}

    def __init__(self, render_mode="machine", physics_steps=10, max_steps=1000,
                 arm_speed=math.radians(50), gripper_speed=0.045, xml_path=None):
        if isinstance(physics_steps, bool) or not isinstance(physics_steps, int) or physics_steps < 1:
            raise ValueError("physics_steps must be a positive integer")
        if not all(math.isfinite(x) and x > 0 for x in (arm_speed, gripper_speed)):
            raise ValueError("Action speeds must be positive and finite")
        self.render_mode = render_mode
        self.physics_steps = physics_steps
        self.core = Environment(mode="ai", render_mode=render_mode, num_episodes=1,
                                max_steps=max_steps, xml_path=xml_path)
        self.model, self.data = self.core.model, self.core.data
        self.decision_dt = float(self.model.opt.timestep) * physics_steps
        self.increments = np.array([arm_speed] * 3 + [gripper_speed]) * self.decision_dt
        joints = [self.model.joint(f"joint{i}").id for i in (1, 2, 3)]
        arm_ranges = [np.array([max(self.model.jnt_range[j, 0], self.model.actuator_ctrlrange[a, 0]),
                               min(self.model.jnt_range[j, 1], self.model.actuator_ctrlrange[a, 1])])
                      for j, a in zip(joints, self.core._arm_motors)]
        finger_ranges = [self.model.jnt_range[self.model.joint(f"gripper_{s}").id]
                         for s in ("left", "right")]
        finger_ranges += [self.model.actuator_ctrlrange[a] for a in self.core._gripper_motors]
        gripper_range = [max(r[0] for r in finger_ranges), min(r[1] for r in finger_ranges)]
        self.limits = np.array(arm_ranges + [gripper_range])
        if np.any(self.limits[:, 0] > self.limits[:, 1]):
            raise ValueError("Joint and actuator limits do not overlap")
        self.targets = np.array([0., 0., 0., GRIPPER_OPEN])
        self.action_space = gym.spaces.Box(-1., 1., shape=(4,), dtype=np.float32)
        # Targets are included because incremental control has memory. Include
        # cube orientation/velocity through full qpos/qvel, not just its centre.
        # Append deadline and decision-budget fractions; earlier slices stay stable.
        size = self.model.nq + self.model.nv + 9 + 4 + 2
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
        self._needs_reset = True
        self._viewer = None

    def _observation(self):
        obs = self.core._observation()
        time_remaining = self.core.remaining / MAX_DURATION
        if time_remaining < 1e-9:
            time_remaining = 0.0
        steps_remaining = (max(0., 1 - self.core.steps / self.core.max_steps)
                           if self.core.max_steps is not None else 1.)
        return np.concatenate([obs["qpos"], obs["qvel"], obs["gripper_pos"],
                               obs["cube_pos"], obs["cube_pos"] - obs["gripper_pos"],
                               self.targets, [time_remaining, steps_remaining]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.core.seed = int(seed)
            self.core.rng = random.Random(int(seed))
        self.core.episode = 0
        self.core.results.clear()
        self.core.is_done = False
        self.core._start_scenario()
        self.targets[:] = [0., 0., 0., GRIPPER_OPEN]
        self._needs_reset = False
        if self.render_mode == "human":
            self.render()
        return self._observation(), {"seed": self.core.seed, "spawn_position": self.core.spawn_position}

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Call reset() before step() and after every episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (4,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("Action must be a finite vector of four values in [-1, 1]")
        start = time.perf_counter()
        before = float(self.data.time)
        previous_targets = self.targets.copy()
        self.targets[:] = np.clip(self.targets + action * self.increments,
                                  self.limits[:, 0], self.limits[:, 1])
        _, reward, terminated, truncated, info = self.core.step(
            self.targets[:3], self.targets[3], physics_steps=self.physics_steps)
        self._needs_reset = bool(terminated or truncated)
        info.update(self.action_diagnostics(action, previous_targets))
        if self.render_mode == "human":
            self.render()
            time.sleep(max(0., float(self.data.time) - before - (time.perf_counter() - start)))
        return self._observation(), reward, bool(terminated), bool(truncated), info

    def action_diagnostics(self, action, previous_targets):
        """Scalar measurements of the executed control, one value per channel."""
        actual = self.data.qpos[:3].tolist() + [float(np.mean(self.data.qpos[self.core._gripper_qpos]))]
        effective = (self.targets - previous_targets) / self.increments
        values = {}
        for i in range(4):
            values.update({f"action_{i}": float(action[i]),
                           f"action_abs_{i}": float(abs(action[i])),
                           f"action_saturated_{i}": float(abs(action[i]) >= .95),
                           f"effective_action_{i}": float(effective[i]),
                           f"action_clipped_{i}": float(abs(effective[i] - action[i]) > 1e-6),
                           f"target_at_limit_{i}": float(np.any(np.isclose(self.targets[i], self.limits[i], atol=1e-7, rtol=0))),
                           f"tracking_error_{i}": float(abs(self.targets[i] - actual[i]))})
        values["distance_below_30cm"] = float(self.core.final_distance < .3)
        return values

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self._viewer.is_running():
                self._viewer.sync()

    @property
    def viewer_closed(self):
        return self._viewer is not None and not self._viewer.is_running()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def state_dict(self):
        """Full integration state allows checkpoints even in mid-episode."""
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, state, spec)
        names = ("seed", "episode", "is_done", "spawn_position", "reward", "total_reward",
                 "steps", "success", "final_distance", "results", "_simulation_started_at")
        return {"physics": state, "core": {n: copy.deepcopy(getattr(self.core, n)) for n in names},
                "scenario_rng": self.core.rng.getstate(), "targets": self.targets.copy(),
                "needs_reset": self._needs_reset,
                "reward_state": copy.deepcopy(vars(self.core.reward_calculator)),
                "gym_rng": copy.deepcopy(self.np_random.bit_generator.state),
                "action_rng": copy.deepcopy(self.action_space.np_random.bit_generator.state)}

    def load_state_dict(self, state):
        mujoco.mj_setState(self.model, self.data, state["physics"], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, self.data)
        for name, value in state["core"].items():
            setattr(self.core, name, copy.deepcopy(value))
        self.core.rng.setstate(state["scenario_rng"])
        self.targets[:] = state["targets"]
        self._needs_reset = state["needs_reset"]
        vars(self.core.reward_calculator).update(copy.deepcopy(state["reward_state"]))
        self.np_random.bit_generator.state = state["gym_rng"]
        self.action_space.np_random.bit_generator.state = state["action_rng"]
