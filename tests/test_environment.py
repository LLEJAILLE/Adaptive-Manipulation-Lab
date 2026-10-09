import unittest
from unittest.mock import patch

from gymnasium.utils.env_checker import check_env
import mujoco
import numpy as np

from adaptive_manipulation.simulation.environment import Environment, GRIPPER_OPEN
from adaptive_manipulation.tasks.reach import RewardCalculator
from adaptive_manipulation.simulation.gym_env import ManipulationEnv


class EnvironmentTests(unittest.TestCase):
    def test_gym_contract_and_reset(self):
        env = ManipulationEnv(max_steps=3)
        self.addCleanup(env.close)
        check_env(env, skip_render_check=True)
        initial, info = env.reset(seed=17)
        self.assertTrue(env.observation_space.contains(initial))
        self.assertEqual(initial.dtype, np.float32)
        env.step(np.ones(4, np.float32))
        repeated, repeated_info = env.reset(seed=17)
        np.testing.assert_array_equal(initial, repeated)
        self.assertEqual(info, repeated_info)
        self.assertEqual(env.core.total_reward, 0)
        self.assertEqual(env.core.steps, 0)
        np.testing.assert_array_equal(env.data.qvel, 0)
        np.testing.assert_allclose(env.targets, [0, 0, 0, GRIPPER_OPEN])
        _, other = env.reset(seed=18)
        self.assertNotEqual(info["spawn_position"], other["spawn_position"])

    def test_action_bounds_limits_and_relative_observation(self):
        env = ManipulationEnv(max_steps=400)
        self.addCleanup(env.close)
        env.reset(seed=42)
        for bad in ([0, 0, 0], [0, 0, 0, np.nan], [1.01, 0, 0, 0]):
            with self.assertRaises(ValueError):
                env.step(bad)
        env.targets[:] = env.limits[:, 1]
        obs, *_ = env.step(np.ones(4))
        np.testing.assert_array_equal(env.targets, env.limits[:, 1])
        np.testing.assert_allclose(obs[29:32], obs[26:29] - obs[23:26], atol=1e-7)
        np.testing.assert_array_equal(obs[32:36], env.targets.astype(np.float32))

    def test_machine_physics_and_reward_frequency(self):
        env = ManipulationEnv(max_steps=3, physics_steps=10)
        self.addCleanup(env.close)
        env.reset(seed=42)
        with patch("time.sleep", side_effect=AssertionError("Headless sleep")), patch.object(
                env.core.reward_calculator, "calculate", wraps=env.core.reward_calculator.calculate) as calculate:
            _, reward, terminated, truncated, info = env.step(np.zeros(4))
        self.assertEqual(calculate.call_count, 1)
        self.assertAlmostEqual(env.model.opt.timestep, 0.002)
        self.assertAlmostEqual(env.data.time, 0.02)
        self.assertFalse(terminated or truncated)
        self.assertAlmostEqual(reward, sum(info[k] for k in (
            "progress_reward", "proximity_reward", "time_penalty", "terminal_reward")))

    def test_observed_deadlines_decrease_and_terminal_target_has_no_future(self):
        import torch
        from adaptive_manipulation.core.config import SACConfig
        from adaptive_manipulation.learning.sac import SACAgent
        env = ManipulationEnv(max_steps=2)
        self.addCleanup(env.close)
        initial, _ = env.reset(seed=42)
        self.assertEqual(len(initial), 38)
        np.testing.assert_array_equal(initial[-2:], [1., 1.])
        middle, *_ = env.step(np.zeros(4))
        self.assertLess(middle[-2], initial[-2])
        self.assertEqual(middle[-1], .5)
        final, reward, terminated, truncated, info = env.step(np.zeros(4))
        self.assertTrue(terminated)
        self.assertFalse(truncated or info['success'])
        self.assertEqual(final[-1], 0)
        agent = SACAgent(38, 4, SACConfig(hidden_sizes=(8,)))
        # A large value after the deadline must have no influence on the target.
        for network in (agent.target_q1, agent.target_q2):
            for parameter in network.parameters():
                parameter.data.zero_()
            network.network[-1].bias.data.fill_(1000)
        batch = dict(next_observations=torch.tensor(final).unsqueeze(0),
                     rewards=torch.tensor([[reward]],dtype=torch.float32),
                     terminated=torch.ones(1,1),truncated=torch.zeros(1,1))
        torch.testing.assert_close(agent.bellman_target(batch),batch['rewards'])

    def test_physical_action_clipping_is_distinguished_from_policy_saturation(self):
        env = ManipulationEnv(max_steps=2)
        self.addCleanup(env.close)
        env.reset(seed=42)
        env.targets[:] = env.limits[:,1]
        _, _, _, _, info = env.step(np.ones(4))
        for i in range(4):
            self.assertEqual(info[f'action_saturated_{i}'], 1)
            self.assertEqual(info[f'action_clipped_{i}'], 1)
            self.assertEqual(info[f'effective_action_{i}'], 0)
            self.assertEqual(info[f'target_at_limit_{i}'], 1)
            self.assertGreaterEqual(info[f'tracking_error_{i}'], 0)

    def test_manual_reward_parity(self):
        manual = Environment(seed=42, render_mode=None, clock=lambda: 0, num_episodes=1, max_steps=10)
        ai = ManipulationEnv(max_steps=10)
        self.addCleanup(ai.close)
        ai.reset(seed=42)
        for _ in range(3):
            a = ai.step(np.zeros(4))
            m = manual.step([0, 0, 0], GRIPPER_OPEN, physics_steps=10)
            self.assertEqual(a[1:4], m[1:4])
            np.testing.assert_array_equal(ai.data.qpos, manual.data.qpos)

    def test_failure_retains_final_state_and_requires_reset(self):
        env = ManipulationEnv(max_steps=1)
        self.addCleanup(env.close)
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(np.zeros(4))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info['success'])
        self.assertEqual(obs[-1], 0)
        self.assertEqual(info["terminal_reward"], -100)
        self.assertEqual(info["reason"], "step_limit")
        self.assertEqual(env.core.total_reward, reward)
        np.testing.assert_array_equal(obs, env._observation())
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(4))
        env.reset(seed=43)
        self.assertFalse(env.core.is_done)

    def test_real_contact_terminates(self):
        env = ManipulationEnv(max_steps=1)
        self.addCleanup(env.close)
        env.reset(seed=42)
        gripper_body = env.model.body("gripper").id
        geom = next(i for i, body in enumerate(env.model.geom_bodyid) if body == gripper_body)
        env.data.qpos[env.core._cube_qpos:env.core._cube_qpos + 3] = env.data.geom_xpos[geom]
        mujoco.mj_forward(env.model, env.data)
        env.core.reward_calculator.reset(env.data.site_xpos[env.core._gripper_site], env.data.geom_xpos[env.core._cube_geom])
        _, _, terminated, truncated, info = env.step(np.zeros(4))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["terminal_reward"], 200)

    def test_simulation_timeout(self):
        env = ManipulationEnv(max_steps=None, physics_steps=10)
        self.addCleanup(env.close)
        env.reset(seed=42)
        env.data.time = 19.99
        _, _, terminated, truncated, info = env.step(np.zeros(4))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info['success'])
        self.assertEqual(env._observation()[-2], 0)
        self.assertEqual(info["reason"], "timeout")

    def test_manual_timer_autoreset_preserved(self):
        now = [0.]
        env = Environment(seed=42, render_mode=None, num_episodes=2, clock=lambda: now[0])
        original = env.spawn_position
        now[0] = 21.
        _, _, terminated, truncated, info = env.step([0, 0, 0], GRIPPER_OPEN)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["reason"], "timeout")
        self.assertEqual(info["scenario_result"].spawn_position, original)
        self.assertIn("next_observation", info)
        self.assertEqual(env.episode, 2)
        self.assertEqual(env.total_reward, 0)

    def test_mid_episode_physics_restore(self):
        env = ManipulationEnv()
        restored = ManipulationEnv()
        self.addCleanup(env.close)
        self.addCleanup(restored.close)
        env.reset(seed=42)
        env.step(np.array([.2, -.5, .7, -.8]))
        restored.load_state_dict(env.state_dict())
        for _ in range(4):
            original = env.step(np.array([-.1, .5, .2, .3]))
            repeated = restored.step(np.array([-.1, .5, .2, .3]))
            np.testing.assert_allclose(original[0], repeated[0], rtol=0, atol=1e-7)
            self.assertAlmostEqual(original[1], repeated[1], places=10)
            self.assertEqual(original[2:4], repeated[2:4])


class RewardTests(unittest.TestCase):
    def test_potential_telescopes_and_terminal_once(self):
        calculator = RewardCalculator()
        cube = np.zeros(3)
        calculator.reset([1, 0, 0], cube)
        first, _ = calculator.calculate([.5, 0, 0], cube, False)
        second, _ = calculator.calculate([1, 0, 0], cube, False)
        self.assertAlmostEqual(first + second, -.2)
        reward, _ = calculator.calculate([1, 0, 0], cube, False, failed=True)
        self.assertAlmostEqual(reward, -100.1)

    def test_success_and_failure_are_mutually_exclusive(self):
        calculator = RewardCalculator()
        calculator.reset([1, 0, 0], np.zeros(3))
        with self.assertRaises(ValueError):
            calculator.calculate([1, 0, 0], np.zeros(3), True, failed=True)


if __name__ == "__main__":
    unittest.main()
