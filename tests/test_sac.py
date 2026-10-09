import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.distributions import Normal

from adaptive_manipulation.data.checkpoints import load_checkpoint, random_state, restore_random_state, save_checkpoint
from adaptive_manipulation.core.config import SACConfig, TrainingConfig
from adaptive_manipulation.learning.replay_buffer import ReplayBuffer
from adaptive_manipulation.learning.sac import Actor, SACAgent
from adaptive_manipulation.workflows.train_sac import train


class SACTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.config = SACConfig(hidden_sizes=(16, 16))

    def test_actor_log_probability_and_saturation(self):
        actor = Actor(3, 2, (8,))
        observations = torch.randn(10, 3)
        action, log_prob = actor.sample(observations)
        self.assertEqual(action.shape, (10, 2))
        self.assertEqual(log_prob.shape, (10, 1))
        self.assertTrue((action.abs() <= 1).all())
        raw = torch.atanh(action)
        mean, log_std = actor.network(observations).chunk(2, -1)
        expected = (Normal(mean, log_std.clamp(-20, 2).exp()).log_prob(raw)
                    - torch.log1p(-action.square())).sum(-1, keepdim=True)
        torch.testing.assert_close(log_prob, expected, atol=1e-5, rtol=1e-5)
        with torch.no_grad():
            actor.network[-1].bias[:2].fill_(100)
        _, saturated = actor.sample(observations, deterministic=True)
        self.assertTrue(torch.isfinite(saturated).all())

    def buffer(self):
        buffer = ReplayBuffer(40, 3, 2, seed=7)
        rng = np.random.default_rng(7)
        for i in range(50):
            buffer.add(rng.normal(size=3), rng.uniform(-1, 1, 2), float(i), rng.normal(size=3), i % 3 == 0, i % 3 == 1)
        return buffer

    def test_replay_wraparound_and_restore_rng(self):
        original = self.buffer()
        restored = ReplayBuffer(40, 3, 2)
        restored.load_state_dict(original.state_dict())
        self.assertEqual(len(original), 40)
        self.assertEqual(original.position, 10)
        self.assertEqual(original.arrays["rewards"][9, 0], 49)
        for key, value in original.sample(8).items():
            # Sample once on each buffer, rather than advancing per field.
            if key == "observations":
                repeated = restored.sample(8)
            torch.testing.assert_close(value, repeated[key])

    def test_target_bootstraps_only_truncation(self):
        agent = SACAgent(3, 2, SACConfig(hidden_sizes=(8,), initial_alpha=.2, automatic_entropy=False))
        for net in (agent.target_q1, agent.target_q2):
            for param in net.parameters():
                param.data.zero_()
            net.network[-1].bias.data.fill_(5)
        batch = {"next_observations": torch.zeros(2, 3), "rewards": torch.ones(2, 1),
                 "terminated": torch.tensor([[1.], [0.]]), "truncated": torch.tensor([[0.], [1.]])}
        agent.actor.sample = lambda obs: (torch.zeros(len(obs), 2), torch.full((len(obs), 1), -2.))
        target = agent.bellman_target(batch)
        torch.testing.assert_close(target, torch.tensor([[1.], [1 + .99 * (5 + .4)]]))

    def test_update_changes_networks_alpha_and_polyak(self):
        agent = SACAgent(3, 2, self.config)
        before_actor = [p.detach().clone() for p in agent.actor.parameters()]
        before_q = [p.detach().clone() for p in agent.q1.parameters()]
        before_target = [p.detach().clone() for p in agent.target_q1.parameters()]
        before_alpha = agent.alpha.clone()
        losses = agent.update(self.buffer().sample(8))
        self.assertTrue(all(np.isfinite(x) for x in losses.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_actor, agent.actor.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_q, agent.q1.parameters())))
        self.assertFalse(torch.equal(before_alpha, agent.alpha))
        for old, target, online in zip(before_target, agent.target_q1.parameters(), agent.q1.parameters()):
            torch.testing.assert_close(target, old * (1 - self.config.tau) + online * self.config.tau)
        self.assertEqual(agent.updates, 1)

    def test_checkpoint_model_optimizer_replay_and_random_states(self):
        agent = SACAgent(3, 2, self.config)
        replay = self.buffer()
        agent.update(replay.sample(8))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_checkpoint(path, agent, TrainingConfig(sac=self.config), {"global_step": 50}, replay)
            payload = load_checkpoint(path)
            restored = SACAgent(3, 2, self.config)
            restored.load_state_dict(payload["agent"])
        np.testing.assert_array_equal(agent.act(np.zeros(3), True), restored.act(np.zeros(3), True))
        self.assertEqual(restored.updates, 1)
        self.assertEqual(len(restored.actor_optimizer.state), len(agent.actor_optimizer.state))
        state = random_state()
        expected = torch.rand(4)
        restore_random_state(state)
        torch.testing.assert_close(torch.rand(4), expected)
        self.assertEqual(payload["training"]["global_step"], 50)

    def test_seed_pool_overlap_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(validation_seed_start=42).validate()

    def test_single_seed_training_and_resume(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            config = TrainingConfig(total_steps=9, max_episode_steps=3, replay_capacity=20,
                                    batch_size=4, start_steps=100, update_after=100,
                                    log_every=9, evaluate_every=9, checkpoint_every=4,
                                    validation_episodes=1, device="cpu", sac=self.config)
            self.assertEqual((config.train_seed_start, config.train_seed_stop), (42, 43))
            original = Path(folder) / "original"
            restored = Path(folder) / "restored"
            train(config, original)
            checkpoint = load_checkpoint(original / "step_000000004.pt")
            train(config, restored, checkpoint)
            for directory, count in ((original, 3), (restored, 2)):
                with (directory / "episodes.csv").open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), count)
                self.assertEqual({int(row["seed"]) for row in rows}, {42})
            original_state = load_checkpoint(original / "latest.pt")
            restored_state = load_checkpoint(restored / "latest.pt")
            self.assertEqual(original_state["environment"]["core"]["spawn_position"],
                             restored_state["environment"]["core"]["spawn_position"])

    def test_full_training_resume_matches_continuous_training(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            config = TrainingConfig(total_steps=12, max_episode_steps=5, batch_size=4,
                                    device="cpu",
                                    start_steps=4, update_after=4, replay_capacity=30,
                                    log_every=8, evaluate_every=8, checkpoint_every=6,
                                    validation_episodes=2, sac=self.config)
            uninterrupted = Path(folder) / "uninterrupted"
            split = Path(folder) / "split"
            train(config, uninterrupted)
            # Periodic snapshot at step 6 contains an active episode and RNGs.
            saved = load_checkpoint(uninterrupted / "step_000000006.pt")
            train(config, split, saved)
            left = load_checkpoint(uninterrupted / "latest.pt")
            right = load_checkpoint(split / "latest.pt")
            for name in ("actor", "q1", "q2", "target_q1", "target_q2"):
                for key, value in left["agent"][name].items():
                    torch.testing.assert_close(value, right["agent"][name][key], rtol=0, atol=1e-7)
            for name in ('critics.csv', 'actions.csv'):
                with (uninterrupted / name).open(newline='') as stream:
                    original_rows = list(csv.DictReader(stream))
                with (split / name).open(newline='') as stream:
                    restored_rows = list(csv.DictReader(stream))
                self.assertEqual(len(original_rows),len(restored_rows))
                for original_row,restored_row in zip(original_rows,restored_rows):
                    self.assertEqual(original_row.keys(),restored_row.keys())
                    for key in original_row:
                        if original_row[key] == '':
                            self.assertEqual(restored_row[key],'')
                        else:
                            self.assertAlmostEqual(float(original_row[key]),float(restored_row[key]),places=6)
            self.assertEqual(left["training"]["episodes"], right["training"]["episodes"])
            np.testing.assert_allclose(left["environment"]["physics"], right["environment"]["physics"], atol=1e-10)
            self.assertTrue((split / "episodes.csv").exists())
            self.assertTrue((split / "best.pt").exists())


if __name__ == "__main__":
    unittest.main()
