import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
import torch
from torch.nn import functional as F

from adaptive_manipulation.learning.behavior_cloning import BCPolicy, initialize_std
from adaptive_manipulation.core.config import BCConfig
from adaptive_manipulation.data.demonstrations import split_episodes
from adaptive_manipulation.workflows.train_bc import train_bc
from adaptive_manipulation.data.checkpoints import check_environment_compatibility, load_checkpoint
from adaptive_manipulation.core.config import SACConfig, TrainingConfig
from adaptive_manipulation.data.demonstrations import DemoRecorder
from adaptive_manipulation.workflows.evaluate import main as evaluate_main
from adaptive_manipulation.simulation.gym_env import ManipulationEnv
from adaptive_manipulation.learning.sac import Actor


class BehaviorCloningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_split_is_reproducible_and_preserves_whole_scenarios(self):
        data = dict(episode_ids=np.repeat(np.arange(1,31),3),seeds=np.repeat(np.arange(3000000,3000030),3))
        train,validation,split = split_episodes(data)
        self.assertEqual(len(split['train_episode_ids']),24)
        self.assertEqual(len(split['validation_episode_ids']),6)
        self.assertFalse(set(split['train_seeds']) & set(split['validation_seeds']))
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation),set(range(90)))
        second = split_episodes(data)
        np.testing.assert_array_equal(train,second[0])
        np.testing.assert_array_equal(validation,second[1])
        for episode in range(1,31):
            indices = set(np.flatnonzero(data['episode_ids']==episode))
            self.assertTrue(indices.issubset(train) or indices.issubset(validation))
        with self.assertRaisesRegex(ValueError,'two successful'):
            split_episodes(dict(episode_ids=np.ones(3),seeds=np.ones(3)))

    def test_mse_learns_actions_while_std_remains_fixed(self):
        actor = Actor(3,4,(32,32))
        initialize_std(actor,.3)
        observations = torch.randn(32,3)
        actions = torch.tanh(torch.cat([observations,torch.zeros(32,1)],dim=1)*.3)
        optimizer = torch.optim.Adam(actor.parameters(),lr=.01)
        before = F.mse_loss(torch.tanh(actor.distribution_parameters(observations)[0]),actions).item()
        for _ in range(60):
            mean,_ = actor.distribution_parameters(observations)
            loss = F.mse_loss(torch.tanh(mean),actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean,log_std = actor.distribution_parameters(observations)
        self.assertLess(F.mse_loss(torch.tanh(mean),actions).item(),before*.1)
        torch.testing.assert_close(log_std.exp(),torch.full_like(log_std,.3))

    def make_data(self,root):
        config = TrainingConfig(max_episode_steps=2,device='cpu',sac=SACConfig(hidden_sizes=(8,)))
        env = ManipulationEnv(**config.environment_kwargs())
        try:
            folder = root/'demonstrations'
            recorder = DemoRecorder(folder,env,episodes=4)
            for _ in range(4):
                _,info = env.reset(seed=recorder.next_seed)
                recorder.begin(recorder.next_seed,info['spawn_position'])
                gripper = env.model.body('gripper').id
                geom = next(i for i,b in enumerate(env.model.geom_bodyid) if b == gripper)
                env.data.qpos[env.core._cube_qpos:env.core._cube_qpos+3] = env.data.geom_xpos[geom]
                mujoco.mj_forward(env.model,env.data)
                env.core.reward_calculator.reset(env.data.site_xpos[env.core._gripper_site],env.data.geom_xpos[env.core._cube_geom])
                observation = env._observation()
                action = np.zeros(4,np.float32)
                next_obs,reward,terminated,truncated,info = env.step(action)
                self.assertTrue(info['success'])
                recorder.add(observation,action,reward,next_obs,terminated,truncated,info)
                recorder.finish(info)
            return folder,config
        finally:
            env.close()

    def test_saved_actor_and_evaluation_preserve_split_and_contract(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            folder,config = self.make_data(root)
            output = root/'bc'
            result = train_bc(folder,output,config,BCConfig(epochs=2,batch_size=2,device='cpu',evaluate_every=1))
            payload = load_checkpoint(output/'best.pt')
            self.assertEqual(payload['kind'],'bc')
            self.assertNotIn('q1',payload)
            self.assertNotIn('log_alpha',payload)
            self.assertEqual(result['completed_epochs'],2)
            self.assertIn('best_validation_rollout',result)
            policy = BCPolicy.from_checkpoint(payload)
            env = ManipulationEnv(**config.environment_kwargs())
            try:
                check_environment_compatibility(payload,env)
                obs,_ = env.reset(seed=42)
                expected,_ = policy.actor.sample(torch.tensor(obs).unsqueeze(0),deterministic=True)
                np.testing.assert_allclose(policy.act(obs),expected.detach().numpy()[0],atol=1e-7)
            finally:
                env.close()
            with (output/'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            best_row = min(rows,key=lambda row:float(row['validation_mse']))
            self.assertEqual(payload['epoch'],int(best_row['epoch']))
            self.assertAlmostEqual(result['best_validation_mse'],float(best_row['validation_mse']))
            evaluate_main([str(output/'best.pt'),'--split','validation','--csv',str(root/'evaluation.csv')])
            with (root/'evaluation.csv').open() as stream:
                evaluation = list(csv.DictReader(stream))
            self.assertEqual([int(row['seed']) for row in evaluation],payload['split']['validation_seeds'])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluate_main([str(output/'best.pt'),'--seeds',str(payload['split']['train_seeds'][0])])
            with self.assertRaisesRegex(ValueError,'not empty'):
                train_bc(folder,output,config,BCConfig(epochs=1,rollouts=False))

    def test_configuration_rejects_empty_splits_or_invalid_parameters(self):
        for config in (BCConfig(epochs=0),BCConfig(validation_fraction=1),BCConfig(initial_std=0),BCConfig(learning_rate=float('nan'))):
            with self.assertRaises(ValueError):
                config.validate()


if __name__ == '__main__':
    unittest.main()
