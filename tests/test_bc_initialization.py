import contextlib
import csv
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
import torch

from adaptive_manipulation.learning.behavior_cloning import BCPolicy, initialize_std
from adaptive_manipulation.core.config import BCConfig
from adaptive_manipulation.data.checkpoints import save_bc_checkpoint
from adaptive_manipulation.data.demonstrations import split_episodes
from adaptive_manipulation.data.checkpoints import load_checkpoint
from adaptive_manipulation.core.config import SACConfig, TrainingConfig
from adaptive_manipulation.data.demonstrations import DemoRecorder, environment_signature, load_demonstrations
from adaptive_manipulation.workflows.evaluate import main as evaluate_main
from adaptive_manipulation.simulation.gym_env import ManipulationEnv
from adaptive_manipulation.workflows.train_sac import train


class BCInitializationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = TrainingConfig(total_steps=4,max_episode_steps=2,batch_size=2,replay_capacity=50,
            start_steps=99,update_after=99,log_every=2,evaluate_every=4,checkpoint_every=1,
            validation_episodes=1,device='cpu',sac=SACConfig(hidden_sizes=(8,)),bc_critic_warmup_updates=3)
        env = ManipulationEnv(**self.config.environment_kwargs())
        try:
            self.source = self.root/'demos'
            recorder = DemoRecorder(self.source,env,episodes=4)
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
                following,reward,terminated,truncated,info = env.step(action)
                recorder.add(observation,action,reward,following,terminated,truncated,info)
                recorder.finish(info)
            data,_ = load_demonstrations(self.source)
            _,_,split = split_episodes(data)
            split['source'] = str(self.source)
            split['source_hashes'] = {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in self.source.glob('episode_*.npz')}
            policy = BCPolicy(38,4,(8,))
            initialize_std(policy.actor,.3)
            self.bc = self.root/'bc.pt'
            save_bc_checkpoint(self.bc,policy,self.config,BCConfig(),environment_signature(env),split,1,{})
            self.bc_payload = load_checkpoint(self.bc)
        finally:
            env.close()

    def run_transfer(self,name='sac'):
        with contextlib.redirect_stdout(io.StringIO()):
            train(self.config,self.root/name,init_bc=self.bc)
        return load_checkpoint(self.root/name/'latest.pt')

    def test_critic_warmup_preserves_actor_alpha_and_excludes_validation(self):
        self.config.total_steps = 2
        latest = self.run_transfer()
        initial = load_checkpoint(self.root/'sac/initialized.pt')
        for key,value in self.bc_payload['actor'].items():
            torch.testing.assert_close(latest['agent']['actor'][key],value,rtol=0,atol=0)
            torch.testing.assert_close(initial['agent']['actor'][key],value,rtol=0,atol=0)
        torch.testing.assert_close(latest['agent']['log_alpha'],initial['agent']['log_alpha'],rtol=0,atol=0)
        self.assertFalse(latest['agent']['actor_optimizer']['state'])
        self.assertTrue(any(not torch.equal(value,latest['agent']['q1'][key]) for key,value in initial['agent']['q1'].items()))
        self.assertEqual(latest['agent']['updates'],2)
        self.assertEqual(initial['replay']['size'],3)
        self.assertEqual(latest['replay']['size'],5)
        source,_ = load_demonstrations(self.source)
        mask = np.isin(source['episode_ids'],self.bc_payload['split']['train_episode_ids'])
        np.testing.assert_array_equal(initial['replay']['arrays']['observations'],source['observations'][mask])
        self.assertFalse(set(latest['config']['train_seeds']) & set(self.bc_payload['split']['validation_seeds']))
        with (self.root/'sac/stats.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[-1]['phase'],'critic_warmup')
        self.assertEqual(float(rows[-1]['actor_update_fraction']),0)

    def test_resume_across_warmup_boundary_matches_continuous_training(self):
        original = self.run_transfer('original')
        checkpoint = load_checkpoint(self.root/'original/step_000000001.pt')
        self.bc.rename(self.root/'bc_archived.pt')  # Resume must not need BC/source files again.
        config = TrainingConfig.from_dict(checkpoint['config'])
        with contextlib.redirect_stdout(io.StringIO()):
            train(config,self.root/'resumed',resume=checkpoint)
        resumed = load_checkpoint(self.root/'resumed/latest.pt')
        for name in ('actor','q1','q2','target_q1','target_q2'):
            for key,value in original['agent'][name].items():
                torch.testing.assert_close(value,resumed['agent'][name][key],atol=1e-7,rtol=0)
        torch.testing.assert_close(original['agent']['log_alpha'],resumed['agent']['log_alpha'],atol=1e-7,rtol=0)
        self.assertTrue(original['agent']['actor_optimizer']['state'])
        self.assertTrue(any(not torch.equal(value,original['agent']['actor'][key]) for key,value in self.bc_payload['actor'].items()))
        np.testing.assert_array_equal(original['replay']['arrays']['actions'],resumed['replay']['arrays']['actions'])
        self.assertEqual(resumed['training']['bc_initialization'],original['training']['bc_initialization'])

    def test_rejects_validation_seed_leakage_changed_data_and_small_replay(self):
        self.config.train_seeds = self.bc_payload['split']['validation_seeds']
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'held-out'):
            train(self.config,self.root/'leak',init_bc=self.bc)
        self.config.train_seeds = None
        self.config.replay_capacity = 2
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'capacity'):
            train(self.config,self.root/'small',init_bc=self.bc)
        self.config.replay_capacity = 50
        filename = f"episode_{self.bc_payload['split']['train_episode_ids'][0]:04d}.npz"
        with (self.source/filename).open('ab') as stream:
            stream.write(b'changed')
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'changed since'):
            train(self.config,self.root/'changed',init_bc=self.bc)

    def test_evaluation_rejects_actual_bc_training_seeds_in_sac_checkpoint(self):
        self.run_transfer()
        seed = self.bc_payload['split']['train_seeds'][0]
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            evaluate_main([str(self.root/'sac/best.pt'),'--seeds',str(seed)])


if __name__ == '__main__':
    unittest.main()
