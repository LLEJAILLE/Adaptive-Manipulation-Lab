import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from adaptive_manipulation.data.demonstrations import DemoRecorder, environment_signature, load_demonstrations, read_episode
from adaptive_manipulation.simulation.environment import GRIPPER_OPEN
from adaptive_manipulation.interfaces.actions import normalized_gamepad_action
from adaptive_manipulation.simulation.gym_env import ManipulationEnv


class DemonstrationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.output = Path(self.folder.name) / 'demos'
        self.env = ManipulationEnv(max_steps=3)
        self.addCleanup(self.env.close)

    def start(self, recorder):
        obs, info = self.env.reset(seed=recorder.next_seed)
        recorder.begin(recorder.next_seed, info['spawn_position'])
        return obs

    def step(self, recorder, obs, action=None):
        if action is None:
            action = np.zeros(4,np.float32)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        recorder.add(obs,action,reward,next_obs,terminated,truncated,info)
        return next_obs,terminated,truncated,info

    def test_30_distinct_completed_scenarios_are_saved_even_on_failure(self):
        recorder = DemoRecorder(self.output,self.env,episodes=30)
        for _ in range(30):
            obs = self.start(recorder)
            while True:
                obs, terminated, truncated, info = self.step(recorder,obs)
                if terminated or truncated:
                    recorder.finish(info)
                    break
        self.assertTrue(recorder.done)
        self.assertEqual(recorder.successes,0)
        self.assertEqual(len(set(row['seed'] for row in recorder.completed)),30)
        data,signature = load_demonstrations(self.output,success_only=False)
        self.assertEqual(data['observations'].shape,(90,38))
        self.assertEqual(signature,environment_signature(self.env))
        self.assertEqual(int(data['terminated'].sum()),30)
        self.assertEqual(int(data['truncated'].sum()),0)
        with self.assertRaisesRegex(ValueError,'successful'):
            load_demonstrations(self.output)
        with self.assertRaises(ValueError):
            DemoRecorder(self.output,self.env,episodes=30)

    def test_partial_resume_and_filtering_successes(self):
        recorder = DemoRecorder(self.output,self.env,episodes=2,seed_start=100)
        obs = self.start(recorder)
        self.step(recorder,obs)
        recorder.interrupt()
        self.assertEqual(len(list(self.output.glob('partial_*.npz'))),1)
        resumed = DemoRecorder(self.output,self.env,episodes=2,seed_start=100,resume=True)
        self.assertEqual(resumed.next_seed,100)
        obs = self.start(resumed)
        # Force a genuine MuJoCo contact before obtaining the initial recorded state.
        gripper = self.env.model.body('gripper').id
        geom = next(i for i,b in enumerate(self.env.model.geom_bodyid) if b == gripper)
        self.env.data.qpos[self.env.core._cube_qpos:self.env.core._cube_qpos+3] = self.env.data.geom_xpos[geom]
        mujoco.mj_forward(self.env.model,self.env.data)
        self.env.core.reward_calculator.reset(self.env.data.site_xpos[self.env.core._gripper_site],
                                              self.env.data.geom_xpos[self.env.core._cube_geom])
        obs = self.env._observation()
        _, terminated, _, info = self.step(resumed,obs)
        self.assertTrue(terminated and info['success'])
        resumed.finish(info)
        obs = self.start(resumed)
        while True:
            obs, terminated, truncated, info = self.step(resumed,obs)
            if terminated or truncated:
                resumed.finish(info)
                break
        data,_ = load_demonstrations(self.output)
        self.assertEqual(len(data['actions']),1)
        np.testing.assert_array_equal(data['seeds'],[100])
        all_data,_ = load_demonstrations(self.output,success_only=False)
        self.assertEqual(len(all_data['actions']),4)

    def test_recorded_actions_reproduce_transitions_and_no_aliasing(self):
        recorder = DemoRecorder(self.output,self.env,episodes=1,seed_start=42)
        obs = self.start(recorder)
        for axes in ((.3,.7,-.5),(-.4,.2,.1),(0.,0.,0.)):
            action = normalized_gamepad_action(axes,0.,self.env.targets,self.env.increments)
            obs, _, _, info = self.step(recorder,obs,action)
            action[:] = 1  # Storage must own its values, not alias mutable callers.
        recorder.finish(info)
        metadata,data = read_episode(self.output/'episode_0001.npz')
        obs,_ = self.env.reset(seed=42)
        for i,action in enumerate(data['actions']):
            np.testing.assert_array_equal(obs,data['observations'][i])
            obs,reward,terminated,truncated,_ = self.env.step(action)
            np.testing.assert_array_equal(obs,data['next_observations'][i])
            self.assertAlmostEqual(reward,float(data['rewards'][i,0]),places=5)
        self.assertEqual(metadata['reason'],'step_limit')
        self.assertTrue(terminated and not truncated)

    def test_gamepad_opening_and_reset_never_jump_targets(self):
        action = normalized_gamepad_action((.2,-.3,.7),0.,self.env.targets,self.env.increments)
        np.testing.assert_array_equal(action,np.array([.2,-.3,.7,-1],np.float32))
        self.env.targets[:] = [.8,-.2,.5,0.]
        reset = normalized_gamepad_action((0.,0.,0.),0.,self.env.targets,self.env.increments,True)
        np.testing.assert_array_equal(reset,[-1,1,-1,1])
        self.assertTrue(np.all(np.abs(reset)<=1))

    def test_resume_recovers_completed_file_after_manifest_write_failure(self):
        recorder = DemoRecorder(self.output,self.env,episodes=1)
        original = (self.output/'session.json').read_text(encoding='utf-8')
        obs = self.start(recorder)
        for _ in range(3):
            obs,_,_,info = self.step(recorder,obs)
        recorder.finish(info)
        (self.output/'session.json').write_text(original,encoding='utf-8')
        restored = DemoRecorder(self.output,self.env,episodes=1,resume=True)
        self.assertTrue(restored.done)
        self.assertEqual(len(json.loads((self.output/'session.json').read_text())['completed']),1)

    def test_resume_rejects_changed_control_contract(self):
        DemoRecorder(self.output,self.env,episodes=2)
        changed = ManipulationEnv(max_steps=4)
        self.addCleanup(changed.close)
        with self.assertRaisesRegex(ValueError,'configuration'):
            DemoRecorder(self.output,changed,episodes=2,resume=True)


if __name__ == '__main__':
    unittest.main()
