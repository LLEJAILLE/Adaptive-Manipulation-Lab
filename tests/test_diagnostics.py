import unittest

import numpy as np
import torch

from adaptive_manipulation.data.checkpoints import check_environment_compatibility
from adaptive_manipulation.core.config import SACConfig
from adaptive_manipulation.data.diagnostics import CRITIC_FIELDS, aggregate_critics
from adaptive_manipulation.simulation.gym_env import ManipulationEnv
from adaptive_manipulation.learning.sac import SACAgent


class DiagnosticTests(unittest.TestCase):
    def test_td_partition_matches_total_loss_and_missing_terminals_are_explicit(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        agent = SACAgent(3,4,SACConfig(hidden_sizes=(8,)))
        batch = dict(observations=torch.zeros(3,3),actions=torch.zeros(3,4),
                     next_observations=torch.zeros(3,3),rewards=torch.tensor([[-100.],[.1],[.2]]),
                     terminated=torch.tensor([[1.],[0.],[0.]]),truncated=torch.zeros(3,1))
        metrics = agent.update(batch)
        self.assertTrue(all(np.isfinite(metrics[key]) for key in CRITIC_FIELDS))
        self.assertEqual(metrics['terminal_samples'],1)
        self.assertEqual(metrics['nonterminal_samples'],2)
        partition = (metrics['td_squared_sum_terminal']+metrics['td_squared_sum_nonterminal'])/3
        np.testing.assert_allclose(partition,metrics['critic_loss'],rtol=1e-6,atol=1e-5)
        batch['terminated'].zero_()
        nonterminal = agent.update(batch)
        self.assertEqual(aggregate_critics([nonterminal])['td_mse_terminal'],'')
        summary = aggregate_critics([metrics,nonterminal])
        self.assertEqual(summary['terminal_samples'],1)
        self.assertEqual(summary['nonterminal_samples'],5)
        self.assertAlmostEqual(summary['td_mse_terminal'],metrics['td_squared_sum_terminal'])
        self.assertAlmostEqual(summary['td_mse_nonterminal'],
                               (metrics['td_squared_sum_nonterminal']+nonterminal['td_squared_sum_nonterminal'])/5)

    def test_old_observation_and_terminal_semantics_cannot_be_silently_reused(self):
        env = ManipulationEnv()
        self.addCleanup(env.close)
        with self.assertRaisesRegex(ValueError,'remaining time'):
            check_environment_compatibility({'agent':{'observation_dim':36}},env)
        check_environment_compatibility({'observation_version':2,'task_version':2,
                                         'agent':{'observation_dim':38}},env)


if __name__ == '__main__':
    unittest.main()
