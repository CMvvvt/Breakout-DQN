import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
from ConvDQN import ConvDQN
from agent import DQNAgent
import torch.nn as nn


# Assuming DQNAgent and other dependencies are imported
class MockConvDQN(nn.Module):
    def __init__(self, input_dims, n_actions, lr):
        super().__init__()  # This should now work as expected
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr = lr

    def forward(self, x):
        # Return a fake tensor that mimics the expected output of the actual ConvDQN
        return torch.rand((x.shape[0], self.n_actions))


class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        self.input_dims = (210, 160, 3)  # Example input dimensions
        self.n_actions = 4
        self.mem_size = 1000
        self.batch_size = 32
        self.eps_min = 0.01
        self.eps_max = 1.0
        self.max_steps = 10000

        # Create an instance of DQNAgent
        with patch("ConvDQN.ConvDQN", new=MockConvDQN):
            self.agent = DQNAgent(
                input_dims=self.input_dims,
                n_actions=self.n_actions,
                mem_size=self.mem_size,
                batch_size=self.batch_size,
                eps_min=self.eps_min,
                eps_max=self.eps_max,
                max_steps=self.max_steps,
            )
            self.agent.model = MockConvDQN()
            self.agent.target = MockConvDQN()

    def test_initialization(self):
        """Test initialization of agent attributes."""
        self.assertEqual(self.agent.input_dims, self.input_dims)
        self.assertEqual(self.agent.n_actions, self.n_actions)
        self.assertEqual(self.agent.mem_size, self.mem_size)
        self.assertEqual(self.agent.batch_size, self.batch_size)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertEqual(self.agent.lr, 0.001)

    def test_replace_target(self):
        """Test target network replacement."""
        self.agent.steps = self.agent.replace_target - 1
        self.agent.replace_target()
        self.agent.target.load_state_dict.assert_not_called()

        self.agent.steps += 1
        self.agent.replace_target()
        self.agent.target.load_state_dict.assert_called_once_with(
            self.agent.model.state_dict()
        )

    def test_choose_action(self):
        """Test action selection."""
        with patch("numpy.random.rand") as mock_rand:
            mock_rand.return_value = 0.05  # Force exploration
            action = self.agent.choose_action(np.random.randn(*self.input_dims))
            self.assertTrue(action in range(self.n_actions))

            mock_rand.return_value = 0.95  # Force exploitation
            self.agent.model.return_value = torch.tensor([[1.0, 0.5, 0.2, 0.1]])
            action = self.agent.choose_action(np.random.randn(*self.input_dims))
            self.assertEqual(action, 0)  # Max Q value index

    def test_learn(self):
        """Test learning process (simplified)."""
        # Setup necessary conditions to pass the buffer size check
        self.agent.buffer.size = self.agent.batch_size
        self.agent.buffer.sample = MagicMock(
            return_value=(
                np.random.randn(self.batch_size, *self.input_dims),  # states
                np.random.randint(0, self.n_actions, self.batch_size),  # actions
                np.random.randn(self.batch_size),  # rewards
                np.random.randn(self.batch_size, *self.input_dims),  # next_states
                np.random.randint(0, 2, self.batch_size, dtype=bool),  # dones
            )
        )

        self.agent.learn()
        # Test that optimizer's methods are called correctly
        self.assertTrue(self.agent.model.optimizer.zero_grad.called)
        self.assertTrue(self.agent.model.optimizer.step.called)


if __name__ == "__main__":
    unittest.main()
