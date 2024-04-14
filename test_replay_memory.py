import unittest
import torch
from collections import namedtuple
from unittest.mock import Mock
from replay_memory import ReplayMemory, Batch
import numpy as np


class TestReplayMemory(unittest.TestCase):
    def setUp(self):
        self.max_size = 100
        self.state_size = 5  # Example state size
        self.replay_memory = ReplayMemory(self.max_size, self.state_size)

    def test_initialization(self):
        self.assertEqual(len(self.replay_memory.states), self.max_size)
        self.assertEqual(self.replay_memory.states.shape[1], self.state_size)
        self.assertEqual(self.replay_memory.actions.shape[0], self.max_size)
        self.assertEqual(self.replay_memory.size, 0)

    def test_add_and_size(self):
        state = np.random.randn(self.state_size)
        action = 1
        reward = 1.0
        next_state = np.random.randn(self.state_size)
        done = False

        for _ in range(150):  # Add more elements than capacity to test circular buffer
            self.replay_memory.add(state, action, reward, next_state, done)

        self.assertEqual(
            self.replay_memory.size, self.max_size
        )  # Size should not exceed max_size
        self.assertEqual(self.replay_memory.idx, 50)  # idx should wrap around

    def test_sample(self):
        num_samples = 10
        for _ in range(20):  # Add some elements
            state = np.random.randn(self.state_size)
            action = np.random.randint(0, 4)
            reward = np.random.random()
            next_state = np.random.randn(self.state_size)
            done = np.random.choice([True, False])
            self.replay_memory.add(state, action, reward, next_state, done)

        sampled = self.replay_memory.sample(num_samples)
        self.assertEqual(len(sampled.states), num_samples)
        self.assertIsInstance(sampled, Batch)

    def test_populate(self):
        env = Mock()
        env.reset = Mock(return_value=(np.zeros(self.state_size), None))
        env.step = Mock(return_value=(np.zeros(self.state_size), 1.0, False, {}, None))
        env.action_space.sample = Mock(return_value=1)

        self.replay_memory.populate(env, 50)
        self.assertEqual(self.replay_memory.size, 50)
        env.reset.assert_called()
        env.step.assert_called()
        env.action_space.sample.assert_called()


if __name__ == "__main__":
    unittest.main()
