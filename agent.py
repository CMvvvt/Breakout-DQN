import numpy as np
from ConvDQN import ConvDQN
from replay_memory import ReplayMemory
from utils import ExponentialSchedule, reshape_CHW
import torch
import torch.nn.functional as F
import random
from atari_wrappers import wrap_deepmind
from collections import deque


class DQNAgent:

    def __init__(
        self,
        env,
        input_dims,
        n_actions,
        mem_size,
        replay_start_size,
        batch_size,
        eps_min,
        eps_max,
        exploration_steps,
        gamma=0.99,
        lr=0.00025,
    ):
        self.env = env
        self.n_actions = n_actions
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.exploration = ExponentialSchedule(eps_max, eps_min, exploration_steps)
        self.gamma = gamma

        # Init replay buffer, DQN model and DQN target
        self.buffer = ReplayMemory(mem_size, input_dims)
        self.buffer.populate(env, replay_start_size)

        self.model = ConvDQN(input_dims, n_actions, lr)
        self.target = ConvDQN(input_dims, n_actions, lr)
        self.model.apply(self.model.init_weights)
        self.target.load_state_dict(self.model.state_dict())

    def insert_buffer(self, state, action, reward, done):
        self.buffer.add(state, action, reward, done)

    def sample_batch(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        return states, actions, rewards, next_states, dones

    def choose_action(self, observation, steps, evaluate=False):
        """
        Choose action from action space with epsilon-greedy algorithm
        """
        eps = self.exploration.value(steps)
        if np.random.rand() < eps:
            action = np.random.choice(self.n_actions)
        else:
            with torch.no_grad():
                action = self.model(observation).max(1)[1].cpu().view(1, 1)
        return action

    def learn(self):
        """
        Train the model
        """
        states, actions, rewards, next_states, dones = self.sample_batch()

        qvals = self.model(states).gather(1, actions)
        target_qvals = self.target(next_states).max(1)[0].detach()

        # Q*
        expected_qvales = (
            target_qvals * self.gamma * (1.0 - dones[:, 0]) + rewards[:, 0]
        )

        # Loss
        loss = F.smooth_l1_loss(qvals, expected_qvales.unsqueeze(1))

        # Optimization
        self.model.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            # Limiting the range of the gradients within [-1,1]
            param.grad.data.clamp_(-1, 1)
        self.model.optimizer.step()

        return loss

    def replace_target_model(self):
        self.target.load_state_dict(self.model.state_dict())

    def save_agent(self, name, step):
        self.model.save_model("training", name, step)

    def load_agent(self, name, step):
        self.model.load_model("training", name, step)

    def evaluate(self, env, step, episodes=10):
        env = wrap_deepmind(env)
        episode_rewards = []
        queue = deque(maxlen=5)
        frames = []
        plot_video = True
        for i in range(episodes):
            env.reset()
            if plot_video:
                frames.append(env.frame)
            episode_reward = 0

            for _ in range(10):
                next_obs, _, done, _ = env.step(0)
                queue.append(reshape_CHW(next_obs))

            while not done:
                state = torch.cat(list(queue))[1:].unsqueeze(0)
                action = self.choose_action(state, 999999, True)
                next_obs, reward, done, info = env.step(action)
                if plot_video:
                    frames.append(env.frame)
                queue.append(reshape_CHW(next_obs))
                episode_reward += reward

            episode_rewards.append(episode_reward)
            plot_video = False

        frames = np.stack(frames, 0)
        avg_reward = float(sum(episode_rewards)) / float(episodes)

        return frames, avg_reward
