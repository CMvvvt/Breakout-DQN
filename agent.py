import numpy as np
from ConvDQN import ConvDQN
from replay_memory import ReplayMemory
from utils import ExponentialSchedule
import torch
import torch.nn.functional as F


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
        replace_target_steps=10000,
        save_model_steps=100000,
        training_name="training",
    ):
        self.env = env
        self.n_actions = n_actions
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.exploration = ExponentialSchedule(eps_max, eps_min, exploration_steps)
        self.gamma = gamma
        self.replace_target_steps = replace_target_steps
        self.steps = 0
        self.save_model_steps = save_model_steps
        self.training_name = training_name

        # Init replay buffer, DQN model and DQN target
        self.buffer = ReplayMemory(mem_size, input_dims)
        self.buffer.populate(env, replay_start_size)

        self.model = ConvDQN(input_dims, n_actions, lr)
        self.target = ConvDQN(input_dims, n_actions, lr)

    def insert_buffer(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def sample_batch(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states = torch.tensor(states, dtype=torch.float32).to(self.model.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.model.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.model.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(
            self.model.device
        )
        dones = torch.tensor(dones, dtype=torch.bool).to(self.model.device)

        return states, actions, rewards, next_states, dones

    def choose_action(self, observation):
        """
        Choose action from action space with epsilon-greedy algorithm
        """
        eps = self.exploration.value(self.steps)
        if np.random.rand() < eps:
            action = np.random.choice(self.n_actions)
            # print("RANDOM SELECTION, action:", action)

        else:
            state = (
                torch.tensor(observation, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.model.device)
            )
            action = self.model(state).unsqueeze(0).max(-1)[-1].item()
        return action

    def replace_target_model(self):
        self.target.load_state_dict(self.model.state_dict())

    def learn(self):
        """
        Train the model
        """
        if self.buffer.size < self.batch_size:
            return

        # Update target model / Save model
        if self.steps % self.replace_target_steps == 0:
            self.replace_target_model()
        if self.steps % self.save_model_steps == 0:
            self.save_agent(self.training_name)

        # Sample batch from buffer
        states, actions, rewards, next_states, dones = self.sample_batch()

        # Compute the predicted/next Q-values and loss
        # Q_pred = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # next_pred = self.target(next_states).max(1)[0].detach()
        # Q_target = rewards + self.gamma * next_pred * (~dones)
        # loss = F.mse_loss(Q_pred, Q_target).to(self.model.device)

        # Compute Q-values for current states and actions with the DQN model.
        values = self.model(states).gather(1, actions.unsqueeze(-1))  # [32, 1]

        # Compute Q-values for next states with the target model.
        with torch.no_grad():
            next_q_values = self.target(next_states).max(1)[0].detach()
            target_values = (rewards + self.gamma * next_q_values * ~dones).unsqueeze(1)

        # DO NOT EDIT
        assert (
            values.shape == target_values.shape
        ), "Shapes of values tensor and target_values tensor do not match."
        assert values.requires_grad, "values tensor requires gradients"
        assert (
            not target_values.requires_grad
        ), "target_values tensor should not require gradients"

        loss = F.mse_loss(values, target_values)
        # Backpropogation
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        self.steps += 1

        return loss

    def save_agent(self, name):
        self.model.save_model("training", self.steps, name)
        # self.target.save_model("target", self.steps, name)

    def load_agent(self, steps, name):
        self.model.load_model("training", steps, name)
        # self.target.load_model("target", steps, name)

    def run_episode(self, max_steps=10000):
        frames = []
        obs = self.env.reset()
        frames.append(self.env.frame)

        idx = 0
        done = False
        reward = 0
        while not done and idx < max_steps:
            state = (
                torch.tensor(obs, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.model.device)
            )
            action = self.model(state).max(-1)[-1].item()
            # action = np.random.choice(self.n_actions)

            obs, r, done, _ = self.env.step(action)
            reward += r
            frames.append(self.env.frame)
            idx += 1

        return reward, np.stack(frames, 0)
