import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ipdb
import os
import gymnasium as gym
from random import sample, random
from tqdm import tqdm
import wandb
from collections import deque
import numpy as np
import time
from ConvDQN import ConvDQN
from utils import FrameStackingAndResizingEnv, ExponentialSchedule, Environment
from agent import DQNAgent
from tqdm import tqdm


render = False
to_save = False
to_load = False

print("Render: ", render)
print("Save: ", to_save)
print("Load: ", to_load)

# Parameters:
MEM_SIZE = 1_000_000
REPLAY_START_SIZE = 100_000
BATCH_SIZE = 32
EPS_MIN = 0.1
EPS_MAX = 1
EXPLORATION_STEPS = 1_000_000
GAMMA = 0.99
LEARNING_RATE = 0.001
REPLACE_TARGET_STEPS = 10_000
SAVE_MODEL_STEPS = 100_000
RENDER_STEPS = 10000
TRAINING_NAME = "DQN-Breakout-video-log"
ENV_NAME = "Breakout-v0"


def main():
    wandb.init(
        project="dqn-tutorial",
        name=f"{TRAINING_NAME} lr_{LEARNING_RATE} | eps_{EPS_MAX}-{EPS_MIN}-1m",
    )
    env = gym.make(ENV_NAME)
    env = FrameStackingAndResizingEnv(env, h=84, w=84, num_stack=4)
    # env = Environment(ENV_NAME)

    best_score = -np.inf
    agent = DQNAgent(
        env=env,
        input_dims=env.observation_space.shape,
        n_actions=env.action_space.n,
        mem_size=MEM_SIZE,
        replay_start_size=REPLAY_START_SIZE,
        batch_size=BATCH_SIZE,
        eps_min=EPS_MIN,
        eps_max=EPS_MAX,
        exploration_steps=EXPLORATION_STEPS,
        gamma=GAMMA,
        lr=LEARNING_RATE,
        replace_target_steps=REPLACE_TARGET_STEPS,
        save_model_steps=SAVE_MODEL_STEPS,
        training_name=TRAINING_NAME,
    )

    if os.path.isfile("models/local.pt") and to_load:
        agent.load_agent()

    epochs = 10000
    episode_rewards = []
    last_log_video_step = 0

    tq = tqdm()
    last_step = 0
    for i in range(epochs):
        done = False
        # observation.shape = [4, 84, 84]
        observation = env.reset()
        total_reward = 0

        while not done:
            if render:
                env.render()
            tq.update(1)
            action = agent.choose_action(observation)

            # next_observation.shape = [4, 84, 84]
            next_observation, reward, done, info = env.step(action)

            total_reward += reward

            agent.insert_buffer(observation, action, reward, next_observation, done)
            loss = agent.learn()

            observation = next_observation

        episode_rewards.append(total_reward)
        print(
            f"reward of episode {i}: {total_reward}, steps of episode: {agent.steps - last_step}"
        )
        last_step = agent.steps

        wandb.log(
            {
                "loss": loss.detach().item(),
                "eps": agent.exploration.value(agent.steps),
                "episode_reward": total_reward,
            },
            step=agent.steps,
        )

        # plot video
        if (
            agent.steps > 0
            and last_log_video_step == 0
            or (agent.steps > last_log_video_step + 100000)
        ):
            rew, frames = agent.run_episode()
            #  frames.shape == (T, H, W, C)
            wandb.log(
                {
                    "reward": rew,
                    "video": wandb.Video(
                        frames.transpose(0, 3, 1, 2), str(rew), fps=25
                    ),
                }
            )

            last_log_video_step = agent.steps

    env.close()


if __name__ == "__main__":
    main()
