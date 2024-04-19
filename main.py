import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ipdb
import os
import gym
from random import sample, random
from tqdm import tqdm
import wandb
from collections import deque
import numpy as np
import time
from ConvDQN import ConvDQN
from utils import FrameStackingAndResizingEnv, ExponentialSchedule, reshape_CHW
from agent import DQNAgent
from tqdm import tqdm
from atari_wrappers import wrap_deepmind, make_atari


to_load = False

# Parameters:
MEM_SIZE = 1_000_000
REPLAY_START_SIZE = 50_000
BATCH_SIZE = 32
EPS_MIN = 0.1
EPS_MAX = 1
EXPLORATION_STEPS = 1_000_000
GAMMA = 0.99
LEARNING_RATE = 0.00025
TARGET_UPDATE = 10_000
RENDER_STEPS = 100_000
MAX_STEPS = 20000000
SAVE_MODEL_STEPS = 200_000
EVALUATION_STEPS = 100_000

ENV_NAME = "SpaceInvaders"
# ENV_NAME = "Breakout"
METHOD = "DQN"


def main():
    wandb.init(
        project="dqn-tutorial",
        name=f"{ENV_NAME}-{METHOD} lr_{LEARNING_RATE} | eps_{EPS_MAX}-{EPS_MIN}-1m",
    )
    env_raw = make_atari(f"{ENV_NAME}NoFrameskip-v4")
    env = wrap_deepmind(
        env_raw, frame_stack=False, episode_life=True, clip_rewards=True
    )
    C, H, W = reshape_CHW(env.reset()).shape

    agent = DQNAgent(
        env=env,
        input_dims=(5, H, W),
        n_actions=env.action_space.n,
        mem_size=MEM_SIZE,
        replay_start_size=REPLAY_START_SIZE,
        batch_size=BATCH_SIZE,
        eps_min=EPS_MIN,
        eps_max=EPS_MAX,
        exploration_steps=EXPLORATION_STEPS,
        gamma=GAMMA,
        lr=LEARNING_RATE,
    )

    if os.path.isfile("models/local.pt") and to_load:
        agent.load_agent()

    queue = deque(maxlen=5)
    tq = tqdm()
    done = True
    progress = tqdm(range(MAX_STEPS), total=MAX_STEPS, ncols=50, leave=False, unit="b")

    for step in progress:
        if done:
            env.reset()
            env.step(1)  # Fire the ball
            for i in range(10):  # fill the queue with noop action
                obs, _, _, _ = env.step(0)
                obs = reshape_CHW(obs)
                queue.append(obs)

        train = agent.buffer.size > 50000

        state = torch.cat(list(queue))[1:].unsqueeze(0)
        # state(1,4,84,84)

        action = agent.choose_action(state, step)

        obs, reward, done, info = env.step(action)
        obs = reshape_CHW(obs)

        # Insert to buffer
        queue.append(obs)
        agent.insert_buffer(torch.cat(list(queue)).unsqueeze(0), action, reward, done)

        if step % 4 == 0:
            loss = agent.learn()
            wandb.log({"loss": loss}, step)

        if step % SAVE_MODEL_STEPS == 0:
            agent.save_agent("DQN-Breakout", step)

        if step % TARGET_UPDATE == 0:
            agent.replace_target_model()

        if step % EVALUATION_STEPS == 0:
            frames, avg_reward = agent.evaluate(env_raw, step, episodes=15)
            wandb.log(
                {
                    "evaluate_video": wandb.Video(frames.transpose(0, 3, 1, 2), fps=25),
                    "avg_reward": avg_reward,
                },
                step,
            )

    env.close()


if __name__ == "__main__":
    main()
