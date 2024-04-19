"""utils.py"""

import math
import numpy as np
import cv2
from gymnasium.spaces import Box
import gymnasium as gym
import torch

from random import randint


def reshape_CHW(obs):
    """
    Change to shape tp (channel, height, weight)
    """
    obs = torch.from_numpy(obs)
    h, w = obs.shape[:2]
    return obs.view(1, h, w)


class ExponentialSchedule:
    """
    Copied from the implementation of my assignment 8
    """

    def __init__(self, value_from, value_to, num_steps):
        """Exponential schedule from `value_from` to `value_to` in `num_steps` steps.

        $value(t) = a exp (b t)$

        :param value_from: Initial value
        :param value_to: Final value
        :param num_steps: Number of steps for the exponential schedule
        """
        self.value_from = value_from
        self.value_to = value_to
        self.num_steps = num_steps

        # YOUR CODE HERE: Determine the `a` and `b` parameters such that the schedule is correct
        self.a = value_from
        self.b = math.log(value_to / value_from) / (num_steps - 1)

    def value(self, step) -> float:
        """Return exponentially interpolated value between `value_from` and `value_to`interpolated value between.

        Returns {
            `value_from`, if step == 0 or less
            `value_to`, if step == num_steps - 1 or more
            the exponential interpolation between `value_from` and `value_to`, if 0 <= steps < num_steps
        }

        :param step: The step at which to compute the interpolation
        :rtype: Float. The interpolated value
        """

        # YOUR CODE HERE: Implement the schedule rule as described in the docstring,
        # using attributes `self.a` and `self.b`.
        if step <= 0:
            return self.value_from
        elif step >= self.num_steps - 1:
            return self.value_to
        else:
            return self.a * math.exp(self.b * step)


class FrameStackingAndResizingEnv:
    def __init__(self, env, h, w, num_stack=4):
        self.env = env
        self.n = num_stack
        self.w = w
        self.h = h

        self.buffer = np.zeros((num_stack, h, w), "uint8")
        self.frame = None

    def _preprocess_frame(self, frame):
        image = cv2.resize(frame, (self.w, self.h))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    @property
    def observation_space(self):
        return Box(low=0, high=255, shape=(self.n, self.h, self.w), dtype=np.uint8)

    @property
    def action_space(self):
        return self.env.action_space

    def step(self, action):
        im, reward, done, info, _ = self.env.step(action)
        self.frame = im.copy()
        im = self._preprocess_frame(im)
        self.buffer[1 : self.n, :, :] = self.buffer[0 : self.n - 1, :, :]
        self.buffer[0, :, :] = im

        return (self.buffer.copy(), reward, done, info)

    def reset(self):
        im, _ = self.env.reset()
        self.frame = im.copy()
        im = self._preprocess_frame(im)
        self.buffer = np.stack([im] * self.n, 0)
        return self.buffer.copy()

    def render(self, mode):
        if mode == "rgb_array":
            return self.frame
        super(FrameStackingAndResizingEnv, self).render(mode)


# class Environment:
#     def __init__(self, env_name):
#         self.env = make_wrap_atari(env_name)

#         # self.action_space = self.env.action_space
#         # self.observation_space = self.env.observation_space

#     def seed(self, seed):
#         """
#         Control the randomness of the environment
#         """
#         self.env.seed(seed)

#     def reset(self):
#         """
#         When running dqn:
#             observation: np.array
#                 stack 4 last frames, shape: (84, 84, 4)

#         When running pg:
#             observation: np.array
#                 current RGB screen of game, shape: (210, 160, 3)
#         """
#         observation = self.env.reset()

#         return np.array(observation)

#     def step(self, action):
#         """
#         When running dqn:
#             observation: np.array
#                 stack 4 last preprocessed frames, shape: (84, 84, 4)
#             reward: int
#                 wrapper clips the reward to {-1, 0, 1} by its sign
#                 we don't clip the reward when testing
#             done: bool
#                 whether reach the end of the episode?

#         When running pg:
#             observation: np.array
#                 current RGB screen of game, shape: (210, 160, 3)
#             reward: int
#                 if opponent wins, reward = +1 else -1
#             done: bool
#                 whether reach the end of the episode?
#         """
#         if not self.env.action_space.contains(action):
#             raise ValueError("Ivalid action!!")

#         observation, reward, done, info = self.env.step(action)

#         return np.array(observation), reward, done, info

#     @property
#     def action_space(self):
#         print("self.env.action_space, should be 4:", self.env.action_space)
#         return self.env.action_space

#     @property
#     def observation_space(self):
#         print(
#             "self.env.observation_space, should be (4, 84, 84)",
#             self.env.observation_space,
#         )
#         return self.env.observation_space


if __name__ == "__main__":
    env = gym.make("Breakout-v0", render_mode="rbg_array")
    env = FrameStackingAndResizingEnv(env, 480, 640)

    # print(env.observation_space.shape)
    # print(env.action_space)

    im = env.reset()
    idx = 0
    ims = []
    for i in range(im.shape[-1]):
        ims.append(im[:, :, i])
    cv2.imwrite(f"/tmp/{idx}.jpg", np.hstack(ims))

    env.step(1)

    for _ in range(10):
        idx += 1
        im, _, _, _ = env.step(randint(0, 3))
        for i in range(im.shape[-1]):
            ims.append(im[:, :, i])

        cv2.imwrite(f"tmp/{idx}.jpg", np.hstack(ims))
        ims = []
