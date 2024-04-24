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


class FrameStackingEnv:
    def __init__(self, env, h=84, w=84, stacks=4):
        self.env = env
        self.n = stacks
        self.w = w
        self.h = h

        self.buffer = np.zeros((stacks, h, w), "uint8")
        self.frame = None

    def _preprocess_frame(self, frame):
        image = cv2.resize(frame, (self.w, self.h))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def step(self, action):
        obs, reward, done, info, _ = self.env.step(action)
        self.frame = obs.copy()
        obs = self._preprocess_frame(obs)
        self.buffer[1 : self.n, :, :] = self.buffer[0 : self.n - 1, :, :]
        self.buffer[0, :, :] = obs

        return (self.buffer.copy(), reward, done, info)

    def reset(self):
        obs, _ = self.env.reset()
        self.frame = obs.copy()
        obs = self._preprocess_frame(obs)
        self.buffer = np.stack([obs] * self.n, 0)
        return self.buffer.copy()
