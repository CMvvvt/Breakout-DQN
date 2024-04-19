import torch
import numpy as np
from collections import namedtuple
from utils import reshape_CHW

# Batch namedtuple, i.e. a class which contains the given attributes
Batch = namedtuple("Batch", ("states", "actions", "rewards", "next_states", "dones"))


class ReplayMemory:
    def __init__(self, max_size, state_size):
        """Replay memory implemented as a circular buffer.

        Experiences will be removed in a FIFO manner after reaching maximum
        buffer size.

        Args:
            - max_size: Maximum size of the buffer
            - state_size: Size of the state-space features for the environment
        """
        self.max_size = max_size
        self.state_size = state_size
        self.device = "mps"

        # Preallocating all the required memory, for speed concerns
        self.states = torch.zeros((max_size, *state_size), dtype=torch.uint8)
        self.actions = torch.zeros((max_size, 1), dtype=torch.long)
        self.rewards = torch.zeros((max_size, 1), dtype=torch.int8)
        self.dones = torch.zeros((max_size, 1), dtype=torch.bool)

        # Pointer to the current location in the circular buffer
        self.idx = 0
        # Indicates number of transitions currently stored in the buffer
        self.size = 0

    def add(self, state, action, reward, done):
        """Add a transition to the buffer.

        :param state: 1-D np.ndarray of state-features
        :param action: Integer action
        :param reward: Float reward
        :param next_state: 1-D np.ndarray of state-features
        :param done: Boolean value indicating the end of an episode
        """

        # YOUR CODE HERE: Store the input values into the appropriate
        # attributes, using the current buffer position `self.idx`

        # print("shape if state when inserting should be 1,84,84:", state.shape)
        self.states[self.idx] = state
        self.actions[self.idx, 0] = action
        self.rewards[self.idx, 0] = reward
        self.dones[self.idx, 0] = done

        # DO NOT EDIT
        # Circulate the pointer to the next position
        self.idx = (self.idx + 1) % self.max_size
        # Update the current buffer size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size) -> Batch:
        """Sample a batch of experiences/transitions.

        If the buffer contains less that `batch_size` transitions, sample all
        of them.

        :param batch_size: Number of transitions to sample
        :rtype: Batch
        """

        # YOUR CODE HERE: Randomly sample an appropriate number of
        # transitions *without replacement*. If the buffer contains less than
        # `batch_size` transitions, return all of them. The return type must
        # be a `Batch`.

        # sample_indices = np.random.choice(self.size, current_size, replace=False)
        sample_indices = torch.randint(0, high=self.size, size=(batch_size,))

        # print("indexes:")
        # print(sample_indices)

        batch = Batch(
            self.states[sample_indices, :4].to(self.device),
            self.actions[sample_indices].to(self.device),
            self.rewards[sample_indices].to(self.device),
            self.states[sample_indices, 1:].to(self.device).float(),
            self.dones[sample_indices].to(self.device).float(),
        )

        # print("BS shape:", self.states[sample_indices, :4].shape)
        # [32, 4, 84, 84]
        return batch

    def populate(self, env, num_steps):
        """Populate this replay memory with `num_steps` from the random policy.

        :param env: Gymnasium environment
        :param num_steps: Number of steps to populate the replay memory
        """

        # YOUR CODE HERE: Run a random policy for `num_steps` time-steps and
        # populate the replay memory with the resulting transitions.
        # Hint: Use the self.add() method.

        state = env.reset()
        state = reshape_CHW(state)
        # print("state's shape in populate:", state.shape)
        times = 0
        for _ in range(num_steps):
            times += 1
            if times % 1000 == 0:
                print(times)
            # random policy
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            next_state = reshape_CHW(next_state)
            self.add(state, action, reward, done)

            # update state
            if not done:
                state = next_state
            else:
                state = env.reset()
                state = reshape_CHW(state)
