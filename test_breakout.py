import gymnasium as gym
import warnings


warnings.filterwarnings("ignore")

# gym.register(id="Breakout-v0")
env = gym.make("Breakout-v0")

observation, _ = env.reset()
observation_, reward, done, info, _ = env.step(1)
print("env.observation_space.shape", env.observation_space.shape)
print("action_space", env.action_space)
print("observation", observation.shape)
print("observation_", observation_.shape)
