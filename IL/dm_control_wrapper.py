import argparse
import os

import gym
import gym.spaces
import numpy as np
from dm_control import suite  # Must be imported after TensorFlow.


class DeepMindWrapper(object):
  """Wraps a DM Control environment into a Gym interface."""

  metadata = {'render.modes': ['rgb_array']}
  reward_range = (-np.inf, np.inf)

  def __init__(self, env, render_size=(64, 64), camera_id=0):
    self._env = env
    self._render_size = render_size
    self._camera_id = camera_id

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    components = {}
    for key, value in self._env.observation_spec().items():
      components[key] = gym.spaces.Box(-np.inf, np.inf, value.shape)
    return gym.spaces.Dict(components)

  @property
  def action_space(self):
    action_spec = self._env.action_spec()
    return gym.spaces.Box(action_spec.minimum, action_spec.maximum)

  def step(self, action):
    time_step = self._env.step(action)
    obs = dict(time_step.observation)
    reward = time_step.reward or 0
    done = time_step.last()
    info = {'discount': time_step.discount}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    return dict(time_step.observation)

  def render(self, mode='rgb_array', *args, **kwargs):
    if mode != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    del args  # Unused.
    del kwargs  # Unused.
    return self._env.physics.render(
        *self._render_size, camera_id=self._camera_id)


class SelectKeysWrapper(object):
  """Select observations from a dict space and concatenate them."""

  def __init__(self, env, keys):
    self._env = env
    self._keys = keys

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    components = self._env.observation_space.spaces
    components = [components[key] for key in self._keys]
    low = np.concatenate([component.low for component in components], 0)
    high = np.concatenate([component.high for component in components], 0)
    return gym.spaces.Box(low, high)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = self._select_keys(obs)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs = self._select_keys(obs)
    return obs

  def _select_keys(self, obs):
    return np.concatenate([obs[key] for key in self._keys], 0)


def create_env():
  env = suite.load('reacher', 'easy')
  env = DeepMindWrapper(env)
  env = SelectKeysWrapper(env, ['position', 'velocity', 'to_target'])
  return env


def reacher():
  locals().update(agents.scripts.configs.default())
  env = create_env
  max_length = 1000
  steps = 1e7  # 10M
  discount = 0.985
  update_every = 60
  return locals()


def main():
  env = suite.load('humanoid', 'walk')
  env = DeepMindWrapper(env)
  # env = SelectKeysWrapper(env, ['position', 'velocity', 'to_target'])
  env = SelectKeysWrapper(env, 
          ['torso_vertical', 'joint_angles', 'velocity', 'com_velocity', 'extremities'])

  step = 0
  video_frames = []
  action_spec = env._env.action_spec()
  time_step = env.reset()
  while step < 1000:
      frame = env.render()
      video_frames.append(frame)
      action =  np.random.uniform(action_spec.minimum,
                                  action_spec.maximum,
                                  size=action_spec.shape)
      state, reward, done, info = env.step(action)
      # print("reward: {}, done: {}, obs: {}".format(reward, done, state))
      step = step + 1

  import matplotlib.pyplot as plt
  for i in range(len(video_frames)):
      img = plt.imshow(video_frames[i])
      plt.pause(0.01)
      plt.draw()
  print(step)

if __name__ == '__main__':
  # parser = argparse.ArgumentParser()
  # parser.add_argument('--mode', choices=['train', 'render'], default='train')
  # parser.add_argument('--logdir', default='~/logdir/varagent')
  # parser.add_argument('--config')
  # args = parser.parse_args()
  main()
