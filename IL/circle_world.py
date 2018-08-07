import random
import numpy as np
from itertools import product
import math

import pdb

def sample_start(set_diff):
  return random.choice(set_diff)

class State(object):
    def __init__(self, coordinates, feat_type='view'):
        # coordinates - tuple, list_of_obstacles - list of tuples
        assert(len(coordinates) == 2)
        self.coordinates = coordinates

        self.state = self.set_features(coordinates,
                                       feat_type=feat_type)
        self.feat_type = feat_type

    def get_features(self):
        assert self.state is not None, 'Feature not set'
        return self.state

    def feature_size(self, feat_type):
        if feat_type == 'view':
            return 2
        else:
            raise ValueError("Incorrect feat_type: {}".format(feat_type))

    def set_features(self, coordinates, feat_type='view'):
        state = np.zeros(self.feature_size(feat_type))
        state[0], state[1] = coordinates
        if feat_type == 'view':
            pass
        else:
            raise ValueError('Incorrect feature type {}'.format(feat_type))
        return state

class StateVector(State):
    def __init__(self, coordinates_arr, feat_type='view'):
        super(StateVector, self).__init__((1, 1),  # Fake coordinates not used
                                          feat_type=feat_type)
        self.coordinates_arr = coordinates_arr
        self.coordinates = coordinates_arr
        self.state_arr = self.set_features_array(coordinates_arr,
                                                 feat_type=feat_type)


    def set_features_array(self, coordinates_arr, feat_type='view'):
        batch_size = coordinates_arr.shape[0]
        state_arr = np.zeros((batch_size,
                              self.feature_size(feat_type)))
        for b in range(batch_size):
            state_arr[b, :] = self.set_features(
                    coordinates_arr[b, :],
                    feat_type=feat_type)
        return state_arr

    def get_features(self):
        assert self.state_arr is not None, 'Feature not set'
        return self.state_arr

class Action(object):
    def __init__(self, delta):
        self.delta = delta

class ActionVector(Action):
    def __init__(self, delta_arr):
        # Pass in a dummy action
        super(ActionVector, self).__init__(delta_arr[0])
        self.delta_arr = delta_arr

class TransitionFunction():
    def __init__(self):
        '''Transition function for the grid world.'''
        pass

    def __call__(self, state, action, batch_radius, time):
        if type(state) is StateVector:
            assert state.coordinates_arr.shape[0] == action.delta_arr.shape[0], \
                    "(State, Action) batch sizes do not match"
            batch_size = state.coordinates_arr.shape[0]
            new_coord_arr = np.zeros(state.coordinates.shape)
            for b in range(batch_size):
                state_coord = state.coordinates_arr[b, :]
                radius = batch_radius[b]
                if type(state_coord) is not type([]):
                    state_coord = state_coord.tolist()
                action_delta = action.delta_arr[b]
                new_coord = self.next_state(state_coord, action_delta, radius, time)
                new_coord_arr[b, :] = new_coord

            new_state = StateVector(new_coord_arr, feat_type=state.feat_type)
        elif type(state) is State:
            new_coord = self.next_state(state.coordinates, action.delta, radius, time)
            new_state = State(new_coord, feat_type=state.feat_type)
            radius = batch_radius[0]
        else:
            raise ValueError('Incorrect state type: {}'.format(type(state)))

        return new_state

    def next_state(self, state_coord, action_delta, radius, time):
        radius_idx = time // 120
        assert radius_idx < 3, "Invalid time input"
        radius = radius[radius_idx]
        w = (2 * math.pi) / 120
        dist = w * radius
        new_state = (state_coord[0] + action_delta[0] * dist,
                     state_coord[1] + action_delta[1] * dist)   
        return new_state


class RewardFunction():
  def __init__(self, penalty, reward):
    # penalty - number (float), reward - number (float)
    self.terminal = False
    self.penalty = penalty
    self.reward = reward
    self.t = 0 # timer

  def __call__(self, state, action, c):
    self.t += 1
    if action.delta != np.argmax(c):
        return self.penalty
    else:
        return self.reward

  def reset(self, goal_1_func=None, goal_2_func=None):
    self.terminal = False
    self.t = 0

class RewardFunction_SR2():
  def __init__(self, penalty, reward, grid_width=12):
    # penalty - number (float), reward - number (float)
    self.terminal = False
    self.penalty = penalty
    self.reward = reward
    self.grid_width = grid_width
    self.action_deltas = [[3,3,1,1,2,2,0,0], [3,1,1,1,2,0,0,0]]
    self.t = 0 # timer

  def __call__(self, state, action, c):
    self.t += 1
    if state.state[0] >= self.grid_width/2:
      if action.delta != self.action_deltas[0][self.t-1]:
        return self.penalty
      else:
        return self.reward
    else:
      if action.delta != self.action_deltas[1][self.t-1]:
        return self.penalty
      else:
        return self.reward

  def reset(self, goal_1_func=None, goal_2_func=None):
    self.terminal = False
    self.t = 0

class GridWorldReward(object):
  def __init__(self, width, height, goals, obstacles):
    self.width = width
    self.height = height
    self.goals = goals
    self.obstacles = [(x[0], x[1]) for x in obstacles.tolist()]
    self.goal_reward = 100
    self.obstacle_reward = -100
    self.other_reward = -1

  def reward_at_location(self, pos, goals=None):
    pos = [int(pos[0]), int(pos[1])]
    goals = self.goals if goals is None else goals
    if pos in goals:
      return self.goal_reward
    if pos in self.obstacles:
      return self.obstacle_reward
    return self.other_reward

class ActionBasedGridWorldReward(GridWorldReward):
  def __init__(self, width, height, goals, obstacles):
    super(ActionBasedGridWorldReward, self).__init__(width, height,
                                                     goals, obstacles)
    self.corret_action_reward = 1.0
    self.incorrect_action_reward = -1.0

  def reward_at_location(self, expert_action, gen_action):
    if expert_action == gen_action:
      return self.corret_action_reward
    else:
      return self.incorrect_action_reward
