import numpy as np
import pdb

import argparse
import copy
import h5py
import math
import os
import pickle
import random

from scipy.stats import norm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models import Policy, Posterior, Reward, Value
from grid_world import State, Action, TransitionFunction
from grid_world import RewardFunction, RewardFunction_SR2, GridWorldReward
from grid_world import ActionBasedGridWorldReward
from grid_world import create_obstacles, obstacle_movement, sample_start
from load_expert_traj import Expert, ExpertHDF5
from utils.replay_memory import Memory
from utils.running_state import ZFilter
from utils.torch_utils import clip_grads

from vae import VAE, VAETrain
from utils.logger import Logger, TensorboardXLogger

class BaseGAIL(object):
    def __init__(self,
                 args,
                 logger,
                 state_size=2,
                 action_size=4,
                 context_size=1,
                 num_goals=4,
                 history_size=1,
                 dtype=torch.FloatTensor):
        self.args = args
        self.logger = logger

        self.state_size = state_size
        self.action_size = action_size
        self.history_size = history_size
        self.context_size = context_size
        self.num_goals = num_goals
        self.dtype = dtype
        self.train_step_count, self.gail_step_count = 0, 0
        self.env_type = args.env_type

        self.policy_net = None
        self.old_policy_net = None
        self.value_net = None
        self.reward_net = None

        self.opt_policy, self.opt_reward, self.opt_value = None, None, None

        self.transition_func, self.true_reward = None, None
        self.expert = None
        self.obstacles, self.set_diff = None, None

    def create_environment(self):
        self.width, self.height = 21, 21
        self.transition_func = TransitionFunction(self.width,
                                                  self.height,
                                                  obstacle_movement)
    def set_expert(self, expert):
        assert self.expert is None, "Trying to set non-None expert"
        self.expert = expert
        assert expert.obstacles is not None, "Obstacles cannot be None"
        self.obstacles = expert.obstacles
        assert expert.set_diff is not None, "set_diff in h5 file cannot be None"
        self.set_diff = expert.set_diff

        # TODO: Hardcoded for now remove this, load it from the expert trajectory
        # file. Also, since the final state is not in expert trajectory we append
        # very next states as goal as well. Else reward is sparse.

        if self.args.flag_true_reward == 'grid_reward':
            self.true_reward = GridWorldReward(self.width,
                                               self.height,
                                               None,  # No default goals
                                               self.obstacles)
        elif self.args.flag_true_reward == 'action_reward':
            self.true_reward = ActionBasedGridWorldReward(
                    self.width, self.height, None, self.obstacles)


    def select_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0).type(self.dtype)
        action, _, _ = self.policy_net(Variable(state))
        return action

    def get_state_features(self, state_obj, use_state_features):
        '''Get state features.

        state_obj: State object.
        '''
        if use_state_features:
            feat = np.array(state_obj.get_features(), dtype=np.float32)
        else:
            feat = np.array(state_obj.coordinates, dtype=np.float32)
        return feat

    def sample_start_state(self):
        '''Randomly sample start state.'''
        start_loc = sample_start(self.set_diff)
        return State(start_loc, self.obstacles)

    def checkpoint_data_to_save(self):
        raise ValueError("Subclass should override.")

    def load_checkpoint_data(self, checkpoint_path):
        raise ValueError("Subclass should override.")

    def model_checkpoint_filepath(self, epoch):
        checkpoint_dir = os.path.join(self.args.results_dir, 'checkpoint')
        return os.path.join(checkpoint_dir, 'cp_{}.pth'.format(epoch))

    def expand_states_numpy(self, states, history_size):
        if self.history_size == 1:
            return states

        N, C = states.shape
        expanded_states = -1*np.ones((N, C*history_size), dtype=np.float32)
        for i in range(N):
            # Append states to the right
            expanded_states[i, -C:] = states[i,:]
            # Copy C:end state values from i-1 to 0:End-C in i
            if i > 0:
                expanded_states[i, :-C] = expanded_states[i-1, C:]

        return expanded_states
