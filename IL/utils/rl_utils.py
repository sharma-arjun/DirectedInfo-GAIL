import numpy as np

import math
import torch
from torch.autograd import Variable
import pdb, ipdb

def epsilon_greedy_linear_decay(action_vector,
                                num_episodes,
                                ep_idx,
                                num_actions,
                                low=0.1,
                                high=0.9):
    if ep_idx <= num_episodes:
        eps = ((low-high)/num_episodes)*ep_idx + high
    else:
        eps = low

    if np.random.uniform() > eps:
        return np.argmax(action_vector)
    else:
        return np.random.randint(low=0, high=num_actions)

def epsilon_greedy(action_vector, eps=0.1):
    if np.random.uniform() > eps:
        return np.argmax(action_vector)
    else:
        return np.random.randint(low=0, high=num_actions)

def greedy(action_vector):
    return np.argmax(action_vector)

def oned_to_onehot(action_delta, num_actions):
    action_onehot = np.zeros(num_actions,)
    action_onehot[int(action_delta)] = 1.0
    return action_onehot

def get_advantage_for_rewards(rewards,
                              masks,
                              gamma,
                              tau,
                              values=None,
                              dtype=torch.FloatTensor):

    returns = torch.Tensor(rewards.size(0), 1).type(dtype)
    deltas = torch.Tensor(rewards.size(0), 1).type(dtype)
    advantages = torch.Tensor(rewards.size(0), 1).type(dtype)

    # compute advantages
    prev_return, prev_value, prev_advantage = 0, 0, 0

    for i in reversed(range(rewards.size(0))):
        # This return is a monte-carlo estimate.
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        if values is not None:
            deltas[i] = rewards[i] + gamma * prev_value * masks[i] \
                    - values.data[i]
            advantages[i] = deltas[i] + \
                    gamma * tau * prev_advantage * masks[i]
            prev_value = values.data[i, 0]
        else:
            advantages[i] = returns[i]

        prev_return = returns[i, 0]
        prev_advantage = advantages[i, 0]

    if values is not None:
        returns = advantages + values.data
    return returns, advantages
