import argparse
import copy
import sys
import os
import pdb, ipdb
import pickle
import math
import random
import gym
from collections import namedtuple
from itertools import count, product

import numpy as np
import scipy.optimize
from scipy.stats import norm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable

from models import DiscretePosterior, Policy, Posterior, Reward, Value
from grid_world import State, Action, TransitionFunction
from grid_world import StateVector, ActionVector
from grid_world import RewardFunction, RewardFunction_SR2, GridWorldReward
from grid_world import ActionBasedGridWorldReward
from grid_world import create_obstacles, obstacle_movement, sample_start
from load_expert_traj import Expert, ExpertHDF5, SeparateRoomTrajExpert
from utils.replay_memory import Memory
from utils.running_state import ZFilter
from utils.torch_utils import clip_grads

from base_gail import BaseGAIL
from vae import VAE, VAETrain
from utils.logger import Logger, TensorboardXLogger
from utils.rl_utils import epsilon_greedy_linear_decay, epsilon_greedy
from utils.rl_utils import greedy, oned_to_onehot
from utils.rl_utils import get_advantage_for_rewards
from utils.torch_utils import get_weight_norm_for_network
from utils.torch_utils import normal_log_density

class CausalGAILMLP(BaseGAIL):
  def __init__(self,
               args,
               vae_train,
               logger,
               state_size=2,
               action_size=4,
               context_size=1,
               num_goals=4,
               history_size=1,
               dtype=torch.FloatTensor):
    super(CausalGAILMLP, self).__init__(args,
                                        logger,
                                        state_size=state_size,
                                        action_size=action_size,
                                        context_size=context_size,
                                        num_goals=num_goals,
                                        history_size=history_size,
                                        dtype=dtype)

    self.vae_train = vae_train
    policy1_state_size = state_size * history_size \
            if vae_train.vae_model.use_history_in_policy else state_size

    self.policy_net = Policy(
            state_size=policy1_state_size,
            action_size=vae_train.vae_model.policy.action_size,
            latent_size=vae_train.vae_model.policy.latent_size,
            output_size=vae_train.vae_model.policy.output_size,
            output_activation=None)

    self.old_policy_net = Policy(
            state_size=policy1_state_size,
            action_size=vae_train.vae_model.policy.action_size,
            latent_size=vae_train.vae_model.policy.latent_size,
            output_size=vae_train.vae_model.policy.output_size,
            output_activation=None)

    if args.use_value_net:
      # context_size contains num_goals
      self.value_net = Value(state_size * history_size + context_size ,
                             hidden_size=64)

    # Reward net is the discriminator network.
    self.reward_net = Reward(state_size * history_size,
                             action_size,
                             context_size,
                             hidden_size=64)

    if vae_train.args.use_discrete_vae:
        self.posterior_net = DiscretePosterior(
                state_size=vae_train.vae_model.posterior.state_size,
                action_size=vae_train.vae_model.posterior.action_size,
                latent_size=vae_train.vae_model.posterior.latent_size,
                hidden_size=vae_train.vae_model.posterior.hidden_size,
                output_size=vae_train.args.vae_context_size)
    else:
        self.posterior_net = Posterior(state_size * history_size,
                                       0,
                                       context_size,
                                       hidden_size=64)

    self.opt_policy = optim.Adam(self.policy_net.parameters(), lr=0.0003)
    self.opt_reward = optim.Adam(self.reward_net.parameters(), lr=0.0003)
    self.opt_posterior = optim.Adam(self.posterior_net.parameters(), lr=0.0003)
    if args.use_value_net:
      self.opt_value = optim.Adam(self.value_net.parameters(), lr=0.0003)

    # Create loss functions
    self.criterion = nn.BCELoss()
    #criterion_posterior = nn.NLLLoss()
    self.criterion_posterior = nn.MSELoss()

    self.transition_func, self.true_reward = None, None
    self.create_environment(args.env_type, args.env_name)
    self.expert = None
    self.obstacles, self.set_diff = None, None

  def create_environment(self, env_type, env_name=None):
    if 'grid' in env_type:
      self.transition_func = TransitionFunction(self.vae_train.width,
                                                self.vae_train.height,
                                                obstacle_movement)
    elif env_type == 'mujoco':
      assert(env_name is not None)
      self.env = gym.make(env_name)

  def select_action(self, inp):
      """Select action using policy net."""
      # TODO: Policy net could output discrete action or continuous action
      action_mean, _, _ = self.policy_net(inp)
      return action_mean

  def get_c_for_traj(self, state_arr, action_arr, c_arr):
    '''Get c[1:T] for given trajectory.'''
    batch_size, episode_len = state_arr.shape[0], state_arr.shape[1]
    history_size = self.history_size

    # Use the Q-network (RNN) to predict goal.
    pred_goal = None
    if self.vae_train.use_rnn_goal_predictor:
        pred_goal, _ = self.vae_train.predict_goal(
            state_arr, action_arr, c_arr, None, self.num_goals)

    if self.args.env_type == 'grid_room':
        true_goal_numpy = np.copy(c_arr)
    else:
        true_goal_numpy = np.zeros((c_arr.shape[0], self.num_goals))
        true_goal_numpy[np.arange(c_arr.shape[0]), c_arr[:, 0]] = 1
    true_goal = Variable(torch.from_numpy(true_goal_numpy).type(self.dtype))

    action_var = Variable(
            torch.from_numpy(action_arr).type(self.dtype))

    # Context output from the VAE encoder
    pred_c_arr = -1 * np.ones((batch_size,
                               episode_len + 1,
                               self.vae_train.vae_model.posterior_latent_size))

    if 'grid' in self.args.env_type:
        x_state_obj = StateVector(state_arr[:, 0, :], self.obstacles)
        x_feat = self.vae_train.get_state_features(
                x_state_obj,
                self.args.use_state_features)
    elif self.args.env_type == 'mujoco':
        x_feat = state_arr[:, 0, :]
        dummy_state = self.env.reset()
        self.env.env.set_state(np.concatenate(
            (np.array([0.0]), x_feat[0, :8]), axis=0), x_feat[0, 8:17])
        dummy_state = x_feat
    else:
        raise ValueError('Incorrect env type: {}'.format(self.args.env_type))

    # x is (N, F)
    x = x_feat

    # Add history to state
    if self.history_size > 1:
        x_hist = -1 * np.ones((x.shape[0], history_size, x.shape[1]),
                              dtype=np.float32)
        x_hist[:, history_size - 1, :] = x_feat
        x = self.get_history_features(x_hist,
                                      self.args.use_velocity_features)

    curr_state_arr = state_arr[:, 0, :]
    for t in range(episode_len):
        c = pred_c_arr[:, t, :]
        x_var = Variable(torch.from_numpy(
            x.reshape((batch_size, -1))).type(self.dtype))

        # Append 'c' at the end.
        if self.vae_train.use_rnn_goal_predictor:
            c_var = torch.cat([final_goal,
                    Variable(torch.from_numpy(c).type(self.dtype))], dim=1)
        else:
            c_var = Variable(torch.from_numpy(c).type(self.dtype))
            if len(true_goal.size()) == 2:
                c_var = torch.cat([true_goal, c_var], dim=1)
            elif len(true_goal.size()) == 3:
                c_var = torch.cat([true_goal[:, t, :], c_var], dim=1)
            else:
                raise ValueError("incorrect true goal size")

        c_next = self.vae_train.get_context_at_state(x_var, c_var)
        pred_c_arr[:, t+1, :] = c_next.data.cpu().numpy()

        if history_size > 1:
            x_hist[:, :(history_size-1), :] = x_hist[:, 1:, :]

        if 'grid' in self.args.env_type:
            if t < episode_len-1:
                next_state = StateVector(state_arr[:, t+1, :], self.obstacles)
            else:
                break

            if history_size > 1:
                x_hist[:, history_size-1] = self.vae_train.get_state_features(
                        next_state, self.args.use_state_features)
                x = self.get_history_features(
                        x_hist, self.args.use_velocity_features)
            else:
                x[:] = self.vae_train.get_state_features(
                        next_state, self.args.use_state_features)
                # Update current state
                curr_state_arr = np.array(
                        next_state.coordinates, dtype=np.float32)
        else:
            raise ValueError("Not implemented yet.")

    return pred_c_arr, pred_goal

  def checkpoint_data_to_save(self):
    return {
        'policy': self.policy_net,
        'posterior': self.posterior_net,
        'reward': self.reward_net
        }

  def load_checkpoint_data(self, checkpoint_path):
    assert os.path.exists(checkpoint_path), \
        'Checkpoint path does not exists {}'.format(checkpoint_path)
    checkpoint_data = torch.load(checkpoint_path)
    self.policy_net = checkpoint_data['policy']
    self.posterior_net = checkpoint_data['posterior']
    self.reward_net = checkpoint_data['reward']

  def load_weights_from_vae(self):
    # deepcopy from vae
    self.policy = copy.deepcopy(self.vae_train.policy)
    self.old_policy = copy.deepcopy(self.vae_train.policy)
    self.posterior = copy.deepcopy(self.vae_train.posterior)

    # re-initialize optimizers
    self.opt_policy = optim.Adam(self.policy_net.parameters(), lr=0.0003)
    self.opt_posterior = optim.Adam(self.posterior_net.parameters(), lr=0.0003)


  def set_expert(self, expert):
    assert self.expert is None, "Trying to set non-None expert"
    self.expert = expert
    assert expert.obstacles is not None, "Obstacles cannot be None"
    self.obstacles = expert.obstacles
    assert expert.set_diff is not None, "set_diff in h5 file cannot be None"
    self.set_diff = expert.set_diff

  def get_posterior_reward(self, x, c, next_ct, goal_var=None):
      if goal_var is not None:
          c = torch.cat([goal_var, c], dim=1)

      if self.vae_train.args.use_discrete_vae:
          logits = self.posterior_net(torch.cat((x, c), 1))
          _, label = torch.max(next_ct, 1)
          posterior_reward_t = F.cross_entropy(logits, label)
      else:
          mu, sigma = self.posterior_net(torch.cat((x, c), 1))
          mu = mu.data.cpu().numpy()[0,0]
          sigma = np.exp(0.5 * sigma.data.cpu().numpy()[0,0])

          # TODO: should ideally be logpdf, but pdf may work better. Try both.
          # use norm.logpdf if flag else use norm.pdf
          reward_func = norm.logpdf if self.args.use_log_rewards else \
                  norm.pdf
          scale = sigma if self.args.use_reparameterize else 0.1
          # use fixed std if not using reparameterize otherwise use sigma.
          posterior_reward_t = reward_func(next_ct, loc=mu, scale=0.1)[0]

      return self.args.lambda_posterior * posterior_reward_t.data.cpu().numpy()

  def update_params_for_batch(self,
                              states,
                              actions,
                              latent_c,
                              latent_next_c,
                              targets,
                              advantages,
                              expert_states,
                              expert_actions,
                              expert_latent_c,
                              optim_batch_size,
                              optim_batch_size_exp,
                              optim_iters):
    '''Update parameters for one batch of data.

    Update the policy network, discriminator (reward) network and the posterior
    network here.
    '''
    args, dtype = self.args, self.dtype
    curr_id, curr_id_exp = 0, 0
    for _ in range(optim_iters):
      curr_batch_size = min(optim_batch_size, actions.size(0) - curr_id)
      curr_batch_size_exp = min(optim_batch_size_exp,
                                expert_actions.size(0) - curr_id_exp)
      start_idx, end_idx = curr_id, curr_id + curr_batch_size

      state_var = Variable(states[start_idx:end_idx])
      action_var = Variable(actions[start_idx:end_idx])
      latent_c_var = Variable(latent_c[start_idx:end_idx])
      latent_next_c_var = Variable(latent_next_c[start_idx:end_idx])
      advantages_var = Variable(advantages[start_idx:end_idx])

      start_idx, end_idx = curr_id_exp, curr_id_exp + curr_batch_size_exp
      expert_state_var = Variable(expert_states[start_idx:end_idx])
      expert_action_var = Variable(expert_actions[start_idx:end_idx])
      expert_latent_c_var = Variable(expert_latent_c[start_idx:end_idx])

      # Update reward net
      self.opt_reward.zero_grad()

      # Backprop with expert demonstrations
      expert_output = self.reward_net(torch.cat((expert_state_var,
                                                 expert_action_var,
                                                 expert_latent_c_var), 1))
      expert_disc_loss = self.criterion(expert_output, Variable(torch.zeros(
          expert_action_var.size(0), 1).type(dtype)))
      expert_disc_loss.backward()

      # Backprop with generated demonstrations
      # latent_next_c_var is actual c_t, latent_c_var is c_{t-1}
      gen_output = self.reward_net(torch.cat((state_var,
                                              action_var,
                                              latent_next_c_var), 1))
      gen_disc_loss = self.criterion(gen_output, Variable(
          torch.ones(action_var.size(0), 1)).type(dtype))
      gen_disc_loss.backward()

      # Add loss scalars.
      self.logger.summary_writer.add_scalars(
          'loss/discriminator',
          {
            'total': expert_disc_loss.data[0] + gen_disc_loss.data[0],
            'expert': expert_disc_loss.data[0],
            'gen': gen_disc_loss.data[0],
            },
          self.gail_step_count)
      self.opt_reward.step()
      reward_l2_norm, reward_grad_l2_norm = \
                        get_weight_norm_for_network(self.reward_net)
      self.logger.summary_writer.add_scalar('weight/discriminator/param',
                                            reward_l2_norm,
                                            self.gail_step_count)
      self.logger.summary_writer.add_scalar('weight/discriminator/grad',
                                            reward_grad_l2_norm,
                                            self.gail_step_count)


      # Update posterior net. We need to do this by reparameterization
      # trick instead.  We should not put action_var (a_t) in the
      # posterior net since we need c_t to predict a_t while till now we
      # only have c_{t-1}.
      mu, logvar = self.posterior_net(
          torch.cat((state_var, latent_c_var), 1))

      if args.use_reparameterize:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        predicted_posterior = eps.mul(std).add_(mu)
      else:
        predicted_posterior = mu

      # latent_next_c is of shape (N, 5) where the 1-4 columns of each row
      # represent the goal vector hence we need to extract the last column for
      # the true posterior.
      true_posterior = torch.zeros(latent_next_c_var.size(0), 1).type(dtype)
      true_posterior[:, 0] = latent_next_c_var.data[:, -1]
      posterior_loss = self.criterion_posterior(predicted_posterior,
                                                Variable(true_posterior))
      posterior_loss.backward()
      self.logger.summary_writer.add_scalar('loss/posterior',
                                            posterior_loss.data[0],
                                            self.gail_step_count)


      # compute old and new action probabilities
      action_means, action_log_stds, action_stds = self.policy_net(
              torch.cat((state_var, latent_next_c_var), 1))
      log_prob_cur = normal_log_density(action_var,
                                        action_means,
                                        action_log_stds,
                                        action_stds)

      action_means_old, action_log_stds_old, action_stds_old = \
              self.old_policy_net(torch.cat((state_var, latent_next_c_var), 1))
      log_prob_old = normal_log_density(action_var,
                                        action_means_old,
                                        action_log_stds_old,
                                        action_stds_old)

      if args.use_value_net:
        # update value net
        self.opt_value.zero_grad()
        value_var = self.value_net(torch.cat((state_var, latent_next_c_var), 1))
        value_loss = (value_var - \
                targets[curr_id:curr_id+curr_batch_size]).pow(2.).mean()
        value_loss.backward()
        self.opt_value.step()

      # Update policy net (PPO step)
      self.opt_policy.zero_grad()
      ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
      surr1 = ratio * advantages_var[:, 0]
      surr2 = torch.clamp(ratio,
                          1.0 - self.args.clip_epsilon,
                          1.0 + self.args.clip_epsilon) * advantages_var[:,0]
      policy_surr = -torch.min(surr1, surr2).mean()
      policy_surr.backward()
      # torch.nn.utils.clip_grad_norm(self.policy_net.parameters(), 40)
      self.opt_policy.step()
      self.logger.summary_writer.add_scalar('loss/policy',
                                            policy_surr.data[0],
                                            self.gail_step_count)

      policy_l2_norm, policy_grad_l2_norm = \
                        get_weight_norm_for_network(self.policy_net)
      self.logger.summary_writer.add_scalar('weight/policy/param',
                                            policy_l2_norm,
                                            self.gail_step_count)
      self.logger.summary_writer.add_scalar('weight/policy/grad',
                                            policy_grad_l2_norm,
                                            self.gail_step_count)

      # set new starting point for batch
      curr_id += curr_batch_size
      curr_id_exp += curr_batch_size_exp

      self.gail_step_count += 1

  def update_params(self, gen_batch, expert_batch, episode_idx,
                    optim_epochs, optim_batch_size):
    '''Update params for Policy (G), Reward (D) and Posterior (q) networks.
    '''
    args, dtype = self.args, self.dtype

    self.opt_policy.lr = self.args.learning_rate \
        * max(1.0 - float(episode_idx)/args.num_epochs, 0)
    clip_epsilon = self.args.clip_epsilon \
        * max(1.0 - float(episode_idx)/args.num_epochs, 0)

    # generated trajectories
    states = torch.Tensor(np.array(gen_batch.state)).type(dtype)
    actions = torch.Tensor(np.array(gen_batch.action)).type(dtype)
    rewards = torch.Tensor(np.array(gen_batch.reward)).type(dtype)
    masks = torch.Tensor(np.array(gen_batch.mask)).type(dtype)

    ## Expand states to include history ##
    # Generated trajectories already have history in them.

    latent_c = torch.Tensor(np.array(gen_batch.c)).type(dtype)
    latent_next_c = torch.Tensor(np.array(gen_batch.next_c)).type(dtype)
    if args.use_value_net:
      values = self.value_net(Variable(torch.cat((states, latent_c), 1)))
    else:
      values = None

    # expert trajectories
    list_of_expert_states, list_of_expert_actions = [], []
    list_of_expert_latent_c, list_of_masks = [], []
    for i in range(len(expert_batch.state)):
      # c sampled from expert trajectories is incorrect since we don't have
      # "true c". Hence, we use the trained VAE to get the "true c".
      expert_c, _ = self.get_c_for_traj(expert_batch.state[i][np.newaxis, :],
                                        expert_batch.action[i][np.newaxis, :],
                                        expert_batch.c[i][np.newaxis, :])
      # Remove b
      # expert_c[0, :] is c_{-1} which does not map to s_0. Hence drop it.
      expert_c = expert_c.squeeze(0)[1:, :]
      ## Expand expert states ##
      expanded_states = self.expand_states_numpy(expert_batch.state[i],
                                                 self.history_size)
      list_of_expert_states.append(torch.Tensor(expanded_states))
      list_of_expert_actions.append(torch.Tensor(expert_batch.action[i]))
      list_of_expert_latent_c.append(torch.Tensor(expert_c))
      list_of_masks.append(torch.Tensor(expert_batch.mask[i]))

    expert_states = torch.cat(list_of_expert_states,0).type(dtype)
    expert_actions = torch.cat(list_of_expert_actions, 0).type(dtype)
    expert_latent_c = torch.cat(list_of_expert_latent_c, 0).type(dtype)
    expert_masks = torch.cat(list_of_masks, 0).type(dtype)

    assert expert_states.size(0) == expert_actions.size(0), \
        "Expert transition size do not match"
    assert expert_states.size(0) == expert_latent_c.size(0), \
        "Expert transition size do not match"
    assert expert_states.size(0) == expert_masks.size(0), \
        "Expert transition size do not match"

    # compute advantages
    returns, advantages = get_advantage_for_rewards(rewards,
                                                    masks,
                                                    self.args.gamma,
                                                    values,
                                                    dtype=dtype)
    targets = Variable(returns)
    advantages = (advantages - advantages.mean()) / advantages.std()

    # Backup params after computing probs but before updating new params
    for old_policy_param, policy_param in zip(self.old_policy_net.parameters(),
                                              self.policy_net.parameters()):
      old_policy_param.data.copy_(policy_param.data)

    # update value, reward and policy networks
    optim_iters = self.args.batch_size // optim_batch_size
    optim_batch_size_exp = expert_actions.size(0) // optim_iters

    # Remove extra 1 array shape from actions, since actions were added as
    # 1-hot vector of shape (1, A).
    actions = np.squeeze(actions)
    expert_actions = np.squeeze(expert_actions)

    for _ in range(optim_epochs):
      perm = np.random.permutation(np.arange(actions.size(0)))
      perm_exp = np.random.permutation(np.arange(expert_actions.size(0)))
      if args.cuda:
        perm = torch.cuda.LongTensor(perm)
        perm_exp = torch.cuda.LongTensor(perm_exp)
      else:
        perm, perm_exp = torch.LongTensor(perm), torch.LongTensor(perm_exp)

      self.update_params_for_batch(
          states[perm],
          actions[perm],
          latent_c[perm],
          latent_next_c[perm],
          targets[perm],
          advantages[perm],
          expert_states[perm_exp],
          expert_actions[perm_exp],
          expert_latent_c[perm_exp],
          optim_batch_size,
          optim_batch_size_exp,
          optim_iters)


  def train_gail(self, batch_size=1):
    '''Train GAIL.'''
    args, dtype = self.args, self.dtype
    results = {'average_reward': [], 'episode_reward': [],
            'true_traj_state': {}, 'true_traj_action': {},
            'pred_traj_state': {}, 'pred_traj_action': {}}

    self.train_step_count, self.gail_step_count = 0, 0

    for ep_idx in range(args.num_epochs):
      memory = Memory()

      num_steps = 0
      reward_batch, true_reward_batch = [], []
      expert_true_reward_batch = []
      true_traj_curr_episode = {'state':[], 'action': []}
      gen_traj_curr_episode = {'state': [], 'action': []}

      while num_steps < args.batch_size:
        traj_expert = self.expert.sample(size=batch_size)
        state_expert, action_expert, c_expert, _ = traj_expert
        state_expert = np.array(state_expert, dtype=np.float32)
        action_expert = np.array(action_expert, dtype=np.float32)
        c_expert = np.array(c_expert, dtype=np.int32)

        expert_episode_len = state_expert.shape[1]

        # Generate c from trained VAE
        c_gen, expert_goal = self.get_c_for_traj(state_expert,
                                                 action_expert,
                                                 c_expert)

        if self.args.env_type == 'grid_room':
            true_goal_numpy = np.copy(c_expert)
        else: 
            true_goal_numpy = np.zeros((c_expert.shape[0], self.num_goals))
            true_goal_numpy[np.arange(c_expert.shape[0]), c_expert[:, 0]] = 1
        true_goal = Variable(torch.from_numpy(true_goal_numpy)).type(
                    self.dtype)

        # Sample start state or should we just choose the start state from the
        # expert trajectory sampled above.
        # curr_state_obj = self.sample_start_state()
        if 'grid' in self.args.env_type:
            x_state_obj = StateVector(state_expert[:, 0, :], self.obstacles)
            x_feat = self.vae_train.get_state_features(
                    x_state_obj, self.vae_train.args.use_state_features)
        elif self.args.env_type == 'mujoco':
            x_feat = ep_state[:, 0, :]
            dummy_state = self.env.reset()
            self.env.env.set_state(np.concatenate(
                (np.array([0.0]), x_feat[0, :8]), axis=0), x_feat[0, 8:17])
            dummy_state = x_feat

        x = x_feat

        # Add history to state
        if args.history_size > 1:
          x_hist = -1 * np.ones(
                  (x.shape[0], args.history_size, x.shape[1]),
                  dtype=np.float32)
          x_hist[:, (args.history_size-1), :] = x_feat
          x = self.vae_train.get_history_features(
                  x_hist, self.args.use_velocity_features)


        # TODO: Make this a separate function. Can be parallelized.
        ep_reward, ep_true_reward, expert_true_reward = 0, 0, 0
        true_traj = {'state': [], 'action': []}
        gen_traj = {'state': [], 'action': []}
        gen_traj_dict = {'features': [], 'actions': [], 'c': [], 'mask': []}
        disc_reward, posterior_reward = 0.0, 0.0
        # Use a hard-coded list for memory to gather experience since we need
        # to mutate it before finally creating a memory object.
        memory_list = []
        curr_state_arr = state_expert[:, 0, :]
        for t in range(expert_episode_len):
          ct = c_gen[:, t, :]
          next_ct = c_gen[:, t+1, :]

          x_var = Variable(torch.from_numpy(
              x.reshape((batch_size, -1))).type(self.dtype))
          c_var = Variable(torch.from_numpy(
              ct.reshape((batch_size, -1))).type(self.dtype))
          next_c_var = Variable(torch.from_numpy(
              next_ct.reshape((batch_size, -1))).type(self.dtype))

          # Generator should predict the action using (x_t, c_t)
          action = self.select_action(torch.cat((x_var, c_var), dim=1))
          action_numpy = action.data.cpu().numpy()

          # Save generated and true trajectories
          true_traj['state'].append(state_expert[:, t, :])
          true_traj['action'].append(action_expert[:, t, :])
          gen_traj['state'].append(curr_state_arr)
          gen_traj['action'].append(action_numpy)
          gen_traj_dict['c'].append(ct)

          # TODO: This decay will not work for discrete actions???
          action = epsilon_greedy_linear_decay(action_numpy,
                                               args.num_epochs * 0.5,
                                               ep_idx,
                                               self.action_size,
                                               low=0.05,
                                               high=0.3)

          # Get the discriminator reward
          disc_reward_t = float(self.reward_net(torch.cat(
              (x_var,
               Variable(torch.from_numpy(oned_to_onehot(
                  action, self.action_size)).unsqueeze(0)).type(dtype),
                next_c_var), 1)).data.cpu().numpy()[0,0])

          if disc_reward_t < 1e-6:
            disc_reward_t += 1e-6

          if args.use_log_rewards:
            disc_reward_t = -math.log(disc_reward_t)
          else:
            disc_reward_t = -disc_reward_t

          disc_reward += disc_reward_t

          goal_var = None  
          if self.vae_train.args.use_rnn_goal:
            raise ValueError("To be implemented.")
          else:
            if len(true_goal.size()) == 2:
              goal_var = true_goal
            elif len(true_goal.size()) == 3:
              goal_var = true_goal[:, t, :]  
            else:
              raise ValueError("incorrect true goal size")


          # Get posterior reward
          posterior_reward_t = self.get_posterior_reward(
                  x_var, c_var, next_c_var, goal_var=goal_var)
          posterior_reward += posterior_reward_t

          # Update Rewards
          ep_reward += (disc_reward_t + posterior_reward_t)

          # Since grid world environments don't have a "true" reward let us
          # fake the true reward.
          '''
          true_goal_state = [int(x) for x in state_expert[-1].tolist()]
          if self.args.flag_true_reward == 'grid_reward':
            ep_true_reward += self.true_reward.reward_at_location(
                curr_state_obj.coordinates, goals=[true_goal_state])
            expert_true_reward += self.true_reward.reward_at_location(
                  state_expert[t], goals=[true_goal_state])
          elif self.args.flag_true_reward == 'action_reward':
            ep_true_reward += self.true_reward.reward_at_location(
                np.argmax(action_expert[t]), action)
            expert_true_reward += self.true_reward.corret_action_reward
          else:
            raise ValueError("Incorrect true reward type")
          '''

          # Update next state
          if 'grid' in self.args.env_type:
            action_vec = ActionVector(np.array([action]))
            # Get current state
            state_vec = StateVector(curr_state_arr, self.obstacles)
            # Get next state
            next_state = self.transition_func(state_vec, action_vec, 0)
            next_state_feat = next_state.coordinates

            if self.history_size > 1:
                x_hist[:, self.args.history_size-1] = \
                        self.vae_train.get_state_features(
                                next_state,
                                self.args.use_state_features)
                x = self.vae_train.get_history_features(
                        x_hist, self.args.use_velocity_features)
            else:
                x[:] = self.vae_train.get_state_features(
                        next_state, self.args.use_state_features)
                # Update current state
                curr_state_arr = np.array(next_state.coordinates,
                                          dtype=np.float32)


          elif self.args.env_type == 'mujoco':
            next_state_feat, true_reward, done, _ = self.env.step(action)
            next_state_feat = np.concatenate((next_state_feat,
                              np.array([(t+1)/(episode_len+1)])), axis=0)
          #next_state = running_state(next_state)

          mask = 0 if t == expert_episode_len - 1 else 1

          # Push to memory
          memory_list.append([
            curr_state_arr.reshape(-1),
            np.array([oned_to_onehot(action, self.action_size)]),
            mask,
            next_state_feat,
            disc_reward_t + posterior_reward_t,
            ct.reshape(-1),
            next_ct.reshape(-1)])

          if args.render:
            env.render()

          if mask == 0:
            break

          # Update current state
          curr_state_arr = np.array(next_state.coordinates, dtype=np.float32)


        # Add RNN goal reward, i.e. compare the goal generated by Q-network
        # for the generated trajectory and the predicted goal for expert
        # trajectory.
        goal_reward = 0
        if self.vae_train.use_rnn_goal_predictor:
            gen_goal, _ = self.vae_train.predict_goal(
                np.array(gen_traj_dict['features']),
                np.array(gen_traj_dict['actions']).reshape(
                    (-1, self.action_size)),
                gen_traj_dict['c'],
                None,
                self.num_goals)
            # Goal reward is sum(p*log(p_hat))
            gen_goal_numpy = gen_goal.data.cpu().numpy().reshape((-1))
            goal_reward = np.sum(np.log(gen_goal_numpy)
                * expert_goal.data.cpu().numpy().reshape((-1)))
        # Add goal_reward to memory
        assert memory_list[-1][2] == 0, "Mask for final end state is not 0."
        for memory_t in memory_list:
          memory_t[4] += (goal_reward / expert_episode_len)
          memory.push(*memory_t)

        self.logger.summary_writer.add_scalars(
            'gen_traj/gen_reward',
            {
              'discriminator': disc_reward,
              'posterior': posterior_reward,
              'goal': goal_reward,
            },
            self.train_step_count
        )

        #ep_memory.push(memory)
        num_steps += (t-1)
        reward_batch.append(ep_reward)
        true_reward_batch.append(ep_true_reward)
        expert_true_reward_batch.append(expert_true_reward)
        results['episode_reward'].append(ep_reward)

        # Append trajectories
        true_traj_curr_episode['state'].append(true_traj['state'])
        true_traj_curr_episode['action'].append(true_traj['action'])
        gen_traj_curr_episode['state'].append(gen_traj['state'])
        gen_traj_curr_episode['action'].append(gen_traj['action'])

      results['average_reward'].append(np.mean(reward_batch))

      # Add to tensorboard
      self.logger.summary_writer.add_scalars(
          'gen_traj/reward', {
            'average': np.mean(reward_batch),
            'max': np.max(reward_batch),
            'min': np.min(reward_batch)
            },
          self.train_step_count)
      self.logger.summary_writer.add_scalars(
          'gen_traj/true_reward',
          {
            'average': np.mean(true_reward_batch),
            'max': np.max(true_reward_batch),
            'min': np.min(true_reward_batch),
            'expert_true': np.mean(expert_true_reward_batch)
          },
          self.train_step_count)

      # Add predicted and generated trajectories to results
      if ep_idx % self.args.save_interval == 0:
        results['true_traj_state'][ep_idx] = copy.deepcopy(
                true_traj_curr_episode['state'])
        results['true_traj_action'][ep_idx] = copy.deepcopy(
                true_traj_curr_episode['action'])
        results['pred_traj_state'][ep_idx] = copy.deepcopy(
                gen_traj_curr_episode['state'])
        results['pred_traj_action'][ep_idx] = copy.deepcopy(
                gen_traj_curr_episode['action'])

      # Update parameters
      gen_batch = memory.sample()

      # We do not get the context variable from expert trajectories. Hence we
      # need to fill it in later.
      expert_batch = self.expert.sample(size=args.num_expert_trajs)

      self.update_params(gen_batch, expert_batch, ep_idx,
                         args.optim_epochs, args.optim_batch_size)

      self.train_step_count += 1

      if ep_idx > 0 and  ep_idx % args.log_interval == 0:
        print('Episode [{}/{}]  Avg R: {:.2f}   Max R: {:.2f} \t' \
              'True Avg {:.2f}   True Max R: {:.2f}   ' \
              'Expert (Avg): {:.2f}'.format(
              ep_idx, args.num_epochs, np.mean(reward_batch),
              np.max(reward_batch), np.mean(true_reward_batch),
              np.max(true_reward_batch), np.mean(expert_true_reward_batch)))

      results_path = os.path.join(args.results_dir, 'results.pkl')
      with open(results_path, 'wb') as results_f:
        pickle.dump((results), results_f, protocol=2)
        # print("Did save results to {}".format(results_path))

      if ep_idx % args.save_interval == 0:
        checkpoint_filepath = self.model_checkpoint_filepath(ep_idx)
        torch.save(self.checkpoint_data_to_save(), checkpoint_filepath)
        print("Did save checkpoint: {}".format(checkpoint_filepath))

def load_VAE_model(model_checkpoint_path, new_args):
    '''Load pre-trained VAE model.'''

    checkpoint_dir_path = os.path.dirname(model_checkpoint_path)
    results_dir_path = os.path.dirname(checkpoint_dir_path)

    # Load arguments used to train the model
    saved_args_filepath = os.path.join(results_dir_path, 'args.pkl')
    with open(saved_args_filepath, 'rb') as saved_args_f:
        saved_args = pickle.load(saved_args_f)
        print('Did load saved args {}'.format(saved_args_filepath))

    dtype = torch.FloatTensor
    # Use new args to load the previously saved models as well
    if new_args.cuda:
        dtype = torch.cuda.FloatTensor
    logger = TensorboardXLogger(os.path.join(args.results_dir, 'log_vae_model'))
    vae_train = VAETrain(
        saved_args,
        logger,
        width=11,
        height=15,
        state_size=saved_args.vae_state_size,
        action_size=saved_args.vae_action_size,
        history_size=saved_args.vae_history_size,
        num_goals=saved_args.vae_goal_size,
        use_rnn_goal_predictor=saved_args.use_rnn_goal,
        dtype=dtype,
        env_type=args.env_type,
        env_name=args.env_name
    )

    vae_train.load_checkpoint(model_checkpoint_path)
    if new_args.cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    vae_train.convert_models_to_type(dtype)
    print("Did load models at: {}".format(model_checkpoint_path))
    return vae_train

def main(args):
    # Create Logger
    if not os.path.exists(os.path.join(args.results_dir, 'log')):
        os.makedirs(os.path.join(args.results_dir, 'log'))
    logger = TensorboardXLogger(os.path.join(args.results_dir, 'log'))

    print('Loading expert trajectories ...')
    if 'grid' in args.env_type:
        if args.env_type == 'grid_room':
            expert = SeparateRoomTrajExpert(args.expert_path, args.state_size)
        else:
            expert = ExpertHDF5(args.expert_path, args.state_size)
        expert.push(only_coordinates_in_state=True, one_hot_action=True)
    elif args.env_type == 'mujoco':
        expert = ExpertHDF5(args.expert_path, args.state_size)
        expert.push(only_coordinates_in_state=False, one_hot_action=False)
    print('Expert trajectories loaded.')

    # Load pre-trained VAE model
    vae_train = load_VAE_model(args.vae_checkpoint_path, args)
    vae_train.set_expert(expert)

    dtype = torch.FloatTensor
    if args.cuda:
        dtype = torch.cuda.FloatTensor

    causal_gail_mlp = CausalGAILMLP(
            args,
            vae_train,
            logger,
            state_size=vae_train.args.vae_state_size,
            action_size=vae_train.args.vae_action_size,
            context_size=vae_train.args.vae_context_size,
            num_goals=vae_train.args.vae_goal_size,
            history_size=vae_train.args.vae_history_size,
            dtype=dtype)

    if args.init_from_vae:
        causal_gail_mlp.load_weights_from_vae()

    causal_gail_mlp.set_expert(expert) # not implemented?
    causal_gail_mlp.train_gail()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Causal GAIL using MLP.')
    parser.add_argument('--expert_path', default="L_expert_trajectories/",
                        help='path to the expert trajectory files')

    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    # Environment parameters
    parser.add_argument('--state_size', type=int, default=2,
                        help='State size for VAE.')
    parser.add_argument('--action_size', type=int, default=4,
                        help='Action size for VAE.')
    parser.add_argument('--history_size', type=int, default=1,
                        help='State history size to use in VAE.')
    parser.add_argument('--context_size', type=int, default=1,
                        help='Context size for VAE.')

    # RL parameters
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae (default: 0.95)')

    parser.add_argument('--lambda_posterior', type=float, default=1.0,
                        help='Parameter to scale MI loss from the posterior.')
    parser.add_argument('--lambda_goal_pred_reward', type=float, default=1.0,
                        help='Reward scale for goal prediction reward from RNN.')

    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='gae (default: 3e-4)')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='batch size (default: 2048)')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='number of episodes (default: 500)')
    parser.add_argument('--max_ep_length', type=int, default=1000,
                        help='maximum episode length.')

    parser.add_argument('--optim_epochs', type=int, default=5,
                        help='number of epochs over a batch (default: 5)')
    parser.add_argument('--optim_batch_size', type=int, default=64,
                        help='batch size for epochs (default: 64)')
    parser.add_argument('--num_expert_trajs', type=int, default=5,
                        help='number of expert trajectories in a batch.')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    # Log interval
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Interval between training status logs')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Interval between saving policy weights')
    parser.add_argument('--entropy_coeff', type=float, default=0.0,
                        help='coefficient for entropy cost')
    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                        help='Clipping for PPO grad')

    # Path to pre-trained VAE model
    parser.add_argument('--vae_checkpoint_path', type=str, required=True,
                        help='Path to pre-trained VAE model.')
    # Path to store training results in
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to store results in.')

    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='enables CUDA training')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                        help='Disable CUDA training')
    parser.set_defaults(cuda=False)

    # Use features
    parser.add_argument('--use_state_features', dest='use_state_features',
                        action='store_true',
                        help='Use features instead of direct (x,y) values in VAE')
    parser.add_argument('--no-use_state_features', dest='use_state_features',
                        action='store_false',
                        help='Do not use features instead of direct (x,y) ' \
                            'values in VAE')
    parser.set_defaults(use_state_features=False)

    # Use reparameterization for posterior training.
    parser.add_argument('--use_reparameterize', dest='use_reparameterize',
                        action='store_true',
                        help='Use reparameterization during posterior training ' \
                            'values in VAE')
    parser.add_argument('--no-use_reparameterize', dest='use_reparameterize',
                        action='store_false',
                        help='Use reparameterization during posterior training ' \
                            'values in VAE')
    parser.set_defaults(use_reparameterize=False)

    parser.add_argument('--flag_true_reward', type=str, default='grid_reward',
                        choices=['grid_reward', 'action_reward'],
                        help='True reward type to use.')

    parser.add_argument('--use_log_rewards', dest='use_log_rewards',
                        action='store_true',
                        help='Use log with rewards.')
    parser.add_argument('--no-use_log_rewards', dest='use_log_rewards',
                        action='store_false',
                        help='Don\'t Use log with rewards.')
    parser.set_defaults(use_log_rewards=True)

    parser.add_argument('--use_value_net', dest='use_value_net',
                        action='store_true',
                        help='Use value network.')
    parser.add_argument('--no-use_value_net', dest='use_value_net',
                        action='store_false',
                        help='Don\'t use value network.')
    parser.set_defaults(use_value_net=True)

    parser.add_argument('--init_from_vae', dest='init_from_vae',
                        action='store_true',
                        help='Init policy and posterior from vae.')
    parser.add_argument('--no-init_from_vae', dest='init_from_vae',
                        action='store_false',
                        help='Don\'t init policy and posterior from vae.')
    parser.set_defaults(init_from_vae=True)

    # Environment - Grid or Mujoco
    parser.add_argument('--env-type', default='grid',
                        choices=['grid', 'grid_room', 'mujoco'],
                        help='Environment type Grid or Mujoco.')
    parser.add_argument('--env-name', default=None,
                        help='Environment name if Mujoco.')

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
        # Directory for TF logs
        os.makedirs(os.path.join(args.results_dir, 'log'))
        # Directory for model checkpoints
        os.makedirs(os.path.join(args.results_dir, 'checkpoint'))

    # Save runtime arguments to pickle file
    args_pkl_filepath = os.path.join(args.results_dir, 'args.pkl')
    with open(args_pkl_filepath, 'wb') as args_pkl_f:
        pickle.dump(args, args_pkl_f, protocol=2)

    main(args)
