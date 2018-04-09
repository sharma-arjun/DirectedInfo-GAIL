import argparse
import sys
import os
import pdb
import pickle
import math
import random
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

from models import Policy, Posterior, Reward, Value
from grid_world import State, Action, TransitionFunction
from grid_world import RewardFunction, RewardFunction_SR2
from grid_world import create_obstacles, obstacle_movement, sample_start
from load_expert_traj import Expert, ExpertHDF5
from replay_memory import Memory
from running_state import ZFilter
from utils.torch_utils import clip_grads

from vae import VAE, VAETrain
from utils.logger import Logger, TensorboardXLogger
from utils.rl_utils import epsilon_greedy_linear_decay, epsilon_greedy
from utils.rl_utils import greedy, oned_to_onehot, normal_log_density
from utils.rl_utils import get_advantage_for_rewards

class CausalGAILMLP(object):
  def __init__(self,
               args,
               vae_model,
               env_data,
               state_size=2,
               action_size=4,
               context_size=1,
               num_goals=4,
               history_size=1,
               dtype=torch.FloatTensor):
    self.args = args
    self.vae_model = vae_model
    self.state_size = state_size
    self.action_size = action_size
    self.history_size = history_size
    self.context_size = context_size
    self.num_goals = num_goals
    self.dtype = dtype

    self.policy_net = Policy(state_size,
                             0,
                             context_size,
                             action_size,
                             hidden_size=64,
                             output_activation='sigmoid')
    self.old_policy_net = Policy(state_size,
                                 0,
                                 context_size,
                                 action_size,
                                 hidden_size=64,
                                 output_activation='sigmoid')

    #value_net = Value(num_inputs+num_c, hidden_size=64).type(dtype)
    # Reward net is the discriminator network.
    self.reward_net = Reward(state_size,
                             action_size,
                             context_size,
                             hidden_size=64)

    self.posterior_net = Posterior(state_size,
                                   action_size,
                                   context_size,
                                   hidden_size=64)

    self.opt_policy = optim.Adam(self.policy_net.parameters(), lr=0.0003)
    #self.opt_value = optim.Adam(value_net.parameters(), lr=0.0003)
    self.opt_reward = optim.Adam(self.reward_net.parameters(), lr=0.0003)
    self.opt_posterior = optim.Adam(self.posterior_net.parameters(), lr=0.0003)

    # Create loss functions
    self.criterion = nn.BCELoss()
    #criterion_posterior = nn.NLLLoss()
    self.criterion_posterior = nn.MSELoss()

    self.create_environment(env_data)
    self.expert = None
    self.obstacles, self.set_diff = None, None

  def create_environment(self, env_data):
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

  def get_c_for_traj(self, state_arr, action_arr, c_arr):
    '''Get c[1:T] for given trajectory.'''
    traj_len = len(state_arr)
    c = -1*np.ones((traj_len+1, self.context_size))

    # Use the Q-network (RNN) to predict goal.
    pred_goal, _ = self.vae_model.predict_goal(
        state_arr, action_arr, c_arr, None, self.num_goals)

    # c[0, -1] Should be -1, hence we don't change it
    c[:, :-1] = pred_goal.data[0].cpu().numpy()
    x = -1*np.ones((1, self.history_size, self.state_size))

    for t in range(traj_len):
      # Shift history
      if self.history_size > 1:
        x[:, :-1, :] = x[:, 1:, :]
      x[:, -1, :] = state_arr[t]
      # Create inputs
      input_x = Variable(torch.from_numpy(
        np.reshape(x, (1, -1))).type(self.dtype))
      input_c = Variable(torch.from_numpy(
        np.reshape(c[t, :], (1, -1))).type(self.dtype))
      # Get c_t
      c_t = self.vae_model.get_context_at_state(input_x, input_c)
      c[t+1, -1] = c_t.data.cpu().numpy()

    return c

  def sample_start_state(self):
    start_loc = sample_start(self.set_diff)
    return State(start_loc, self.obstacles)

  def checkpoint_data_to_save(self):
    return {
        'policy': policy_net,
        'posterior': posterior_net,
        'reward': self.reward_net
        }

  def load_checkpoint_data(self, checkpoint_path):
    assert os.path.exists(checkpoint_path), \
        'Checkpoint path does not exists {}'.format(checkpoint_path)
    checkpoint_data = torch.load(checkpoint_path)
    self.policy_net = checkpoint_data['policy']
    self.posterior_net = checkpoint_data['posterior']
    self.reward_net = checkpoint_data['reward']

  def model_checkpoint_filepath(self, epoch):
    checkpoint_dir = os.path.join(self.args.results_dir, 'checkpoint')
    return os.path.join(checkpoint_dir, 'cp_{}.pth'.format(epoch))

  def expand_states_torch(self, states, history_size):
    if self.history_size == 1:
      return states

    expanded_states = -1*torch.ones(
        states.size(0), states.size(1)*history_size).type(self.dtype)

    for i in range(states.size(0)):
        expanded_states[i, :states.size(1)] = states[i,:]
        if i > 0:
            expanded_states[i, states.size(1):] = \
                expanded_states[i-1, :(history_size-1)*states.size(1)]

    return expanded_states

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
      gen_output = self.reward_net(torch.cat((state_var,
                                              action_var,
                                              latent_c_var), 1))
      gen_disc_loss = self.criterion(gen_output, Variable(
          torch.ones(action_var.size(0), 1)).type(dtype))
      gen_disc_loss.backward()

      self.opt_reward.step()


      # Update posterior net. We need to do this by reparameterization
      # trick instead.  We should not put action_var (a_t) in the
      # posterior net since we need c_t to predict a_t while till now we
      # only have c_{t-1}.
      mu, _ = self.posterior_net(torch.cat((state_var,
                                            action_var,
                                            latent_c_var), 1))

      # latent_next_c is of shape (N, 5) where the 1-4 columns of each row
      # represent the goal vector hence we need to extract the last column for
      # the true posterior.
      true_posterior = torch.zeros(latent_next_c_var.size(0), 1).type(dtype)
      true_posterior[:, 0] = latent_next_c_var.data[:, -1]
      posterior_loss = self.criterion_posterior(mu, Variable(true_posterior))
      posterior_loss.backward()

      # compute old and new action probabilities
      action_means, action_log_stds, action_stds = self.policy_net(
              torch.cat((state_var, latent_c_var), 1))
      log_prob_cur = normal_log_density(action_var,
                                        action_means,
                                        action_log_stds,
                                        action_stds)

      action_means_old, action_log_stds_old, action_stds_old = \
              self.old_policy_net(torch.cat((state_var, latent_c_var), 1))
      log_prob_old = normal_log_density(action_var,
                                        action_means_old,
                                        action_log_stds_old,
                                        action_stds_old)

      # update value net
      # opt_value.zero_grad()
      # value_var = value_net(state_var)
      # value_loss = (value_var - \
      #    targets[cur_id:cur_id+cur_batch_size]).pow(2.).mean()
      # value_loss.backward()
      # opt_value.step()

      # Update policy net (PPO step)
      self.opt_policy.zero_grad()
      ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
      surr1 = ratio * advantages_var[:, 0]
      surr2 = torch.clamp(ratio,
                          1.0 - self.args.clip_epsilon,
                          1.0 + self.args.clip_epsilon) * advantages_var[:,0]
      policy_surr = -torch.min(surr1, surr2).mean()
      policy_surr.backward()
      torch.nn.utils.clip_grad_norm(self.policy_net.parameters(), 40)
      self.opt_policy.step()

      # set new starting point for batch
      curr_id += curr_batch_size
      curr_id_exp += curr_batch_size_exp

      # TODO: Save statistics in tensorboard


  def update_params(self, gen_batch, expert_batch, episode_idx,
                    optim_epochs, optim_batch_size):
    '''Update params for Policy (G), Reward (D) and Posterior (q) networks.
    '''
    args, dtype = self.args, self.dtype

    self.opt_policy.lr = self.args.learning_rate \
        * max(1.0 - float(episode_idx)/args.num_episodes, 0)
    clip_epsilon = self.args.clip_epsilon \
        * max(1.0 - float(episode_idx)/args.num_episodes, 0)

    # generated trajectories
    states = torch.Tensor(np.array(gen_batch.state)).type(dtype)
    actions = torch.Tensor(np.array(gen_batch.action)).type(dtype)
    rewards = torch.Tensor(np.array(gen_batch.reward)).type(dtype)
    masks = torch.Tensor(np.array(gen_batch.mask)).type(dtype)

    ## Expand states to include history ##
    states = self.expand_states_torch(states, self.history_size)

    latent_c = torch.Tensor(np.array(gen_batch.c)).type(dtype)
    latent_next_c = torch.Tensor(np.array(gen_batch.next_c)).type(dtype)
    #values = value_net(Variable(states))

    # expert trajectories
    list_of_expert_states, list_of_expert_actions = [], []
    list_of_expert_latent_c, list_of_masks = [], []
    for i in range(len(expert_batch.state)):
      # c sampled from expert trajectories is incorrect since we don't have
      # "true c". Hence, we use the trained VAE to get the "true c".
      expert_c = self.get_c_for_traj(expert_batch.state[i],
                                     expert_batch.action[i],
                                     expert_batch.c[i])
      list_of_expert_states.append(torch.Tensor(expert_batch.state[i]))
      list_of_expert_actions.append(torch.Tensor(expert_batch.action[i]))
      list_of_expert_latent_c.append(torch.Tensor(expert_c))
      list_of_masks.append(torch.Tensor(expert_batch.mask[i]))

    expert_states = torch.cat(list_of_expert_states,0).type(dtype)
    expert_actions = torch.cat(list_of_expert_actions, 0).type(dtype)
    expert_latent_c = torch.cat(list_of_expert_latent_c, 0).type(dtype)
    expert_masks = torch.cat(list_of_masks, 0).type(dtype)

    ## Expand expert states ##
    expert_states = self.expand_states_torch(expert_states, self.history_size)

    # compute advantages
    returns, advantages = get_advantage_for_rewards(rewards,
                                                    masks,
                                                    self.args.gamma,
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


  def train_gail(self, expert):
    '''Train GAIL.'''
    args, dtype = self.args, self.dtype
    stats = {'average_reward': [], 'episode_reward': []}
    for ep_idx in range(args.num_episodes):
      memory = Memory()

      num_steps, num_episodes = 0, 0
      reward_batch, true_reward_batch = 0, 0
      while num_steps < args.batch_size:
        traj_expert = expert.sample(size=1)
        state_expert, action_expert, c_expert, _ = traj_expert

        # Expert state and actions
        state_expert = state_expert[0]
        action_expert = action_expert[0]
        c_expert = c_expert[0]
        episode_len = len(state_expert)

        # Generate c from trained VAE
        c_gen = self.get_c_for_traj(state_expert, action_expert, c_expert)

        # Sample start state or should we just choose the start state from the
        # expert trajectory sampled above.
        curr_state_obj = self.sample_start_state()
        curr_state_feat = self.get_state_features(curr_state_obj,
                                                  self.args.use_state_features)
        #state = running_state(state)

        # TODO: Make this a separate function. Can be parallelized.
        #memory = Memory()
        ep_reward = 0
        for t in range(episode_len):
          ct = c_gen[t, :]
          action = self.select_action(np.concatenate((curr_state_feat, ct)))
          action = epsilon_greedy_linear_decay(action.data.cpu().numpy(),
                                               args.num_episodes * 0.5,
                                               ep_idx,
                                               self.action_size,
                                               low=0.05,
                                               high=0.3)

          # Get the discriminator reward
          # TODO: Shouldn't we take the log of discriminator reward.
          reward = -float(self.reward_net(torch.cat(
            (Variable(torch.from_numpy(curr_state_feat).unsqueeze(
                0)).type(dtype),
              Variable(torch.from_numpy(oned_to_onehot(
                action, self.action_size)).unsqueeze(0)).type(dtype),
              Variable(torch.from_numpy(ct).unsqueeze(0)).type(dtype)),
             1)).data.cpu().numpy()[0,0])

          if t < episode_len:
            # Predict c_t given (x_t, c_{t-1})
            mu, sigma = self.posterior_net(
                torch.cat((
                  Variable(torch.from_numpy(curr_state_feat).unsqueeze(
                    0)).type(dtype),
                  Variable(torch.from_numpy(oned_to_onehot(
                    action, self.action_size)).unsqueeze(0)).type(dtype),
                  Variable(torch.from_numpy(ct).unsqueeze(0)).type(dtype)), 1))

            mu = mu.data.cpu().numpy()[0,0]
            sigma = sigma.data.cpu().numpy()[0,0]

            # TODO: should ideally be logpdf, but pdf may work better. Try both.
            next_ct = c_gen[t+1, -1]
            reward += (self.args.lambda_posterior *
                norm.pdf(next_ct, loc=mu, scale=abs(sigma)))

          ep_reward += reward

          next_state_obj = self.transition_func(curr_state_obj,
                                                Action(action),
                                                0)
          next_state_feat = self.get_state_features(
              next_state_obj, self.args.use_state_features)
          #next_state = running_state(next_state)
          mask = 0 if t == args.max_ep_length - 1 else 1

          # Create next_ct_array since next_ct is a scalar.
          next_ct_array = np.copy(ct)
          next_ct_array[-1] = next_ct

          # Push to memory
          memory.push(curr_state_feat,
                      np.array([oned_to_onehot(action, self.action_size)]),
                      mask,
                      next_state_feat,
                      reward,
                      ct,
                      next_ct_array)

          if args.render:
            env.render()

          if not mask:
            break

          curr_state_obj, curr_state_feat = next_state_obj, next_state_feat

        #ep_memory.push(memory)
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += ep_reward
        true_reward_batch += 0.0
        stats['episode_reward'].append(ep_reward)

      stats['average_reward'].append(reward_batch / num_episodes)

      #optim_batch_size = min(num_episodes,
      #                        max(10,int(num_episodes*0.05)))

      # Update parameters
      gen_batch = memory.sample()

      # We do not get the context variable from expert trajectories. Hence we
      # need to fill it in later.
      expert_batch = expert.sample(size=args.num_expert_trajs)

      self.update_params(gen_batch, expert_batch, ep_idx,
                         args.optim_epochs, args.optim_batch_size)

      if ep_idx > 0 and  ep_idx % args.log_interval == 0:
        print('Episode [{}/{}]   Last R: {:.2f}   Avg R: {:.2f} \t' \
              'Last true R {:.2f}   Avg true R: {:.2f}'.format(
              ep_idx, args.num_episodes, 0.0, reward_batch/num_episodes,
              0.0, true_reward_batch/num_episodes))

      results_path = os.path.join(args.results_dir, 'results.pkl')
      with open(results_path, 'wb') as results_f:
        pickle.dump((stats), results_f, protocol=2)

      if ep_idx > 0 and ep_idx % args.save_interval == 0:
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
  if saved_args.cuda:
    dtype = torch.cuda.FloatTensor
  logger = TensorboardXLogger(os.path.join(args.results_dir, 'log_vae_model'))
  vae_train = VAETrain(
    saved_args,
    logger,
    width=21,
    height=21,
    state_size=saved_args.vae_state_size,
    action_size=saved_args.vae_action_size,
    history_size=saved_args.vae_history_size,
    num_goals=4,
    use_rnn_goal_predictor=saved_args.use_rnn_goal,
    dtype=dtype
  )
  vae_train.load_checkpoint(model_checkpoint_path)
  print("Did load models at: {}".format(model_checkpoint_path))
  return vae_train

def main(args):
  expert = ExpertHDF5(args.expert_path, 2)
  print('Loading expert trajectories ...')
  expert.push(only_coordinates_in_state=True, one_hot_action=True)
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
      None,
      state_size=args.state_size,
      action_size=args.action_size,
      context_size=4 + args.context_size,  # num_goals + context_size
      num_goals=4,
      history_size=args.history_size,
      dtype=dtype)
  causal_gail_mlp.set_expert(expert)
  causal_gail_mlp.train_gail(expert)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Causal GAIL using MLP.')
  parser.add_argument('--expert-path', default="L_expert_trajectories/",
                      help='path to the expert trajectory files')

  parser.add_argument('--seed', type=int, default=1,
                      help='random seed (default: 1)')

  # Environment parameters
  parser.add_argument('--state-size', type=int, default=2,
                      help='State size for VAE.')
  parser.add_argument('--action-size', type=int, default=4,
                      help='Action size for VAE.')
  parser.add_argument('--history-size', type=int, default=1,
                        help='State history size to use in VAE.')
  parser.add_argument('--context-size', type=int, default=1,
                      help='Context size for VAE.')

  # RL parameters
  parser.add_argument('--gamma', type=float, default=0.99,
                      help='discount factor (default: 0.99)')
  parser.add_argument('--tau', type=float, default=0.95,
                      help='gae (default: 0.95)')

  parser.add_argument('--lambda_posterior', type=float, default=1.0,
                      help='Parameter to scale MI loss from the posterior.')

  # Training parameters
  parser.add_argument('--learning-rate', type=float, default=3e-4,
                      help='gae (default: 3e-4)')
  parser.add_argument('--batch-size', type=int, default=2048,
                      help='batch size (default: 2048)')
  parser.add_argument('--num-episodes', type=int, default=500,
                      help='number of episodes (default: 500)')
  parser.add_argument('--max-ep-length', type=int, default=1000,
                      help='maximum episode length.')

  parser.add_argument('--optim-epochs', type=int, default=5,
                      help='number of epochs over a batch (default: 5)')
  parser.add_argument('--optim-batch-size', type=int, default=64,
                      help='batch size for epochs (default: 64)')
  parser.add_argument('--num-expert-trajs', type=int, default=5,
                      help='number of expert trajectories in a batch.')
  parser.add_argument('--render', action='store_true',
                      help='render the environment')
  # Log interval
  parser.add_argument('--log-interval', type=int, default=1,
                      help='Interval between training status logs')
  parser.add_argument('--save-interval', type=int, default=100,
                      help='Interval between saving policy weights')
  parser.add_argument('--entropy-coeff', type=float, default=0.0,
                      help='coefficient for entropy cost')
  parser.add_argument('--clip-epsilon', type=float, default=0.2,
                      help='Clipping for PPO grad')

  # Path to pre-trained VAE model
  parser.add_argument('--vae-checkpoint-path', type=str, required=True,
                      help='Path to pre-trained VAE model.')
  # Path to store training results in
  parser.add_argument('--results-dir', type=str, required=True,
                      help='Path to store results in.')

  parser.add_argument('--cuda', dest='cuda', action='store_true',
                      help='enables CUDA training')
  parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                      help='Disable CUDA training')
  parser.set_defaults(cuda=False)

  # Use features
  parser.add_argument('--use-state-features', dest='use_state_features',
                      action='store_true',
                      help='Use features instead of direct (x,y) values in VAE')
  parser.add_argument('--no-use-state-features', dest='use_state_features',
                      action='store_false',
                      help='Do not use features instead of direct (x,y) ' \
                          'values in VAE')
  parser.set_defaults(use_state_features=False)

  args = parser.parse_args()
  torch.manual_seed(args.seed)

  if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)
    # Directory for TF logs
    os.makedirs(os.path.join(args.results_dir, 'log'))
    # Directory for model checkpoints
    os.makedirs(os.path.join(args.results_dir, 'checkpoint'))

  main(args)
