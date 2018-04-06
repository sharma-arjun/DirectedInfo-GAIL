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

from utils.rl_utils import epsilon_greedy_linear_decay, epsilon_greedy
from utils.rl_utils import greedy, oned_to_onehot, normal_log_density

class CausalGAILMLP(object):
  def __init__(self,
               args,
               vae_model,
               env_data,
               state_size=2,
               action_size=4,
               num_goals=4,
               history_size=1,
               dtype=torch.FloatTensor):
    self.args = args
    self.vae_model = vae_model
    self.state_size = state_size
    self.action_size = action_size
    self.history_size = history_size
    self.context_size = 0  # TODO
    self.num_goals = num_goals
    self.dtype = dtype

    self.policy_net = Policy(num_inputs,
                             0,
                             num_c,
                             num_actions,
                             hidden_size=64,
                             output_activation='sigmoid')
    self.old_policy_net = Policy(num_inputs,
                                 0,
                                 num_c,
                                 num_actions,
                                 hidden_size=64,
                                 output_activation='sigmoid')

    #value_net = Value(num_inputs+num_c, hidden_size=64).type(dtype)
    # Reward net is the discriminator network.
    self.reward_net = Reward(num_inputs,
                             num_actions,
                             num_c,
                             hidden_size=64)

    self.posterior_net = Posterior(num_inputs,
                                   num_actions,
                                   num_c,
                                   hidden_size=64)

    self.opt_policy = optim.Adam(policy_net.parameters(), lr=0.0003)
    #self.opt_value = optim.Adam(value_net.parameters(), lr=0.0003)
    self.opt_reward = optim.Adam(reward_net.parameters(), lr=0.0003)
    self.opt_posterior = optim.Adam(posterior_net.parameters(), lr=0.0003)

    self.create_environment(env_data)

    # Load the true true reward
    if args.expert_path == 'SR2_expert_trajectories/':
        self.R = RewardFunction_SR2(-1.0,1.0,width)
    else:
        self.R = RewardFunction(-1.0,1.0)

  def create_environment(self, env_data):
    self.width, self.height = 21, 21
    obstacles = create_obstacles(self.width, self.height, 'diverse')
    set_diff = list(set(product(tuple(range(7,13)),tuple(range(7,13)))))
    start_loc = sample_start(set_diff)

    s = State(start_loc, obstacles)
    self.T = TransitionFunction(width, height, obstacle_movement)

  def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0).type(dtype)
    action, _, _ = policy_net(Variable(state))
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

  def get_c_for_traj(self, traj):
    '''Get c[1:T] for given trajectory.'''
    num_t_steps = len(traj)
    c = -1*np.ones((num_t_steps+1,num_c))
    # TODO: We should use the RNN (Q-network) to predict the goal
    c[0,1] = expert_sample.g[0]
    x = -1*np.ones((1, self.history_size, self.state_size))
    for t in range(len(expert_states)):
      # Shift history
      x[:, :(history_size-1), :] = x[:, 1:, :]
      x[:, -1, :] = expert.states[t]
      input_x = Variable(torch.from_numpy(
        np.reshape(x, (1, -1))).type(self.dtype))
      mu, logvar = vae_model.encode(input_x, c[t,:])
      c[t+1, :] = vae_model.reparameterize(mu, logvar)
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
    return os.path.join(args.checkpoint, 'cp_{}.pth'.format(epoch))

  def update_params(self, gen_batch, expert_batch, i_episode,
                    optim_epochs, optim_batch_size):
    dtype = self.dtype
    criterion = nn.BCELoss()
    #criterion_posterior = nn.NLLLoss()
    criterion_posterior = nn.MSELoss()

    opt_policy.lr = self.args.learning_rate \
            * max(1.0 - float(i_episode)/self.args.num_episodes, 0)
    clip_epsilon = self.args.clip_epsilon \
            * max(1.0 - float(i_episode)/self.args.num_episodes, 0)

    # generated trajectories
    rewards = torch.Tensor(gen_batch.reward).type(dtype)
    masks = torch.Tensor(gen_batch.mask).type(dtype)
    actions = torch.Tensor(np.concatenate(gen_batch.action, 0)).type(dtype)
    states = torch.Tensor(gen_batch.state).type(dtype)

    ## Expand states to include history ##
    expanded_states = -1*torch.ones(
        states.size(0), states.size(1)*self.history_size).type(dtype)

    for i in range(states.size(0)):
        expanded_states[i,:states.size(1)] = states[i,:]
        if i > 0:
            expanded_states[i, states.size(1):] = \
                expanded_states[i-1, :(self.history_size-1)*states.size(1)]

    ## set it back to states ##
    states = expanded_states

    latent_c = torch.Tensor(gen_batch.c).type(dtype)
    latent_next_c = torch.Tensor(gen_batch.next_c).type(dtype)
    #values = value_net(Variable(states))

    # expert trajectories
    list_of_expert_states = []
    for i in range(len(expert_batch.state)):
        list_of_expert_states.append(torch.Tensor(expert_batch.state[i]))
    expert_states = torch.cat(list_of_expert_states,0).type(dtype)

    list_of_expert_actions = []
    for i in range(len(expert_batch.action)):
        list_of_expert_actions.append(torch.Tensor(expert_batch.action[i]))
    expert_actions = torch.cat(list_of_expert_actions, 0).type(dtype)

    # latent c here will all be g for corresponding expert traj. Set later
    list_of_expert_latent_c = []
    for i in range(len(expert_batch.c)):
        list_of_expert_latent_c.append(torch.Tensor(expert_batch.c[i]))
    expert_latent_c = torch.cat(list_of_expert_latent_c, 0).type(dtype)

    list_of_masks = []
    for i in range(len(expert_batch.mask)):
        list_of_masks.append(torch.Tensor(expert_batch.mask[i]))
    expert_masks = torch.cat(list_of_masks, 0).type(dtype)

    ## Expand expert states ##
    expanded_expert_states = -1*torch.ones(
        expert_states.size(0),
        self.history_size*expert_states.size(1)).type(dtype)
    for i in range(expert_states.size(0)):
      expanded_expert_states[i,:expert_states.size(1)] = expert_states[i,:]
      if i > 0:
        expanded_expert_states[i, expert_states.size(1):] = \
            expanded_expert_states[i-1,
                :(self.history_size-1)*expert_states.size(1)]

    ## set back to expert_state
    expert_states = expanded_expert_states

    ## Extract expert latent variables ##
    for i in range(expert_states.size(0)):
      if i == 0 or expert_masks[i-1] == 0:
        continue
      mu, sigma = vae_model.encode(Variable(expert_states[i-1]),
                                   Variable(expert_latent_c[i-1]))
      expert_latent_c[i] = vae_model.reparameterize(mu, logvar)


    returns = torch.Tensor(actions.size(0), 1).type(dtype)
    deltas = torch.Tensor(actions.size(0), 1).type(dtype)
    advantages = torch.Tensor(actions.size(0), 1).type(dtype)

    # compute advantages
    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        # advantages[i] = deltas[i] + \
        #   args.gamma * args.tau * prev_advantage * masks[i]

        # Changed by me to see if value function is causing problems
        advantages[i] = returns[i]
        prev_return = returns[i, 0]

        # prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    advantages = (advantages - advantages.mean()) / advantages.std()

    # Backup params after computing probs but before updating new params
    # policy_net.backup()
    for old_policy_param, policy_param in zip(old_policy_net.parameters(),
                                              policy_net.parameters()):
      old_policy_param.data.copy_(policy_param.data)

    # update value, reward and policy networks
    optim_iters = self.args.batch_size // optim_batch_size
    optim_batch_size_exp = expert_actions.size(0) // optim_iters

    for _ in range(optim_epochs):
      perm = np.arange(actions.size(0))
      np.random.shuffle(perm)
      perm = torch.LongTensor(perm).type(dtype_Long)
      states = states[perm]
      actions = actions[perm]
      latent_c = latent_c[perm]
      latent_next_c = latent_next_c[perm]
      #values = values[perm]
      targets = targets[perm]
      advantages = advantages[perm]
      perm_exp = np.arange(expert_actions.size(0))
      np.random.shuffle(perm_exp)
      perm_exp = torch.LongTensor(perm_exp).type(dtype_Long)
      expert_states = expert_states[perm_exp]
      expert_actions = expert_actions[perm_exp]
      expert_latent_c = expert_latent_c[perm_exp]
      cur_id = 0
      cur_id_exp = 0

      for _ in range(optim_iters):
        cur_batch_size = min(optim_batch_size, actions.size(0) - cur_id)
        cur_batch_size_exp = min(optim_batch_size_exp,
                                 expert_actions.size(0) - cur_id_exp)
        state_var = Variable(states[cur_id:cur_id+cur_batch_size])
        action_var = Variable(actions[cur_id:cur_id+cur_batch_size])
        latent_c_var = Variable(latent_c[cur_id:cur_id+cur_batch_size])
        latent_next_c_var = Variable(
                latent_next_c[cur_id:cur_id+cur_batch_size])
        advantages_var = Variable(advantages[cur_id:cur_id+cur_batch_size])
        expert_state_var = Variable(
                expert_states[cur_id_exp:cur_id_exp+cur_batch_size_exp])
        expert_action_var = Variable(
                expert_actions[cur_id_exp:cur_id_exp+cur_batch_size_exp])
        expert_latent_c_var = Variable(
                expert_latent_c[cur_id_exp:cur_id_exp+cur_batch_size_exp])


        # update reward net
        opt_reward.zero_grad()

        # backprop with expert demonstrations
        o = reward_net(torch.cat((expert_state_var,
                                  expert_action_var,
                                  expert_latent_c_var),1))
        loss = criterion(o, Variable(torch.zeros(
            expert_action_var.size(0),1).type(dtype)))
        loss.backward()

        # backprop with generated demonstrations
        o = reward_net(torch.cat((state_var, action_var, latent_c_var),1))
        loss = criterion(o, Variable(
            torch.ones(action_var.size(0),1)).type(dtype))
        loss.backward()

        opt_reward.step()

        # update posterior net # We need to do this by reparameterization
        # trick instead.  We should not put action_var (a_t) in the
        # posterior net since we need c_t to predict a_t while till now we
        # only have c_{t-1}.
        mu, _ = posterior_net(torch.cat((state_var,
                                         action_var,
                                         latent_c_var), 1))

        loss = criterion_posterior(mu, latent_next_c)
        loss.backward()

        # compute old and new action probabilities
        action_means, action_log_stds, action_stds = policy_net(
                torch.cat((state_var, latent_c_var), 1))
        log_prob_cur = normal_log_density(action_var,
                                          action_means,
                                          action_log_stds,
                                          action_stds)

        action_means_old, action_log_stds_old, action_stds_old = \
                old_policy_net(torch.cat((state_var, latent_c_var), 1))
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

        # update policy net
        opt_policy.zero_grad()
        ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
        surr1 = ratio * advantages_var[:,0]
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) \
                * advantages_var[:,0]
        policy_surr = -torch.min(surr1, surr2).mean()
        policy_surr.backward()
        torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
        opt_policy.step()

        # set new starting point for batch
        cur_id += cur_batch_size
        cur_id_exp += cur_batch_size_exp

  def train_gail(self, expert):
    '''Train GAIL.'''
    args = self.args
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
        episode_len = len(state_expert)

        # Generate c from trained VAE
        c_gen = self.get_c_for_traj(state_expert, action_expert, c_expert)

        # Sample start state
        curr_state_obj = sample_start_state()
        curr_state_feat = self.get_state_features(curr_state_obj,
                                                  self.args.use_state_features)
        #state = running_state(state)

        # TODO: Make this a separate function. Can be parallelized.
        #memory = Memory()
        ep_reward = 0
        for t in range(episode_len):
          ct = c_gen[t, :]
          action = select_action(np.concatenate((curr_state_feat, ct)))
          action = epsilon_greedy_linear_decay(action.data.cpu().numpy(),
                                               args.num_episodes * 0.5,
                                               ep_idx,
                                               low=0.05,
                                               high=0.3)

          # Get the discriminator reward
          reward = -float(self.reward_net(torch.cat(
            (Variable(torch.from_numpy(s.state).unsqueeze(0)).type(dtype),
              Variable(torch.from_numpy(oned_to_onehot(
                action)).unsqueeze(0)).type(dtype),
              Variable(torch.from_numpy(ct).unsqueeze(0)).type(dtype)),
             1)).data.cpu().numpy()[0,0])

          if t < args.max_ep_length-1:
            mu, sigma = self.posterior_net(
                torch.cat((
                  Variable(torch.from_numpy(s.state).unsqueeze(0)).type(dtype),
                  Variable(torch.from_numpy(oned_to_onehot(
                    action)).unsqueeze(0)).type(dtype),
                  Variable(torch.from_numpy(ct).unsqueeze(0)).type(dtype)), 1))

            mu = mu.data.cpu().numpy()[0,0]
            sigma = sigma.data.cpu().numpy()[0,0]

            # should ideally be logpdf, but pdf may work better. Try both.
            next_ct = c_gen[t+1,:]
            reward += norm.pdf(np.argmax(next_ct), loc=mu, scale=sigma)

            # Also, argmax for now, but has to be c[t+1, 1:]
            # when reverting to proper c ...

            #reward += math.exp(np.sum(np.multiply(
            # posterior_net(torch.cat((
            #   Variable(torch.from_numpy(
            #       s.state).unsqueeze(0)).type(dtype),
            #   Variable(torch.from_numpy(
            #       oned_to_onehot(action)).unsqueeze(0)).type(dtype),
            #   Variable(torch.from_numpy(ct).unsqueeze(0)).type(
            #     dtype)),1)).data.cpu().numpy()[0,:], c[t+1,:])))


          ep_reward += reward

          next_state_obj = self.transition_func(state, Action(action), 0)
          next_state_feat = self.get_state_features(
              next_state_feat, self.args.use_state_features)
          #next_state = running_state(next_state)
          mask = 0 if t == args.max_ep_length - 1 else 1

          # Push to memory
          memory.push(curr_state_feat,
                      np.array([oned_to_onehot(action)]),
                      mask,
                      next_state_features,
                      reward,
                      ct,
                      next_ct)

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
      expert_batch = expert.sample(size=args.num_expert_trajs)
      update_params(gen_batch, expert_batch, ep_idx,
                    args.optim_epochs, args.optim_batch_size)

      if ep_idx > 0 and  ep_idx % args.log_interval == 0:
        print('Episode [{}/{}]   Last R: {:.2f}   Avg R: {:.2f} \t' \
              'Last true R {:.2f}   Avg true R: {:.2f}'.format(
              ep_idx, args.num_episodes, reward_sum, reward_batch/num_episodes,
              true_reward_sum, true_reward_batch/num_episodes))

      results_path = os.path.join(args.checkpoint, 'results.pkl')
      with open(results_path, 'wb') as results_f:
        pickle.dump((stats), results_f, protocol=2)

      if ep_idx > 0 and ep_idx % args.save_interval == 0:
        checkpoint_filepath = self.model_checkpoint_filepath(ep_idx)
        torch.save(self.checkpoint_data_to_save(), checkpoint_filepath)
        print("Did save checkpoint: {}".format(checkpoint_filepath))


def main(args):
  expert = Expert(args.expert_path, num_inputs)
  print('Loading expert trajectories ...')
  expert.push()
  print('Expert trajectories loaded.')

  dtype = torch.FloatTensor
  if args.cuda:
    dtype = torch.cuda.FloatTensor

  causal_gail_mlp = CausalGAILMLP(
      args,
      vae_model,
      env_data,
      state_size=args.state_size,
      action_size=args.action_size,
      num_goals=4,
      history_size=args.history_size,
      dtype=dtype)
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
  parser.add_argument('--action_size', type=int, default=4,
                      help='Action size for VAE.')
  parser.add_argument('--history-size', type=int, default=1,
                        help='State history size to use in VAE.')
  parser.add_argument('--vae-context-size', type=int, default=1,
                      help='Context size for VAE.')

  # RL parameters
  parser.add_argument('--gamma', type=float, default=0.99,
                      help='discount factor (default: 0.99)')
  parser.add_argument('--tau', type=float, default=0.95,
                      help='gae (default: 0.95)')

  # Training parameters
  parser.add_argument('--learning-rate', type=float, default=3e-4,
                      help='gae (default: 3e-4)')
  parser.add_argument('--batch-size', type=int, default=2048,
                      help='batch size (default: 2048)')
  parser.add_argument('--num-episodes', type=int, default=500,
                      help='number of episodes (default: 500)')
  parser.add_argument('--max-ep-length', type=int, default=1000,
                      help='maximum episode length (default: 6)')

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

  parser.add_argument('--vae_weights', type=str, required=True,
                      help='path to checkpoint')
  parser.add_argument('--checkpoint', type=str, required=True,
                      help='path to checkpoint')

  # Use features
  parser.add_argument('--use_state_features', dest='use_state_features',
                      action='store_true',
                      help='Use features instead of direct (x,y) values in VAE')
  parser.add_argument('--no-use_state_features', dest='use_state_features',
                      action='store_false',
                      help='Do not use features instead of direct (x,y) ' \
                          'values in VAE')
  parser.set_defaults(use_state_features=False)

  args = parser.parse_args()
  torch.manual_seed(args.seed)

  if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)

  main(args)
