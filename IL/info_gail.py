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
import torchvision.transforms as T
from torch.autograd import Variable

from models import Policy, Posterior, Reward, Value
from grid_world import State, Action, TransitionFunction
from grid_world import RewardFunction, RewardFunction_SR2, GridWorldReward
from grid_world import ActionBasedGridWorldReward
from grid_world import create_obstacles, obstacle_movement, sample_start
from load_expert_traj import Expert, ExpertHDF5
from replay_memory import Memory
from running_state import ZFilter
from utils.torch_utils import clip_grads

from base_gail import BaseGAIL
from vae import VAE, VAETrain
from utils.logger import Logger, TensorboardXLogger
from utils.rl_utils import epsilon_greedy_linear_decay, epsilon_greedy
from utils.rl_utils import greedy, oned_to_onehot, normal_log_density
from utils.rl_utils import get_advantage_for_rewards
from utils.torch_utils import get_weight_norm_for_network

class InfoGAIL(BaseGAIL):
    def __init__(self,
                 args,
                 logger,
                 state_size=2,
                 action_size=4,
                 context_size=1,
                 num_goals=4,
                 history_size=1,
                 dtype=torch.FloatTensor):
        super(self, InfoGAIL).__init__(args,
                                       logger,
                                       state_size=state_size,
                                       action_size=action_size,
                                       context_size=context_size,
                                       num_goals=num_goals,
                                       history_size=history_size,
                                       dtype=dtype)

        # Create networks
        self.policy_net = Policy(state_size * history_size,
                                 0,
                                 context_size,
                                 action_size,
                                 hidden_size=64,
                                 output_activation='sigmoid')
        self.old_policy_net = Policy(state_size * history_size,
                                     0,
                                     context_size,
                                     action_size,
                                     hidden_size=64,
                                     output_activation='sigmoid')

        # context_size contains num_goals
        self.value_net = Value(state_size * history_size + context_size ,
                               hidden_size=64)

        # Reward net is the discriminator network.
        self.reward_net = Reward(state_size * history_size,
                                 action_size,
                                 context_size,
                                 hidden_size=64)

        self.posterior_net = Posterior(state_size * history_size,
                                       0,
                                       context_size,
                                       hidden_size=64)

        self.opt_policy = optim.Adam(self.policy_net.parameters(), lr=0.0003)
        self.opt_reward = optim.Adam(self.reward_net.parameters(), lr=0.0003)
        self.opt_value = optim.Adam(self.value_net.parameters(), lr=0.0003)
        self.opt_posterior = optim.Adam(self.posterior_net.parameters(),
                                        lr=0.0003)

        # Create loss functions
        self.criterion = nn.BCELoss()
        #criterion_posterior = nn.NLLLoss()
        self.criterion_posterior = nn.MSELoss()

        self.create_environment()

    def checkpoint_data_to_save(self):
        return {
            'policy': self.policy_net,
            'value': self.value_net,
            'reward': self.reward_net
        }

    def load_checkpoint_data(self, checkpoint_path):
        assert os.path.exists(checkpoint_path), \
            'Checkpoint path does not exists {}'.format(checkpoint_path)
        checkpoint_data = torch.load(checkpoint_path)
        self.policy_net = checkpoint_data['policy']
        self.value_net = checkpoint_data['value']
        self.reward_net = checkpoint_data['reward']

    def train_gail(self, expert):
        '''Train GAIL.'''
        args, dtype = self.args, self.dtype
        self.train_step_count, self.gail_step_count = 0, 0
    
        for ep_idx in range(args.num_epochs):
            memory = Memory()

            num_steps = 0
            reward_batch, true_reward_batch = [], []
            expert_true_reward_batch = []
            true_traj_curr_episode, gen_traj_curr_episode = [], []

            while num_steps < args.batch_size:
                traj_expert = expert.sample(size=1)
                state_expert, action_expert, c_expert, _ = traj_expert

                # Expert state and actions
                state_expert = state_expert[0]
                action_expert = action_expert[0]
                c_expert = c_expert[0]
                expert_episode_len = len(state_expert)

                # Sample start state or should we just choose the start state
                # from the expert trajectory sampled above.
                # curr_state_obj = self.sample_start_state()
                curr_state_obj = State(state_expert[0], self.obstacles)
                curr_state_feat = self.get_state_features(
                        curr_state_obj, self.args.use_state_features)

                # Add history to state
                if args.history_size > 1:
                    curr_state = -1 * np.ones(
                            (args.history_size * curr_state_feat.shape[0]),
                            dtype=np.float32)
                    curr_state[(args.history_size-1) \
                            * curr_state_feat.shape[0]:] = curr_state_feat
                else:
                    curr_state = curr_state_feat


                # TODO: Make this a separate function. Can be parallelized.
                ep_reward, ep_true_reward, expert_true_reward = 0, 0, 0
                true_traj, gen_traj = [], []
                gen_traj_dict = {'features': [], 'actions': [],
                                 'c': [], 'mask': []}
                disc_reward, posterior_reward = 0.0, 0.0
                # Use a hard-coded list for memory to gather experience since we
                # need to mutate it before finally creating a memory object.

                memory_list = []
                for t in range(expert_episode_len):
                    action = self.select_action(
                            np.concatenate((curr_state, c_expert)))
                    action_numpy = action.data.cpu().numpy()

                    # Save generated and true trajectories
                    true_traj.append((state_expert[t], action_expert[t]))
                    gen_traj.append((curr_state_obj.coordinates, action_numpy))
                    gen_traj_dict['features'].append(self.get_state_features(
                        curr_state_obj, self.args.use_state_features))
                    gen_traj_dict['actions'].append(action_numpy)
                    gen_traj_dict['c'].append(ct)

                    action = epsilon_greedy_linear_decay(
                            action_numpy,
                            args.num_epochs * 0.5,
                            ep_idx,
                            self.action_size,
                            low=0.05,
                            high=0.3)

                    # Get the discriminator reward
                    disc_reward_t = float(self.reward_net(torch.cat(
                        (Variable(torch.from_numpy(
                            curr_state).unsqueeze(0)).type(dtype),
                        Variable(torch.from_numpy(
                            oned_to_onehot(
                                action, self.action_size)).unsqueeze(0)).type(dtype),
                        Variable(torch.from_numpy(
                            next_ct_array).unsqueeze(0)).type(
                                dtype)), 1)).data.cpu().numpy()[0,0])

                    if disc_reward_t < 1e-6:
                        disc_reward_t += 1e-6

                    if args.use_log_rewards:
                        disc_reward_t = -math.log(disc_reward_t)
                    else:
                        disc_reward_t = -disc_reward_t

                    disc_reward += disc_reward_t

                    # use norm.logpdf if flag else use norm.pdf
                    if args.use_log_rewards:
                        reward_func = norm.logpdf
                    else:
                        reward_func = norm.pdf

                    posterior_reward += posterior_reward_t

                    # Update Rewards
                    ep_reward += (disc_reward_t + posterior_reward_t)
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

                    # Update next state
                    next_state_obj = self.transition_func(
                            curr_state_obj, Action(action), 0)
                    next_state_feat = self.get_state_features(
                            next_state_obj, self.args.use_state_features)
                    #next_state = running_state(next_state)

                    mask = 0 if t == expert_episode_len - 1 else 1

                    # Push to memory
                    memory_list.append([
                        curr_state,
                        np.array([oned_to_onehot(action, self.action_size)]),
                        mask,
                        next_state_feat,
                        disc_reward_t + posterior_reward_t,
                        ct,
                        next_ct_array])

                    if args.render:
                        env.render()

                    if not mask:
                        break

                    curr_state_obj = next_state_obj
                    curr_state_feat = next_state_feat

                    if args.history_size > 1:
                        curr_state[:(args.history_size-1) \
                                * curr_state_feat.shape[0]] = \
                                curr_state[curr_state_feat.shape[0]:]
                        curr_state[(args.history_size-1) \
                                * curr_state_feat.shape[0]:] = curr_state_feat
                    else:
                        curr_state = curr_state_feat



            assert memory_list[-1][2] == 0, "Mask for final end state is not 0."
            for memory_t in memory_list:
              memory.push(*memory_t)

            self.logger.summary_writer.add_scalars(
                'gen_traj/gen_reward',
                {
                  'discriminator': disc_reward,
                  'posterior': posterior_reward,
                },
                self.train_step_count
            )

            num_steps += (t-1)
            reward_batch.append(ep_reward)
            true_reward_batch.append(ep_true_reward)
            expert_true_reward_batch.append(expert_true_reward)
            results['episode_reward'].append(ep_reward)

            # Append trajectories
            true_traj_curr_episode.append(true_traj)
            gen_traj_curr_episode.append(gen_traj)

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
            results['true_traj'][ep_idx] = copy.deepcopy(true_traj_curr_episode)
            results['pred_traj'][ep_idx] = copy.deepcopy(gen_traj_curr_episode)

          # Update parameters
          gen_batch = memory.sample()

          # We do not get the context variable from expert trajectories. Hence we
          # need to fill it in later.
          expert_batch = expert.sample(size=args.num_expert_trajs)

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

def main():
    if not os.path.exists(os.path.join(args.results_dir, 'log')):
        os.makedirs(os.path.join(args.results_dir, 'log'))
    logger = TensorboardXLogger(os.path.join(args.results_dir, 'log'))

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
            logger,
            state_size=args.state_size,
            action_size=args.action_size,
            context_size=4 + args.context_size,  # num_goals + context_size
            num_goals=4,
            history_size=args.history_size,
            dtype=dtype)
    causal_gail_mlp.set_expert(expert)
    causal_gail_mlp.train_gail(expert)

    if args.init_from_vae:
        causal_gail_mlp.load_weights_from_vae()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Info-GAIL using MLP.')
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
