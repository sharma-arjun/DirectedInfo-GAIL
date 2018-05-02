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

from models import DiscretePosterior, Policy, Posterior, Reward, Value
from grid_world import State, Action, TransitionFunction
from grid_world import RewardFunction, RewardFunction_SR2, GridWorldReward
from grid_world import ActionBasedGridWorldReward
from grid_world import create_obstacles, obstacle_movement, sample_start
from load_expert_traj import Expert, ExpertHDF5
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
        super(InfoGAIL, self).__init__(args,
                                       logger,
                                       state_size=state_size,
                                       action_size=action_size,
                                       context_size=context_size,
                                       num_goals=num_goals,
                                       history_size=history_size,
                                       dtype=dtype)

        # Create networks
        self.policy_net = Policy(
                state_size=state_size * history_size,
                action_size=0,
                latent_size=context_size,
                output_size=action_size,
                hidden_size=64,
                output_activation='sigmoid')
        self.old_policy_net = Policy(
                state_size=state_size * history_size,
                action_size=0,
                latent_size=context_size,
                output_size=action_size,
                hidden_size=64,
                output_activation='sigmoid')

        # Use value network for calculating GAE. We should use this for
        # training the policy network.
        if args.use_value_net:
            # context_size contains num_goals
            self.value_net = Value(state_size * history_size + context_size,
                                   hidden_size=64)

        # Reward net is the discriminator network. Discriminator does not
        # receive the latent vector in InfoGAIL.
        self.reward_net = Reward(state_size * history_size,
                                 action_size,       # action size
                                 0,                 # latent size
                                 hidden_size=64)

        self.posterior_net = DiscretePosterior(
                state_size=state_size * history_size,   # state
                action_size=0,                          # action
                latent_size=0,                          # context
                hidden_size=64,
                output_size=num_goals)

        self.opt_policy = optim.Adam(self.policy_net.parameters(), lr=0.0003)
        self.opt_reward = optim.Adam(self.reward_net.parameters(), lr=0.0003)
        self.opt_value = optim.Adam(self.value_net.parameters(), lr=0.0003)
        self.opt_posterior = optim.Adam(self.posterior_net.parameters(),
                                        lr=0.0003)

        # Create loss functions
        self.criterion = nn.BCELoss()
        self.criterion_posterior = nn.CrossEntropyLoss()

        self.create_environment()

    def checkpoint_data_to_save(self):
        return {
            'policy': self.policy_net,
            'value': self.value_net,
            'reward': self.reward_net,
            'posterior': self.posterior_net,
        }

    def load_checkpoint_data(self, checkpoint_path):
        assert os.path.exists(checkpoint_path), \
            'Checkpoint path does not exists {}'.format(checkpoint_path)
        checkpoint_data = torch.load(checkpoint_path)
        self.policy_net = checkpoint_data['policy']
        self.value_net = checkpoint_data['value']
        self.reward_net = checkpoint_data['reward']
        self.posterior_net = checkpoint_data['posterior']


    def update_params_for_batch(self,
                                states,
                                actions,
                                latent_c,
                                targets,
                                advantages,
                                expert_states,
                                expert_actions,
                                optim_batch_size,
                                optim_batch_size_exp,
                                optim_iters):
        '''Update parameters for one batch of data.

        Update the policy network, discriminator (reward) network and the
        posterior network here.
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
            advantages_var = Variable(advantages[start_idx:end_idx])

            start_idx, end_idx = curr_id_exp, curr_id_exp + curr_batch_size_exp
            expert_state_var = Variable(expert_states[start_idx:end_idx])
            expert_action_var = Variable(expert_actions[start_idx:end_idx])

            # Update reward net
            self.opt_reward.zero_grad()

            # Backprop with expert demonstrations
            expert_output = self.reward_net(torch.cat((expert_state_var,
                                                       expert_action_var), 1))
            expert_disc_loss = self.criterion(
                    expert_output,
                    Variable(torch.zeros(expert_action_var.size(0),
                        1).type(dtype)))
            expert_disc_loss.backward()

            # Backprop with generated demonstrations
            gen_output = self.reward_net(torch.cat((state_var,
                                                    action_var), 1))
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
            # trick.
            predicted_posterior = self.posterior_net(state_var)
            # There is no GOAL info in latent_c_var here.
            # TODO: This 0 and -1 stuff is not needed here. Confirm?
            true_posterior = torch.zeros(latent_c_var.size(0), 1).type(dtype)
            true_posterior[:, 0] = latent_c_var.data[:, -1]
            posterior_loss = self.criterion_posterior(predicted_posterior,
                                                      Variable(true_posterior))
            posterior_loss.backward()
            self.logger.summary_writer.add_scalar('loss/posterior',
                                                  posterior_loss.data[0],
                                                  self.gail_step_count)


            # compute old and new action probabilities
            action_means, action_log_stds, action_stds = self.policy_net(
                    torch.cat((state_var, latent_c_var), 1))
            log_prob_cur = normal_log_density(action_var,
                                              action_means,
                                              action_log_stds,
                                              action_stds)

            action_means_old, action_log_stds_old, action_stds_old = \
                    self.old_policy_net(torch.cat(
                        (state_var, latent_c_var), 1))
            log_prob_old = normal_log_density(action_var,
                                              action_means_old,
                                              action_log_stds_old,
                                              action_stds_old)

            if args.use_value_net:
                # update value net
                self.opt_value.zero_grad()
                value_var = self.value_net(
                        torch.cat((state_var, latent_c_var), 1))
                value_loss = (value_var - \
                        targets[curr_id:curr_id+curr_batch_size]).pow(2.).mean()
                value_loss.backward()
                self.opt_value.step()

            # Update policy net (PPO step)
            self.opt_policy.zero_grad()
            ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
            surr1 = ratio * advantages_var[:, 0]
            surr2 = torch.clamp(
                    ratio,
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
        values = None
        if args.use_value_net:
            values = self.value_net(Variable(torch.cat((states, latent_c), 1)))

        # expert trajectories
        list_of_expert_states, list_of_expert_action = [], []
        list_of_masks = []
        for i in range(len(expert_batch.state)):
            ## Expand expert states ##
            expanded_states = self.expand_states_numpy(expert_batch.state[i],
                                                       self.history_size)
            list_of_expert_states.append(torch.Tensor(expanded_states))
            list_of_expert_actions.append(torch.Tensor(expert_batch.action[i]))
            list_of_masks.append(torch.Tensor(expert_batch.mask[i]))

        expert_states = torch.cat(list_of_expert_states,0).type(dtype)
        expert_actions = torch.cat(list_of_expert_actions, 0).type(dtype)
        expert_masks = torch.cat(list_of_masks, 0).type(dtype)

        assert expert_states.size(0) == expert_actions.size(0), \
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
        for old_policy_param, policy_param in zip(
                self.old_policy_net.parameters(), self.policy_net.parameters()):
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
                    targets[perm],
                    advantages[perm],
                    expert_states[perm_exp],
                    expert_actions[perm_exp],
                    optim_batch_size,
                    optim_batch_size_exp,
                    optim_iters)


    def train_gail(self, expert):
        '''Train Info-GAIL.'''
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
                state_expert, action_expert, _, _ = traj_expert

                # Expert state and actions
                state_expert = state_expert[0]
                action_expert = action_expert[0]
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

                c_sampled = np.zeros((self.num_goals), dtype=np.float32)
                c_sampled[np.random.randint(0, self.num_goals)] = 1.0
                c_sampled_tensor = torch.zeros((1)).type(torch.LongTensor)
                c_sampled_tensor[0] = int(np.argmax(c_sampled))
                if self.args.cuda:
                    c_sampled_tensor = torch.cuda.LongTensor(c_sampled_tensor)

                memory_list = []
                for t in range(expert_episode_len):
                    action = self.select_action(
                            np.concatenate((curr_state, c_sampled)))
                    action_numpy = action.data.cpu().numpy()

                    # Save generated and true trajectories
                    true_traj.append((state_expert[t], action_expert[t]))
                    gen_traj.append((curr_state_obj.coordinates, action_numpy))
                    gen_traj_dict['features'].append(self.get_state_features(
                        curr_state_obj, self.args.use_state_features))
                    gen_traj_dict['actions'].append(action_numpy)
                    gen_traj_dict['c'].append(c_sampled)

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
                                action, self.action_size)).unsqueeze(0)).type(
                                    dtype)), 1)).data.cpu().numpy()[0,0])

                    if disc_reward_t < 1e-6:
                        disc_reward_t += 1e-6

                    disc_reward_t = -math.log(disc_reward_t) \
                            if args.use_log_rewards else -disc_reward_t
                    disc_reward += disc_reward_t

        
                    # Predict c given (x_t)
                    predicted_posterior = self.posterior_net(
                            Variable(torch.from_numpy(curr_state).unsqueeze(
                                0)).type(dtype))
                    posterior_reward_t = self.criterion_posterior(
                            predicted_posterior, c_sampled_tensor)
                    pdb.set_trace()

                    posterior_reward += (self.args.lambda_posterior *
                            posterior_reward_t)
                    pdb.set_trace()

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
                        c_sampled,
                        c_sampled])

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



                assert memory_list[-1][2] == 0, \
                        "Mask for final end state is not 0."
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
                    'gen_traj/reward',
                    {
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
                results['true_traj'][ep_idx] = copy.deepcopy(
                        true_traj_curr_episode)
                results['pred_traj'][ep_idx] = copy.deepcopy(
                        gen_traj_curr_episode)

            # Update parameters
            gen_batch = memory.sample()

            # We do not get the context variable from expert trajectories.
            # Hence we need to fill it in later.
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
                          np.max(true_reward_batch),
                          np.mean(expert_true_reward_batch)))

            results_path = os.path.join(args.results_dir, 'results.pkl')
            with open(results_path, 'wb') as results_f:
                pickle.dump((results), results_f, protocol=2)
                # print("Did save results to {}".format(results_path))

            if ep_idx % args.save_interval == 0:
                checkpoint_filepath = self.model_checkpoint_filepath(ep_idx)
                torch.save(self.checkpoint_data_to_save(), checkpoint_filepath)
                print("Did save checkpoint: {}".format(checkpoint_filepath))

def main(args):
    if not os.path.exists(os.path.join(args.results_dir, 'log')):
        os.makedirs(os.path.join(args.results_dir, 'log'))
    logger = TensorboardXLogger(os.path.join(args.results_dir, 'log'))

    expert = ExpertHDF5(args.expert_path, 2)
    print('Loading expert trajectories ...')
    expert.push(only_coordinates_in_state=True, one_hot_action=True)
    print('Expert trajectories loaded.')

    dtype = torch.FloatTensor
    if args.cuda:
        dtype = torch.cuda.FloatTensor

    info_gail_mlp = InfoGAIL(
            args,
            logger,
            state_size=args.state_size,
            action_size=args.action_size,
            context_size=4, # num_goals
            num_goals=4,
            history_size=args.history_size,
            dtype=dtype)
    info_gail_mlp.set_expert(expert)
    info_gail_mlp.train_gail(expert)


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

    main(args)
