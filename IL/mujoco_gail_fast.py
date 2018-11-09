import argparse
import copy
import sys
import os
import pdb
import pickle
import math
import random
import gym

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
from grid_world import create_obstacles, obstacle_movement, sample_start
from load_expert_traj import Expert, ExpertHDF5, SeparateRoomTrajExpert
from utils.replay_memory import Memory
from utils.torch_utils import clip_grads, clip_grad_value

from base_gail import BaseGAIL
from vae import VAE, VAETrain
from utils.logger import Logger, TensorboardXLogger
from utils.rl_utils import epsilon_greedy_linear_decay, epsilon_greedy
from utils.rl_utils import greedy, oned_to_onehot
from utils.rl_utils import get_advantage_for_rewards
from utils.torch_utils import get_weight_norm_for_network
from utils.torch_utils import normal_log_density
from utils.torch_utils import add_scalars_to_summary_writer

from multiprocessing import Pool
import time

def unwrap_self_f(arg, **kwarg):
    return C.f(*arg, **kwarg)

class C:
    def f(self, name):
        print('hello {},'.format(name))
        time.sleep(5)
        print('nice to meet you.')

    def run(self):
        pool = Pool(processes=2)
        names = ('frank', 'justin', 'osi', 'thomas')
        pool.map(unwrap_self_f, zip([self]*len(names), names))


def get_c_for_traj(vae_model, env, args, state_arr, action_arr, c_arr,
                   history_size, env_type, num_goals, use_discrete_vae, dtype):
    '''Get c[1:T] for given trajectory.'''
    batch_size, episode_len = state_arr.shape[0], state_arr.shape[1]
    c_arr = np.array(c_arr, dtype=np.int32)

    true_goal_numpy = np.zeros((c_arr.shape[0], num_goals))
    true_goal_numpy[np.arange(c_arr.shape[0]), c_arr[:, 0]] = 1
    true_goal = Variable(torch.from_numpy(true_goal_numpy).type(dtype))

    action_var = Variable(torch.from_numpy(action_arr).type(dtype))

    # Context output from the VAE encoder
    pred_c_arr = -1 * np.ones((
        batch_size,
        episode_len + 1,
        vae_model.posterior_latent_size))

    if env_type == 'mujoco':
        x_feat = state_arr[:, 0, :]
        dummy_state = env.reset()
        if 'Hopper' in args.env_name:
            env.env.set_state(np.concatenate(
                (np.array([0.0]), x_feat[0, :5]), axis=0), x_feat[0, 5:])
        elif 'Walker' in args.env_name:
            env.env.set_state(np.concatenate(
                (np.array([0.0]), x_feat[0, :8]), axis=0), x_feat[0, 8:17])
        else:
            raise ValueError("Incorrect env name for mujoco")

        dummy_state = x_feat
    elif env_type == 'gym':
        x_feat = state_arr[:, 0, :]
        dummy_state = env.reset()
        theta = (np.arctan2(x_feat[:, 1], x_feat[:, 0]))[:, np.newaxis]
        theta_dot = (x_feat[:, 2])[:, np.newaxis]
        env.env.state = np.concatenate((theta, theta_dot), axis=1).reshape(-1)
        if args.use_time_in_state:
            x_feat = np.concatenate(
                    [x_feat, np.zeros((state_arr.shape[0], 1))], axis=1)
    else:
        raise ValueError('Incorrect env type: {}'.format(args.env_type))

    # x is (N, F)
    x = x_feat

    # Add history to state
    if history_size > 1:
        x_hist = -1 * np.ones((x.shape[0], history_size, x.shape[1]),
                              dtype=np.float32)
        x_hist[:, history_size - 1, :] = x_feat
        x = x_hist

    for t in range(episode_len):
        c = pred_c_arr[:, t, :]
        x_var = Variable(torch.from_numpy(
            x.reshape((batch_size, -1))).type(dtype))

        c_var = Variable(torch.from_numpy(c).type(dtype))
        if len(true_goal.size()) == 2:
            c_var = torch.cat([true_goal, c_var], dim=1)
        elif len(true_goal.size()) == 3:
            c_var = torch.cat([true_goal[:, t, :], c_var], dim=1)
        else:
            raise ValueError("incorrect true goal size")

        # copied code from vae.py get_context_at_state
        if use_discrete_vae:
            logits = vae_model.encode(x_var, c_var)
            c_next = vae_model.reparameterize(logits, vae_model.temperature)
        else:
            mu, logvar = vae_model.encode(x_var, c_var)
            c_next = vae_model.reparameterize(mu, logvar)

        pred_c_arr[:, t+1, :] = c_next.data.cpu().numpy()

        if history_size > 1:
            x_hist[:, :(history_size-1), :] = x_hist[:, 1:, :]

    return pred_c_arr

def get_history_features(state, use_velocity=False):
    '''
    state: Numpy array of shape (N, H, F)
    '''
    if use_velocity:
        _, history_size, state_size = state.shape
        new_state = np.zeros(state.shape)

        state_int = np.array(state, dtype=np.int32)
        for t in range(history_size-1):
            minus_one_idx = state_int[:, t, 0] == -1
            new_state[minus_one_idx, t, :] = 0.0
            one_idx = (state_int[:, t, 0] != -1)
            new_state[one_idx, t, :] = state[one_idx, t+1, :] - state[one_idx, t, :]

        new_state[:, history_size-1, :] = state[:, history_size-1, :]

        return new_state
    else:
        return state

def get_context_at_state(vae_model, x, c, use_discrete_vae=True):
    '''Get context variable c_t for given x_t, c_{t-1}.

    x: State at time t. (x_t)
    c: Context at time t-1. (c_{t-1})
    '''
    if use_discrete_vae:
        logits = vae_model.encode(x, c)
        return vae_model.reparameterize(logits, vae_model.temperature)
    else:
        mu, logvar = vae_model.encode(x, c)
        return vae_model.reparameterize(mu, logvar)

def select_action(policy_net, x_var, c_var, goal_var, use_goal_in_policy=False,
                  train=False):
    # Select action using policy net.
    if use_goal_in_policy:
        inp_var = torch.cat((x_var, goal_var), dim=1)
    else:
        inp_var = torch.cat((x_var, c_var), dim=1)
    action_mean, action_log_std, action_std = policy_net(inp_var)
    if train:
        action = torch.normal(action_mean, action_std)
    else:
        action = action_mean
    return action

def get_discriminator_reward(reward_net, x, a, c, next_c,
                             dtype=torch.cuda.FloatTensor,
                             goal_var=None,
                             disc_reward_type='log_d',
                            ):
    '''Get discriminator reward.'''
    if goal_var is not None:
        next_c = torch.cat([goal_var, next_c], dim=1)

    disc_reward = float(reward_net(torch.cat(
        (x,
        Variable(torch.from_numpy(a)).type(dtype),
        goal_var), 1)).data.cpu().numpy()[0,0])

    if disc_reward_type == 'log_d':
        if disc_reward < 1e-8:
            disc_reward += 1e-8
        disc_reward = -math.log(disc_reward)
    elif disc_reward_type == 'log_1-d':
        if disc_reward >= 1.0:
            disc_reward = 1.0 - 1e-8
        disc_reward = math.log(1 - disc_reward)
    elif disc_reward_type == 'no_log':
        disc_reward = -disc_reward
    else:
        raise ValueError("Incorrect Disc reward type: {}".format(disc_reward_type))
    return disc_reward

def get_posterior_reward(posterior_net, x, c, next_ct,
                         use_discrete_vae=True, use_reparameterize=True,
                         goal_var=None):
    '''Get posterior reward.'''
    if goal_var is not None:
        c = torch.cat([goal_var, c], dim=1)

    if use_discrete_vae:
        logits = posterior_net(torch.cat((x, c), 1))
        _, label = torch.max(next_ct, 1)
        posterior_reward_t = F.cross_entropy(logits, label)
    else:
        mu, sigma = posterior_net(torch.cat((x, c), 1))
        mu = mu.data.cpu().numpy()[0,0]
        sigma = np.exp(0.5 * sigma.data.cpu().numpy()[0,0])
        # TODO: should ideally be logpdf, but pdf may work better. Try both.
        # use norm.logpdf if flag else use norm.pdf
        use_log_rewards = True
        reward_func = norm.logpdf if use_log_rewards else norm.pdf
        scale = sigma if use_reparameterize else 0.1
        # use fixed std if not using reparameterize otherwise use sigma.
        posterior_reward_t = reward_func(next_ct, loc=mu, scale=0.1)[0]

    return posterior_reward_t.data.cpu().numpy()[0]


def run_agent_worker(run_args,
                     vae_model,
                     policy_net,
                     reward_net,
                     posterior_net,
                     expert,
                     env,
                     cached_expert_c_list,
                     gen_batch_size,
                     posterior_latent_size,
                     num_goals,
                     use_rnn_goal_predictor,
                     use_discrete_vae,
                     dtype,
                     train,
                     sample_c_from_expert):
    '''Collect experience for agent.
    env: Environment to collect samples from.
    batch_size: Number of example instances to be collected. Int.
    max_episode_len: Maximum episode length for each episode. Int.
    config: Agent config.
    volatile: True if input state variable should be volatile else False.
    Return: Dictionary with `memory_list` and `log_list` keys.
    '''
    memory_list, log_list = [], []
    num_steps, batch_size = 0, 1

    while num_steps < gen_batch_size:
        # Sample trajectory expert.sample()
        traj_expert, traj_expert_indices = expert.sample(size=batch_size,
                                                         return_indices=True)

        state_expert, action_expert, c_expert, _ = traj_expert
        state_expert = np.array(state_expert, dtype=np.float32)
        action_expert = np.array(action_expert, dtype=np.float32)
        c_expert = np.array(c_expert, dtype=np.int32)

        expert_episode_len = state_expert.shape[1]

        # Generate c from trained VAE
        if sample_c_from_expert:
            pred_c_tensor = np.array([np.vstack((
                -1 * np.ones((1, cached_expert_c_list[0].shape[1])),
                cached_expert_c_list[i])) for i in traj_expert_indices])
            # We have already computed this, don't need to call again.
            # Saves a LOT of computation!!
            # pred_c_tensor = get_c_for_traj(
                    # vae_model, env, run_args, state_expert, action_expert,
                    # c_expert, run_args.history_size, run_args.env_type,
                    # num_goals, use_discrete_vae, dtype)
            
            pred_c_tensor = torch.from_numpy(pred_c_tensor).type(dtype)
        else:
            pred_c_tensor = -1 * torch.ones((
                batch_size,
                max(expert_episode_len, run_args.max_ep_length) + 1, # + 1 for c[-1]
                posterior_latent_size)).type(dtype)

        true_goal_numpy = np.zeros((c_expert.shape[0], num_goals))
        true_goal_numpy[np.arange(c_expert.shape[0]), c_expert[:, 0]] = 1
        true_goal = Variable(torch.from_numpy(true_goal_numpy)).type(dtype)

        if run_args.use_random_starts:
            ### if using random states ###
            dummy_state = env.reset()
            x_feat = np.concatenate((dummy_state, np.array([0.0])),
                                     axis=0)[np.newaxis, :]
        else:
            ### use start state from data ###
            if run_args.env_type == 'mujoco':
                x_feat = state_expert[:, 0, :]
                dummy_state = env.reset()
                if 'Hopper' in run_args.env_name:
                    env.env.set_state(np.concatenate(
                                (np.array([0.0]), x_feat[0, :5]), axis=0), x_feat[0, 5:])
                elif 'Walker' in run_args.env_name:
                    env.env.set_state(np.concatenate(
                                (np.array([0.0]), x_feat[0, :8]), axis=0), x_feat[0, 8:17])
                else:
                    raise ValueError("Incorrect env name for mujoco")
                dummy_state = x_feat
            elif run_args.env_type == 'gym':
                x_feat = state_expert[:, 0, :]
                dummy_state = env.reset()
                theta = (np.arctan2(x_feat[:, 1], x_feat[:, 0]))[:, np.newaxis]
                theta_dot = (x_feat[:, 2])[:, np.newaxis]
                env.env.state = np.concatenate(
                        (theta, theta_dot), axis=1).reshape(-1)
                x_feat = np.concatenate(
                        [x_feat, np.zeros((state_expert.shape[0], 1))], axis=1)
            else:
                raise ValueError("Incorrect env type: {}".format(
                    run_args.env_type))

        x = x_feat
        curr_state_arr = x_feat

        # Add history to state
        if run_args.history_size > 1:
            x_hist = -1 * np.ones((x.shape[0], run_args.history_size, x.shape[1]),
                                  dtype=np.float32)
            x_hist[:, (run_args.history_size-1), :] = x_feat
            x = get_history_features(x_hist)

        ep_reward, expert_true_reward = 0, 0
        env_reward_dict = {'true_reward': 0.0}
        disc_reward, posterior_reward = 0.0, 0.0

        memory = Memory()
        for t in range(expert_episode_len):
            # First predict next ct
            ct = pred_c_tensor[:, t, :]

            # Get state and context variables
            x_var = Variable(torch.from_numpy(x.reshape((batch_size, -1))).type(
                dtype))
            # Append 'c' at the end.
            if use_rnn_goal_predictor:
                raise ValueError("Do not use this.")
            else:
                c_var = Variable(ct)
                if len(true_goal.size()) == 2:
                    c_var = torch.cat([true_goal, c_var], dim=1)
                elif len(true_goal.size()) == 3:
                    c_var = torch.cat([true_goal[:, t, :], c_var], dim=1)
                else:
                    raise ValueError("incorrect true goal size")

            if sample_c_from_expert:
                next_ct = Variable(pred_c_tensor[:, t+1, :])
            else:
                next_ct = get_context_at_state(vae_model, x_var, c_var,
                                               use_discrete_vae=use_discrete_vae)
                pred_c_tensor[:, t+1, :] = next_ct.data

            # Reassign correct c_var and next_c_var
            c_var, next_c_var = Variable(ct), next_ct

            # Get the goal variable (either true or predicted)
            goal_var = None
            if len(true_goal.size()) == 2:
                goal_var = true_goal
            elif len(true_goal.size()) == 3:
                goal_var = true_goal[:, t, :]
            else:
                raise ValueError("incorrect true goal size")
            # ==== END ====


            # Generator should predict the action using (x_t, c_t)
            # Note c_t is next_c_var.
            action = select_action(policy_net, x_var, next_c_var, goal_var,
                                   use_goal_in_policy=run_args.use_goal_in_policy,
                                   train=train)
            action_numpy = action.data.cpu().numpy()
            action = action_numpy

            # Get the discriminator reward
            disc_reward_t = get_discriminator_reward(
                            reward_net, x_var, action, c_var, next_c_var,
                            dtype=dtype,
                            goal_var=goal_var,
                            disc_reward_type=run_args.disc_reward)
            disc_reward += disc_reward_t

            # Get posterior reward
            posterior_reward_t = 0.0
            if not run_args.use_goal_in_policy:
                posterior_reward_t = run_args.lambda_posterior \
                        * get_posterior_reward(
                                posterior_net,
                                x_var, c_var,
                                next_c_var,
                                use_discrete_vae=use_discrete_vae,
                                use_reparameterize=args.use_reparameterize,
                                goal_var=goal_var)
            posterior_reward += posterior_reward_t

            # Update Rewards, do not use posterior reward during test.
            if train:
                ep_reward += (disc_reward_t + posterior_reward_t)
            else:
                ep_reward += disc_reward_t

            next_state_feat, true_reward, done, _ = env.step(action.reshape(-1))
            env_reward_dict['true_reward'] += true_reward
            if run_args.use_time_in_state:
                next_state_feat = np.concatenate(
                        (next_state_feat,
                        np.array([(t+1)/(expert_episode_len+1)])), axis=0)
            # next_state_feat = running_state(next_state_feat)
            x[:] = next_state_feat.copy()

            mask = 0 if t == expert_episode_len - 1 or done else 1

            # Push to memory
            memory.push(
                    curr_state_arr.copy().reshape(-1),
                    action,
                    mask,
                    next_state_feat,
                    disc_reward_t + posterior_reward_t,
                    ct.cpu().numpy().reshape(-1),
                    next_ct.data.cpu().numpy().reshape(-1),
                    goal_var.data.cpu().numpy().copy().reshape(-1)
                    )

            num_steps += 1

            if run_args.render:
                env.render()

            if mask == 0:
                break

            # Update current state
            curr_state_arr = np.array(next_state_feat, dtype=np.float32)

        memory_list.append(memory)
        log_list.append({
            'total_disc_reward': ep_reward,
            'disc_reward': disc_reward,
            'posterior_reward': posterior_reward,
            'true_reward': env_reward_dict['true_reward'],
            })

    return {
        'memory_list': memory_list,
        'log_list': log_list,
    }

def parallel_run_agent_worker(args):
    return run_agent_worker(*args)


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

        if args.use_goal_in_policy:
            policy_latent_size = num_goals
        else:
            policy_latent_size = vae_train.vae_model.policy.latent_size


        self.policy_net = Policy(
                state_size=policy1_state_size,
                action_size=vae_train.vae_model.policy.action_size,
                latent_size=policy_latent_size,
                output_size=vae_train.vae_model.policy.output_size,
                output_activation=None)

        self.old_policy_net = Policy(
                state_size=policy1_state_size,
                action_size=vae_train.vae_model.policy.action_size,
                latent_size=policy_latent_size,
                output_size=vae_train.vae_model.policy.output_size,
                output_activation=None)

        if args.use_value_net:
            # context_size contains num_goals
            self.value_net = Value(state_size * history_size + num_goals,
                                   hidden_size=64)

        # Reward net is the discriminator network.
        self.reward_net = Reward(state_size * history_size,
                                 action_size,
                                 num_goals,
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

        self.opt_policy = optim.Adam(self.policy_net.parameters(),
                                     lr=args.gen_learning_rate)
        self.opt_reward = optim.Adam(self.reward_net.parameters(),
                                     lr=args.learning_rate)
        self.opt_posterior = optim.Adam(self.posterior_net.parameters(),
                                        lr=args.posterior_learning_rate)
        if args.use_value_net:
            self.opt_value = optim.Adam(self.value_net.parameters(),
                                        lr=args.learning_rate)

        self.transition_func, self.true_reward = None, None
        self.create_environment(args.env_type,
                                env_name=args.env_name,
                                num_threads=args.num_threads)
        self.expert = None
        self.obstacles, self.set_diff = None, None

    def convert_models_to_type(self, dtype):
        self.vae_train.convert_models_to_type(dtype)
        self.policy_net = self.policy_net.type(dtype)
        self.old_policy_net = self.old_policy_net.type(dtype)
        if self.value_net is not None:
            self.value_net = self.value_net.type(dtype)
        self.reward_net = self.reward_net.type(dtype)
        self.posterior_net = self.posterior_net.type(dtype)

    def create_environment(self, env_type, env_name=None, num_threads=1):
        assert(env_name is not None)
        self.env = gym.make(env_name)
        self.envs = []
        for i in range(num_threads):
            env = gym.make(env_name)
            if 'FetchPickAndPlace' in env_name:
                env = gym.wrappers.FlattenDictWrapper(env,
                        ['observation', 'desired_goal'])
            env.seed(int(time.time()) + i)
            self.envs.append(env)
        self.env_pool = Pool(num_threads)

    def select_action(self, x_var, c_var, goal_var, train=True):
        """Select action using policy net."""
        if self.args.use_goal_in_policy:
            inp_var = torch.cat((x_var, goal_var), dim=1)
        else:
            inp_var = torch.cat((x_var, c_var), dim=1)
        action_mean, action_log_std, action_std = self.policy_net(inp_var)
        if train:
            action = torch.normal(action_mean, action_std)
        else:
            action = action_mean
        return action

    def get_c_for_all_expert_trajs(self, expert):
        memory = expert.memory
        expert_c_list = []
        for i in range(len(memory)):
            curr_memory = memory[i]
            expert_c, _ = self.get_c_for_traj(
                    curr_memory.state[np.newaxis, :],
                    curr_memory.action[np.newaxis, :],
                    curr_memory.c[np.newaxis, :])
            # expert_c[0, :] is c_{-1} which does not map to s_0. Hence drop it.
            expert_c = expert_c.squeeze(0)[1:, :]
            expert_c_list.append(expert_c)
        return expert_c_list

    def get_precomputed_c_for_samples(self, size=5):
        ind = np.random.randint(len(self.cached_expert_c_list), size=size)
        batch_list = []
        for i in ind:
            batch_list.append(self.cached_expert_c_list[i])

        return batch_list

    def get_c_for_traj(self, state_arr, action_arr, c_arr):
        '''Get c[1:T] for given trajectory.'''
        batch_size, episode_len = state_arr.shape[0], state_arr.shape[1]
        history_size = self.history_size
        c_arr = np.array(c_arr, dtype=np.int32)

        # Use the Q-network (RNN) to predict goal.
        pred_goal = None
        if self.vae_train.use_rnn_goal_predictor:
            pred_goal, _ = self.vae_train.predict_goal(
                state_arr, action_arr, c_arr, None, self.num_goals)

        true_goal_numpy = np.zeros((c_arr.shape[0], self.num_goals))
        true_goal_numpy[np.arange(c_arr.shape[0]), c_arr[:, 0]] = 1
        true_goal = Variable(torch.from_numpy(true_goal_numpy).type(self.dtype))

        action_var = Variable(
                torch.from_numpy(action_arr).type(self.dtype))

        # Context output from the VAE encoder
        pred_c_arr = -1 * np.ones((
            batch_size,
            episode_len + 1,
            self.vae_train.vae_model.posterior_latent_size))

        if self.env_type == 'mujoco':
            x_feat = state_arr[:, 0, :]
            dummy_state = self.env.reset()
            if 'Hopper' in self.args.env_name:
                self.env.env.set_state(np.concatenate(
                    (np.array([0.0]), x_feat[0, :5]), axis=0), x_feat[0, 5:])
            elif 'Walker' in self.args.env_name:
                self.env.env.set_state(np.concatenate(
                    (np.array([0.0]), x_feat[0, :8]), axis=0), x_feat[0, 8:17])
            else:
                raise ValueError("Incorrect env name for mujoco")

            dummy_state = x_feat
        elif self.env_type == 'gym':
            x_feat = state_arr[:, 0, :]
            dummy_state = self.env.reset()
            theta = (np.arctan2(x_feat[:, 1], x_feat[:, 0]))[:, np.newaxis]
            theta_dot = (x_feat[:, 2])[:, np.newaxis]
            self.env.env.state = np.concatenate(
                    (theta, theta_dot), axis=1).reshape(-1)
            if self.args.use_time_in_state:
                x_feat = np.concatenate(
                        [x_feat, np.zeros((state_arr.shape[0], 1))], axis=1)
        else:
            raise ValueError('Incorrect env type: {}'.format(
                self.args.env_type))

        # x is (N, F)
        x = x_feat

        # Add history to state
        if self.history_size > 1:
            x_hist = -1 * np.ones((x.shape[0], history_size, x.shape[1]),
                                  dtype=np.float32)
            x_hist[:, history_size - 1, :] = x_feat
            x = self.vae_train.get_history_features(x_hist)

        for t in range(episode_len - 1):
            c = pred_c_arr[:, t, :]
            x_var = Variable(torch.from_numpy(
                x.reshape((batch_size, -1))).type(self.dtype))

            # Append 'c' at the end.
            if self.vae_train.use_rnn_goal_predictor:
                c_var = torch.cat([
                    final_goal,
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
                x_hist[:, history_size-1,  :] = state_arr[:, t+1, :]
                x = x_hist
            else:
                x = state_arr[:, t+1, :]

        return pred_c_arr, pred_goal

    def checkpoint_data_to_save(self):
        value_net_state_dict = None if self.value_net is None else \
                self.value_net.state_dict()
        return {
                'policy': self.policy_net.state_dict(),
                'posterior': self.posterior_net.state_dict(),
                'reward': self.reward_net.state_dict(),
                'value': value_net_state_dict,
        }

    def load_checkpoint_data(self, checkpoint_path):
        assert os.path.exists(checkpoint_path), \
                'Checkpoint path does not exists {}'.format(checkpoint_path)
        checkpoint_data = torch.load(checkpoint_path)
        self.policy_net.load_state_dict(checkpoint_data['policy'])
        self.posterior_net.load_state_dict(checkpoint_data['posterior'])
        self.reward_net.load_state_dict(checkpoint_data['reward'])
        if checkpoint_data.get('value') is not None:
            self.value_net.load_state_dict(checkpoint_data['value'])

    def load_weights_from_vae(self):
        # deepcopy from vae
        # self.policy_net = copy.deepcopy(self.vae_train.vae_model.policy)
        # self.old_policy_net = copy.deepcopy(self.vae_train.vae_model.policy)
        self.posterior_net = copy.deepcopy(self.vae_train.vae_model.posterior)

        # re-initialize optimizers
        # self.opt_policy = optim.Adam(self.policy_net.parameters(),
        #                              lr=self.args.gen_learning_rate)
        self.opt_posterior = optim.Adam(self.posterior_net.parameters(),
                                        lr=self.args.posterior_learning_rate)


    def set_expert(self, expert):
        assert self.expert is None, "Trying to set non-None expert"
        self.expert = expert
        assert expert.obstacles is not None, "Obstacles cannot be None"
        self.obstacles = expert.obstacles
        assert expert.set_diff is not None, "set_diff in h5 file cannot be None"
        self.set_diff = expert.set_diff

    def get_policy_learning_rate(self, epoch):
        '''Update policy learning rate schedule.'''
        if epoch < self.args.discriminator_pretrain_epochs:
            return 0.0
        else:
            return self.args.gen_learning_rate

    def get_discriminator_reward(self, x, a, c, next_c, goal_var=None):
        # Get discriminator reward.
        if goal_var is not None:
            next_c = torch.cat([goal_var, next_c], dim=1)

        disc_reward = float(self.reward_net(torch.cat(
            (x,
             Variable(torch.from_numpy(a)).type(self.dtype),
             goal_var), 1)).data.cpu().numpy()[0,0])

        if self.args.disc_reward == 'log_d':
            if disc_reward < 1e-8:
                disc_reward += 1e-8
            disc_reward = -math.log(disc_reward)
        elif self.args.disc_reward == 'log_1-d':
            if disc_reward >= 1.0:
                disc_reward = 1.0 - 1e-8
            disc_reward = math.log(1 - disc_reward)
        elif self.args.disc_reward == 'no_log':
            disc_reward = -disc_reward
        else:
            raise ValueError("Incorrect Disc reward type: {}".format(
                self.args.disc_reward))
        return disc_reward

    def get_posterior_reward(self, x, c, next_ct, goal_var=None):
        """Get posterior reward."""
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
            use_log_rewards = True
            reward_func = norm.logpdf if use_log_rewards else norm.pdf
            scale = sigma if self.args.use_reparameterize else 0.1
            # use fixed std if not using reparameterize otherwise use sigma.
            posterior_reward_t = reward_func(next_ct, loc=mu, scale=0.1)[0]

        return posterior_reward_t.data.cpu().numpy()[0]

    def update_posterior_net(self, state_var, c_var, next_c_var, goal_var=None):
        if goal_var is not None:
            c_var = torch.cat([goal_var, c_var], dim=1)

        if self.vae_train.args.use_discrete_vae:
            logits = self.posterior_net(torch.cat((state_var, c_var), 1))
            _, label = torch.max(next_c_var, 1)
            posterior_loss = F.cross_entropy(logits, label)
        else:
            mu, logvar = self.posterior_net(torch.cat((state_var, c_var), 1))
            posterior_loss = F.mse_loss(mu, next_c_var)
        return posterior_loss

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
                                optim_iters,
                                goal=None,
                                expert_goal=None):
        '''Update parameters for one batch of data.

        Update the policy network, discriminator (reward) network and the
        posterior network here.
        '''
        args, dtype = self.args, self.dtype
        curr_id, curr_id_exp = 0, 0
        for optim_idx in range(optim_iters):
            curr_batch_size = min(optim_batch_size, actions.size(0) - curr_id)
            curr_batch_size_exp = min(optim_batch_size_exp,
                                      expert_actions.size(0) - curr_id_exp)
            start_idx, end_idx = curr_id, curr_id + curr_batch_size

            state_var = Variable(states[start_idx:end_idx])
            action_var = Variable(actions[start_idx:end_idx])
            latent_c_var = Variable(latent_c[start_idx:end_idx])
            latent_next_c_var = Variable(latent_next_c[start_idx:end_idx])
            advantages_var = Variable(advantages[start_idx:end_idx])
            goal_var = None
            if goal is not None:
                goal_var = Variable(goal[start_idx:end_idx])

            start_idx, end_idx = curr_id_exp, curr_id_exp + curr_batch_size_exp
            expert_state_var = Variable(expert_states[start_idx:end_idx])
            expert_action_var = Variable(expert_actions[start_idx:end_idx])
            expert_latent_c_var = Variable(expert_latent_c[start_idx:end_idx])
            expert_goal_var = None
            if expert_goal is not None:
                expert_goal_var = Variable(expert_goal[start_idx:end_idx])

            # ==== Update reward net ====
            self.opt_reward.zero_grad()

            # Backprop with expert demonstrations
            expert_output = self.reward_net(
                    torch.cat((expert_state_var,
                               expert_action_var,
                               expert_goal_var), 1))
            expert_disc_loss = F.binary_cross_entropy(
                    expert_output,
                    Variable(torch.zeros(expert_action_var.size(0), 1)).type(
                        dtype))
            expert_disc_loss.backward()

            # Backprop with generated demonstrations
            # latent_next_c_var is actual c_t, latent_c_var is c_{t-1}
            gen_output = self.reward_net(
                    torch.cat((state_var,
                               action_var,
                               goal_var), 1))
            gen_disc_loss = F.binary_cross_entropy(
                    gen_output,
                    Variable(torch.ones(action_var.size(0), 1)).type(dtype))
            gen_disc_loss.backward()

            # clip_grad_value(self.reward_net.parameters(), 50)
            self.opt_reward.step()
            # ==== END ====

            # Add loss scalarsmemory.
            add_scalars_to_summary_writer(
                self.logger.summary_writer,
                'loss/discriminator',
                {
                  'total': expert_disc_loss.data[0] + gen_disc_loss.data[0],
                  'expert': expert_disc_loss.data[0],
                  'gen': gen_disc_loss.data[0],
                  },
                self.gail_step_count)
            reward_l2_norm, reward_grad_l2_norm = \
                              get_weight_norm_for_network(self.reward_net)
            self.logger.summary_writer.add_scalar('weight/discriminator/param',
                                                  reward_l2_norm,
                                                  self.gail_step_count)
            self.logger.summary_writer.add_scalar('weight/discriminator/grad',
                                                  reward_grad_l2_norm,
                                                  self.gail_step_count)

            # ==== Update posterior net ====
            #self.opt_posterior.zero_grad()
            #posterior_loss = self.update_posterior_net(state_var,
            #                                           latent_c_var,
            #                                           latent_next_c_var,
            #                                           goal_var=goal_var)
            #posterior_loss.backward()
            #self.opt_posterior.step()
            #self.logger.summary_writer.add_scalar('loss/posterior',
            #                                      posterior_loss.data[0],
            #                                      self.gail_step_count)
            # ==== END ====

            # compute old and new action probabilities
            if self.args.use_goal_in_policy:
                action_means, action_log_stds, action_stds = self.policy_net(
                        torch.cat((state_var, goal_var), 1))
                action_means_old, action_log_stds_old, action_stds_old = \
                        self.old_policy_net(
                                torch.cat((state_var, goal_var), 1))
            else:
                action_means, action_log_stds, action_stds = self.policy_net(
                        torch.cat((state_var, latent_next_c_var), 1))
                action_means_old, action_log_stds_old, action_stds_old = \
                        self.old_policy_net(
                                torch.cat((state_var, latent_next_c_var), 1))

            if self.vae_train.args.discrete_action:
                discrete_action_eps = 1e-10
                discrete_action_eps_var = Variable(torch.Tensor(
                    [discrete_action_eps])).type(self.dtype)
                # action_probs is (N, A)
                action_means = action_means + discrete_action_eps
                action_softmax = F.softmax(action_means, dim=1)
                action_probs = (action_var * action_softmax).sum(dim=1)

                action_means_old = action_means_old + discrete_action_eps
                action_old_softmax = F.softmax(action_means_old, dim=1)
                action_old_probs = (action_var * action_old_softmax).sum(dim=1)

                log_prob_cur = action_probs.log()
                log_prob_old = action_old_probs.log()
            else:
                log_prob_cur = normal_log_density(action_var,
                                                  action_means,
                                                  action_log_stds,
                                                  action_stds)

                log_prob_old = normal_log_density(action_var,
                                                  action_means_old,
                                                  action_log_stds_old,
                                                  action_stds_old)

            # ==== Update value net ====
            if args.use_value_net:
                self.opt_value.zero_grad()
                value_inp_var = None
                if self.args.use_goal_in_value:
                    value_inp_var = torch.cat((state_var, goal_var), 1)
                else:
                    value_inp_var = torch.cat((state_var, latent_next_c_var), 1)
                value_var = self.value_net(value_inp_var)
                value_loss = (value_var - \
                        targets[curr_id:curr_id+curr_batch_size]).pow(2.).mean()
                value_loss.backward()
                # clip_grad_value(self.value_net.parameters(), 50)
                self.opt_value.step()
                self.logger.summary_writer.add_scalar(
                        'loss/value',
                        value_loss.data.cpu().numpy()[0],
                        self.gail_step_count)
            # ==== END ====

            # ==== Update policy net (PPO step) ====
            self.opt_policy.zero_grad()
            ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
            surr1 = ratio * advantages_var
            surr2 = torch.clamp(
                    ratio,
                    1.0 - self.args.clip_epsilon,
                    1.0 + self.args.clip_epsilon) * advantages_var
            policy_surr = -torch.min(surr1, surr2).mean()
            policy_surr.backward()
            # This clips the entire norm.
            # torch.nn.utils.clip_grad_norm(self.policy_net.parameters(), 10)
            clip_grad_value(self.policy_net.parameters(), 50)
            self.opt_policy.step()
            # ==== END ====

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

    def update_params(self, gen_batch, expert_batch, expert_batch_indices,
                      episode_idx, optim_epochs, optim_batch_size):
        '''Update params for Policy (G), Reward (D) and Posterior (q) networks.
        '''
        args, dtype = self.args, self.dtype

        # generated trajectories
        states = torch.Tensor(np.array(gen_batch.state)).type(dtype)
        actions = torch.Tensor(np.array(gen_batch.action)).type(dtype)
        rewards = torch.Tensor(np.array(gen_batch.reward)).type(dtype)
        masks = torch.Tensor(np.array(gen_batch.mask)).type(dtype)
        goal = torch.Tensor(np.array(gen_batch.goal)).type(dtype)

        ## Expand states to include history ##
        # Generated trajectories already have history in them.

        latent_c = torch.Tensor(np.array(gen_batch.c)).type(dtype)
        latent_next_c = torch.Tensor(np.array(gen_batch.next_c)).type(dtype)
        values = None
        if args.use_value_net:
            value_net_inp = None
            if self.args.use_goal_in_value:
                value_net_inp = torch.cat((states, goal), 1)
            else:
                value_net_inp = torch.cat((states, latent_next_c), 1)
            values = self.value_net(Variable(value_net_inp))

        # expert trajectories
        list_of_expert_states, list_of_expert_actions = [], []
        list_of_expert_latent_c, list_of_masks = [], []
        list_of_expert_goals = []
        for i in range(len(expert_batch.state)):
            # c sampled from expert trajectories is incorrect since we don't
            # have "true c". Hence, we use the trained VAE to get the "true c".
            assert self.cached_expert_c_list is not None
            expert_batch_index = expert_batch_indices[i]
            expert_c = self.cached_expert_c_list[expert_batch_index]

            expert_goal = None
            if self.vae_train.use_rnn_goal_predictor:
                raise ValueError("Not implemented.")
            else:
                batch_goals = expert_batch.c[i].astype(np.int32)
                expert_goal = np.zeros((batch_goals.shape[0], self.num_goals))
                expert_goal[np.arange(batch_goals.shape[0]), batch_goals] = 1.0

            ## Expand expert states ##
            expert_state_i = expert_batch.state[i]
            if self.args.env_type == 'gym' and self.args.use_time_in_state:
                time_arr = np.arange(expert_batch.state[i].shape[0]) \
                        / (expert_batch.state[i].shape[0])
                expert_state_i = np.concatenate(
                        [expert_state_i, time_arr[:, None]], axis=1)
            expanded_states = self.expand_states_numpy(expert_state_i,
                                                       self.history_size)
            list_of_expert_states.append(torch.Tensor(expanded_states))
            list_of_expert_actions.append(torch.Tensor(expert_batch.action[i]))
            list_of_expert_latent_c.append(torch.Tensor(expert_c))
            list_of_expert_goals.append(torch.Tensor(expert_goal))
            list_of_masks.append(torch.Tensor(expert_batch.mask[i]))

        expert_states = torch.cat(list_of_expert_states,0).type(dtype)
        expert_actions = torch.cat(list_of_expert_actions, 0).type(dtype)
        expert_latent_c = torch.cat(list_of_expert_latent_c, 0).type(dtype)
        expert_goals = torch.cat(list_of_expert_goals, 0).type(dtype)
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
                                                        self.args.tau,
                                                        values=values,
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
        if len(actions.shape) == 1:
            actions = actions.view(-1, 1)
        elif len(actions.shape) == 3:
            actions = actions.view(actions.size(0), -1)

        for _ in range(optim_epochs):
            perm = np.random.permutation(np.arange(actions.size(0)))
            perm_exp = np.random.permutation(np.arange(expert_actions.size(0)))
            if args.cuda:
                perm = torch.cuda.LongTensor(perm)
                perm_exp = torch.cuda.LongTensor(perm_exp)
            else:
                perm = torch.LongTensor(perm)
                perm_exp = torch.LongTensor(perm_exp)

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
                optim_iters,
                goal=goal[perm],
                expert_goal=expert_goals[perm_exp])

    def collect_samples(self, batch_size, max_episode_len, train=True):
        run_agent_worker_args = {}
        dtype = self.dtype

        args_list, per_worker_batch_size = [], batch_size // len(self.envs)
        for i, env in enumerate(self.envs):
            # Generate trajectories by sampling both from expert and by current
            # policy.
            sample_c_from_expert = (i % 2 == 0)
            # sample_c_from_expert = train
            run_agent_worker_args[i] = (args,
                                        self.vae_train.vae_model,
                                        self.policy_net,
                                        self.reward_net,
                                        self.posterior_net,
                                        self.expert,
                                        env,
                                        self.cached_expert_c_list,
                                        per_worker_batch_size,
                                        self.vae_train.vae_model.posterior_latent_size,
                                        self.num_goals,
                                        self.vae_train.use_rnn_goal_predictor,
                                        self.vae_train.args.use_discrete_vae,
                                        self.dtype,
                                        train,
                                        sample_c_from_expert,
                                        )
            args_list.append(run_agent_worker_args[i])

        start_time = time.time()
        # Get the results by making a BLOCKING call.
        results = self.env_pool.map(parallel_run_agent_worker, args_list)
        end_time = time.time()
        print("END: Blocking call, time: {:.3f}".format(end_time-start_time))

        memory_list, log_list = [], []
        for i in range(len(self.envs)):
            m_list = results[i]['memory_list']
            l_list = results[i]['log_list']
            memory_list += m_list
            log_list += l_list
        total_memory = Memory()
        total_memory.merge_list(memory_list)
        return total_memory.sample(), memory_list, log_list

    def train_gail(self, num_epochs, results_pkl_path,
                   gen_batch_size=1, train=True):
        # Train GAIL.
        args, dtype = self.args, self.dtype
        results = {'average_reward': [], 'episode_reward': [],
                   'true_traj_state': {}, 'true_traj_action': {},
                   'pred_traj_state': {}, 'pred_traj_action': {}}

        self.train_step_count, self.gail_step_count = 0, 0
        gen_traj_step_count = 0
        self.convert_models_to_type(dtype)

        #running_state = ZFilter((self.vae_train.args.vae_state_size), clip=5)
        if train:
            print("Will get c for all expert trajectories.")
            t1 = time.time()
            self.cached_expert_c_list = self.get_c_for_all_expert_trajs(self.expert)
            print("Time: {:.3f} Did get c for all expert trajectories. ".format(
                time.time() - t1))
        else:
            self.cached_expert_c_list = None

        for ep_idx in range(num_epochs):
            num_steps, batch_size = 0, 1
            reward_batch, expert_true_reward_batch = [], []
            true_traj_curr_epoch = {'state':[], 'action': []}
            gen_traj_curr_epoch = {'state': [], 'action': []}
            #env_reward_batch_dict = {'linear_traj_reward': [],
            #                         'map_traj_reward': []}
            env_reward_batch_dict = {'true_reward': []}

            collect_sample_start_time = time.time()
            # Update learning rate schedules (decay etc.)
            self.opt_policy.lr = self.get_policy_learning_rate(ep_idx)

            gen_batch, _, log_list = self.collect_samples(
                    gen_batch_size, self.args.max_ep_length + 1, train=train)

            # Add to tensorboard if training.
            true_reward =[log['true_reward'] for log in log_list]
            gen_reward = [log['total_disc_reward'] for log in log_list]
            if train:
                add_scalars_to_summary_writer(
                        self.logger.summary_writer,
                        'gen_traj/reward', {
                            'average': np.mean(gen_reward),
                            'max': np.max(gen_reward),
                            'min': np.min(gen_reward)
                        },
                        self.train_step_count)
                add_scalars_to_summary_writer(
                        self.logger.summary_writer,
                        'gen_traj/true_reward', {
                            'average': np.mean(true_reward),
                            'max': np.max(true_reward),
                            'min': np.min(true_reward),
                        },
                        self.train_step_count)

            collect_sample_end_time = time.time()

            gail_train_start_time, gail_train_end_time = 0, 0
            if train:
                gail_train_start_time = time.time()
                # We do not get the context variable from expert trajectories.
                # Hence we need to fill it in later.
                expert_batch, expert_batch_indices = self.expert.sample(
                        size=args.num_expert_trajs, return_indices=True)

                self.update_params(gen_batch, expert_batch, expert_batch_indices,
                                   ep_idx, args.optim_epochs,
                                   args.optim_batch_size)
                gail_train_end_time = time.time()

                self.train_step_count += 1

            if not train or (ep_idx > 0 and  ep_idx % args.log_interval == 0):
                update_time =\
                        (collect_sample_end_time - collect_sample_start_time) \
                        + (gail_train_end_time - gail_train_start_time)
                print('Episode [{}/{}]   Time: {:.3f} \t Gen Reward: Avg R: {:.2f} Max R: {:.2f} \t' \
                        '\tTrue Reawrd => Avg {:.2f} \t std: {:.2f} \t Max: {:.2f}'.format(
                      ep_idx, args.num_epochs,
                      update_time,
                      np.mean(gen_reward), np.max(gen_reward), np.mean(true_reward),
                      np.std(true_reward), np.max(true_reward)))
                print("Gen batch time: {:.3f}, GAIL update time: {:.3f}: Mean param: {}".format(
                    collect_sample_end_time - collect_sample_start_time,
                    gail_train_end_time - gail_train_start_time,
                    np.array_str(
                        self.policy_net.action_log_std.data.cpu().numpy()[0],
                        precision=4, suppress_small=True)))

            with open(results_pkl_path, 'wb') as results_f:
                pickle.dump((results), results_f, protocol=2)

            if train and ep_idx > 0 and ep_idx % args.save_interval == 0:
                if self.dtype != torch.FloatTensor:
                    self.convert_models_to_type(torch.FloatTensor)
                checkpoint_filepath = self.model_checkpoint_filepath(ep_idx)
                torch.save(self.checkpoint_data_to_save(), checkpoint_filepath)
                print("Did save checkpoint: {}".format(checkpoint_filepath))
                if self.dtype != torch.FloatTensor:
                    self.convert_models_to_type(self.dtype)


def check_args(saved_args, new_args):
    assert saved_args.use_state_features == new_args.use_state_features, \
            'Args do not match - use_state_features'

def load_VAE_model(model_checkpoint_path, new_args):
    # Load pre-trained VAE model.

    checkpoint_dir_path = os.path.dirname(model_checkpoint_path)
    results_dir_path = os.path.dirname(checkpoint_dir_path)

    # Load arguments used to train the model
    saved_args_filepath = os.path.join(results_dir_path, 'args.pkl')
    with open(saved_args_filepath, 'rb') as saved_args_f:
        saved_args = pickle.load(saved_args_f)
        print('Did load saved args {}'.format(saved_args_filepath))

    # check args
    check_args(saved_args, new_args)

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

def create_result_dirs(results_dir):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        # Directory for TF logs
        os.makedirs(os.path.join(results_dir, 'log'))
        # Directory for model checkpoints
        os.makedirs(os.path.join(results_dir, 'checkpoint'))

def main(args):
    # Create Logger
    if not os.path.exists(os.path.join(args.results_dir, 'log')):
        os.makedirs(os.path.join(args.results_dir, 'log'))
    logger = TensorboardXLogger(os.path.join(args.results_dir, 'log'))

    # Load finetune args.
    finetune_args = None
    if len(args.finetune_path) > 0:
        finetune_args_path = os.path.dirname(os.path.dirname(args.finetune_path))
        finetune_args_path = os.path.join(finetune_args_path, 'args.pkl')
        assert os.path.exists(finetune_args_path), "Finetune args does not exist."
        with open(finetune_args_path, 'rb') as finetune_args_f:
            finetune_args = pickle.load(finetune_args_f)

    print('Loading expert trajectories ...')
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
    causal_gail_mlp.set_expert(expert)

    if len(args.checkpoint_path) > 0:
        print("Test checkpoint: {}".format(args.checkpoint_path))
        causal_gail_mlp.load_checkpoint_data(args.checkpoint_path)
        results_pkl_path = os.path.join(
                args.results_dir,
                'results_' + os.path.basename(args.checkpoint_path)[:-3] \
                        + 'pkl')

        causal_gail_mlp.train_gail(
                1,
                results_pkl_path,
                gen_batch_size=1000*10,
                train=False)
        print("Did save results to: {}".format(results_pkl_path))
        return

    if len(args.finetune_path) > 0:
        # -4 removes .pth from finetune path
        checkpoint_name = os.path.basename(args.finetune_path)[:-4]
        # Create results directory for finetune results.
        results_dir = os.path.join(args.results_dir,
                                   'finetune_' + checkpoint_name)
        create_result_dirs(results_dir)
        # Set new Tensorboard logger for finetune results.
        logger = TensorboardXLogger(os.path.join(results_dir, 'log'))
        causal_gail_mlp.logger = logger

        print("Finetune checkpoint: {}".format(args.finetune_path))
        causal_gail_mlp.load_checkpoint_data(args.finetune_path)
        causal_gail_mlp.train_gail(
                args.num_epochs,
                os.path.join(results_dir, 'results.pkl'),
                gen_batch_size=args.batch_size,
                train=True)
        return

    if args.init_from_vae:
        print("Will load generator and posterior from pretrianed VAE.")
        causal_gail_mlp.load_weights_from_vae()

    results_path = os.path.join(args.results_dir, 'results.pkl')
    causal_gail_mlp.train_gail(
            args.num_epochs,
            os.path.join(args.results_dir, 'results.pkl'),
            gen_batch_size=args.batch_size,
            )

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
    parser.add_argument('--posterior_learning_rate', type=float, default=3e-4,
                        help='VAE posterior lr (default: 3e-4)')
    parser.add_argument('--gen_learning_rate', type=float, default=3e-4,
                        help='Generator lr (default: 3e-4)')
    parser.add_argument('--discriminator_pretrain_epochs', type=int, default=0,
                        help='Number of epochs to pre-train discriminator.')
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
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Checkpoint path to load pre-trained models.')
    parser.add_argument('--finetune_path', type=str, default='',
                        help='pre-trained models to finetune.')

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
    parser.add_argument('--disc_reward', choices=['no_log', 'log_d', 'log_1-d'],
                        default='log_d',
                        help='Discriminator reward to use.')

    parser.add_argument('--use_value_net', dest='use_value_net',
                        action='store_true',
                        help='Use value network.')
    parser.add_argument('--no-use_value_net', dest='use_value_net',
                        action='store_false',
                        help='Don\'t use value network.')
    parser.set_defaults(use_value_net=True)

    parser.add_argument('--use_goal_in_policy', dest='use_goal_in_policy',
                        action='store_true',
                        help='Use goal instead of context in Policy.')
    parser.add_argument('--no-use_goal_in_policy', dest='use_goal_in_policy',
                        action='store_false',
                        help='Use context instead of goal in Policy.')
    parser.set_defaults(use_goal_in_policy=False)

    parser.add_argument('--use_goal_in_value', dest='use_goal_in_value',
                        action='store_true',
                        help='Use goal instead of context in Value net.')
    parser.add_argument('--no-use_goal_in_value', dest='use_goal_in_value',
                        action='store_false',
                        help='Use context instead of goal in Value.')
    parser.set_defaults(use_goal_in_value=False)
    parser.add_argument('--use_random_starts', dest='use_random_starts',
                        action='store_true',
                        help='Use random start states.')
    parser.set_defaults(use_random_starts=False)

    parser.add_argument('--init_from_vae', dest='init_from_vae',
                        action='store_true',
                        help='Init policy and posterior from vae.')
    parser.add_argument('--no-init_from_vae', dest='init_from_vae',
                        action='store_false',
                        help='Don\'t init policy and posterior from vae.')
    parser.set_defaults(init_from_vae=True)

    # Environment - Grid or Mujoco
    parser.add_argument('--env-type', default='grid',
                        choices=['grid', 'grid_room', 'mujoco', 'gym'],
                        help='Environment type Grid or Mujoco.')
    parser.add_argument('--env-name', default=None,
                        help='Environment name if Mujoco.')
    parser.add_argument('--num_threads', type=int, default=1,
                        help='Number of threads to collect train samples.')

    parser.add_argument('--use_time_in_state', dest='use_time_in_state',
                        action='store_true',
                        help='Use time to goal completion in state.')
    parser.add_argument('--no-use_time_in_state', dest='use_time_in_state',
                        action='store_false',
                        help='Dont use time to goal completion in state.')
    parser.set_defaults(use_time_in_state=False)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    create_result_dirs(args.results_dir)

    # Save runtime arguments to pickle file
    args_pkl_filepath = os.path.join(args.results_dir, 'args.pkl')
    with open(args_pkl_filepath, 'wb') as args_pkl_f:
        pickle.dump(args, args_pkl_f, protocol=2)

    if args.use_goal_in_policy:
        raise ValueError("This script does not support having goal in policy." \
                "It only uses context.")

    main(args)
