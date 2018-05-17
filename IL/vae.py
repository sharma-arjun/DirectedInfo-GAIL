import numpy as np
import argparse
import h5py
import os
import pdb
import pickle
import torch
import gym
import math

from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

from load_expert_traj import Expert, ExpertHDF5
from load_expert_traj import recursively_save_dict_contents_to_group
from grid_world import State, Action, TransitionFunction
from grid_world import StateVector, ActionVector
from grid_world import RewardFunction, RewardFunction_SR2
from grid_world import create_obstacles, obstacle_movement, sample_start
from itertools import product
from models import Policy, Posterior, DiscretePosterior

from utils.logger import Logger, TensorboardXLogger
from utils.torch_utils import get_weight_norm_for_network

#-----Environment-----#

#if args.expert_path == 'SR2_expert_trajectories/':
#    R = RewardFunction_SR2(-1.0,1.0,width)
#else:
#    R = RewardFunction(-1.0,1.0)


class VAE(nn.Module):
    def __init__(self,
                 policy_state_size=1, posterior_state_size=1,
                 policy_action_size=1, posterior_action_size=1,
                 policy_latent_size=1, posterior_latent_size=1,
                 posterior_goal_size=1,
                 policy_output_size=1,
                 history_size=1,
                 hidden_size=64,
                 use_goal_in_policy=True,
                 use_separate_goal_policy=True,
                 use_history_in_policy=False):
        '''
        state_size: State size
        latent_size: Size of 'c' variable
        goal_size: Number of goals.
        output_size:
        '''
        super(VAE, self).__init__()

        self.history_size = history_size
        self.policy_state_size = policy_state_size
        self.posterior_latent_size = posterior_latent_size
        self.posterior_goal_size = posterior_goal_size
        self.use_goal_in_policy = use_goal_in_policy
        self.use_separate_goal_policy = use_separate_goal_policy
        self.use_history_in_policy = use_history_in_policy

        self.policy_latent_size = policy_latent_size
        if use_goal_in_policy:
            self.policy_latent_size += posterior_goal_size

        #if args.discrete:
        #    output_activation='sigmoid'
        #else:
        output_activation=None

        if use_history_in_policy:
            policy1_state_size = policy_state_size * history_size
        else:
            policy1_state_size = policy_state_size

        self.policy = Policy(state_size=policy1_state_size,
                             action_size=policy_action_size,
                             latent_size=self.policy_latent_size,
                             output_size=policy_output_size,
                             hidden_size=hidden_size,
                             output_activation=output_activation)

        if use_separate_goal_policy:
            self.policy_goal = Policy(
                    state_size=policy_state_size*self.history_size,
                    action_size=policy_action_size,
                    latent_size=posterior_goal_size,
                    output_size=policy_output_size,
                    hidden_size=hidden_size,
                    output_activation=output_activation)

        self.posterior = Posterior(
                state_size=posterior_state_size*self.history_size,
                action_size=posterior_action_size,
                latent_size=posterior_latent_size+posterior_goal_size,
                output_size=posterior_latent_size,
                hidden_size=hidden_size)


    def encode(self, x, c):
        return self.posterior(torch.cat((x, c), 1))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


    def decode_goal_policy(self, x, g):
        action_mean, _, _ = self.policy_goal(torch.cat((x, g), 1))
        return action_mean

    def decode(self, x, c):
        action_mean, action_log_std, action_std = self.policy(
                torch.cat((x, c), 1))

        return action_mean

    def forward(self, x, c, g, only_goal_policy=False):
        if only_goal_policy:
            decoder_output_2 = self.decode_goal_policy(x, g)
            # Return a tuple as the else part below. Caller should expect a
            # tuple always.
            return decoder_output_2,
        else:
            mu, logvar = self.encode(x, c)
            c[:,-self.posterior_latent_size:] = self.reparameterize(mu, logvar)

            decoder_output_1 = None
            decoder_output_2 = None


        if self.use_goal_in_policy:
            if self.use_history_in_policy:
                decoder_output_1 = self.decode(x, c)
            else:
                decoder_output_1 = self.decode(
                        x[:, -self.policy_state_size:], c)
        else:
            if self.use_history_in_policy:
                decoder_output_1 = self.decode(
                        x, c[:,-self.posterior_latent_size:])
            else:
                decoder_output_1 = self.decode(
                        x[:, -self.policy_state_size:],
                        c[:,-self.posterior_latent_size:])

            if self.use_separate_goal_policy:
                decoder_output_2 = self.decode_goal_policy(x, g)


            return decoder_output_1, decoder_output_2, mu, logvar

class DiscreteVAE(VAE):
    def __init__(self, temperature=5.0, **kwargs):
        '''
        state_size: State size
        latent_size: Size of 'c' variable
        goal_size: Number of goals.
        output_size:
        '''
        super(DiscreteVAE, self).__init__(**kwargs)
        self.posterior = DiscretePosterior(
                state_size=kwargs['posterior_state_size']*self.history_size,
                action_size=kwargs['posterior_action_size'],
                latent_size=kwargs['posterior_latent_size']+kwargs['posterior_goal_size'],
                output_size=kwargs['posterior_latent_size'],
                hidden_size=kwargs['hidden_size'],
        )
        self.encoder_softmax = nn.Softmax(dim=1)
        self.temperature = temperature
        self.init_temperature = temperature

    def update_temperature(self, epoch):
        '''Update temperature.'''
        r = 2e-3
        self.temperature = max(0.5, self.init_temperature * math.exp(-r*epoch))


    def encode(self, x, c):
        '''Return the log probability output for the encoder.'''
        logits = self.posterior(torch.cat((x, c), 1))
        return logits

    def sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dtype = logits.data.type()
        y = logits + Variable(self.sample_gumbel(logits.size())).type(dtype)
        y = F.softmax(y / temperature, dim=1)
        # shape = y.size()
        # _, ind = y.max(dim=-1)
        # y_hard = torch.zeros_like(y).view(-1, shape[-1])
        # y_hard.scatter_(1, ind.view(-1, 1), 1)
        # y_hard = y_hard.view(*shape)
        # return (y_hard - y).detach() + y
        return y

    def reparameterize(self, logits, temperature, eps=1e-10):
        if self.training:
            probs = self.gumbel_softmax_sample(logits, temperature)
        else:
            probs = F.softmax(logits / temperature, dim=1)
        return probs

    def forward(self, x, c, g, only_goal_policy=False):
        if only_goal_policy:
            decoder_output_2 = self.decode_goal_policy(x, g)
            # Return a tuple as the else part below. Caller should expect a
            # tuple always.
            return decoder_output_2,

        c_logits = self.encode(x, c)
        c[:, -self.posterior_latent_size:] = self.reparameterize(
                c_logits, self.temperature)

        decoder_output_1 = None
        decoder_output_2 = None

        if self.use_goal_in_policy:
            if self.use_history_in_policy:
                decoder_output_1 = self.decode(x, c)
            else:
                decoder_output_1 = self.decode(x[:,-self.policy_state_size:], c)
        else:
            if self.use_history_in_policy:
                decoder_output_1 = self.decode(
                        x, c[:,-self.posterior_latent_size:])
            else:
                decoder_output_1 = self.decode(
                        x[:, -self.policy_state_size:],
                        c[:,-self.posterior_latent_size:])

        if self.use_separate_goal_policy:
            decoder_output_2 = self.decode_goal_policy(x, g)

        return decoder_output_1, decoder_output_2, c_logits


class VAETrain(object):
    def __init__(self, args,
                 logger,
                 width=21,
                 height=21,
                 state_size=2,
                 action_size=4,
                 num_goals=4,
                 history_size=1,
                 use_rnn_goal_predictor=False,
                 dtype=torch.FloatTensor,
                 env_type='grid',
                 env_name=None):

        self.args = args
        self.logger = logger
        self.width, self.height = width, height
        self.state_size = state_size
        self.action_size = action_size
        self.history_size = history_size
        self.num_goals = num_goals
        self.dtype = dtype
        self.use_rnn_goal_predictor = use_rnn_goal_predictor
        self.env_type = env_type


        self.train_step_count = 0

        # Create models
        self.Q_model = nn.LSTMCell(self.state_size + action_size, 64)
        self.Q_2_model = nn.LSTMCell(64, 64)

        # Output of linear model num_goals = 4
        self.Q_model_linear = nn.Linear(64, num_goals)

        # action_size is 0
        # Hack -- VAE input dim (s + a + latent).
        if args.use_discrete_vae:
            self.vae_model = DiscreteVAE(
                    temperature=args.temperature,
                    policy_state_size=state_size,
                    posterior_state_size=state_size,
                    policy_action_size=0,
                    posterior_action_size=0,
                    policy_latent_size=args.vae_context_size,
                    posterior_latent_size=args.vae_context_size,
                    posterior_goal_size=num_goals,
                    policy_output_size=action_size,
                    history_size=history_size,
                    hidden_size=64,
                    use_goal_in_policy=args.use_goal_in_policy,
                    use_separate_goal_policy=args.use_separate_goal_policy)
        else:
            self.vae_model = VAE(
                    policy_state_size=state_size,
                    posterior_state_size=state_size,
                    policy_action_size=0,
                    posterior_action_size=0,
                    policy_latent_size=args.vae_context_size,
                    posterior_latent_size=args.vae_context_size,
                    posterior_goal_size=num_goals,
                    policy_output_size=action_size,
                    history_size=history_size,
                    hidden_size=64,
                    use_goal_in_policy=args.use_goal_in_policy,
                    use_separate_goal_policy=args.use_separate_goal_policy,
                    use_history_in_policy=args.use_history_in_policy)

        self.obstacles, self.transition_func = None, None

        if args.run_mode == 'train':
            if use_rnn_goal_predictor:
                self.vae_opt = optim.Adam(self.vae_model.parameters(), lr=1e-3)
                self.Q_model_opt = optim.Adam([
                        {'params': self.Q_model.parameters()},
                        {'params': self.Q_2_model.parameters()},
                        {'params': self.Q_model_linear.parameters()},
                    ],
                    lr=1e-3)
            else:
                self.vae_opt = optim.Adam(self.vae_model.parameters(), lr=1e-3)
        elif args.run_mode == 'train_goal_pred':
            self.vae_opt = optim.Adam(self.vae_model.policy_goal.parameters(),
                                      lr=1e-3)
            self.Q_model_opt = optim.Adam([
                {'params': self.Q_model.parameters()},
                {'params': self.Q_2_model.parameters()},
                {'params': self.Q_model_linear.parameters()}],
                lr=1e-3)
        elif args.run_mode == 'test' or args.run_mode == 'test_goal_pred':
            pass
        else:
            raise ValueError('Incorrect value for run_mode {}'.format(
                args.run_mode))
        self.create_environment(env_type, env_name)
        self.expert = None
        self.obstacles, self.set_diff = None, None

    def model_checkpoint_dir(self):
        '''Return the directory to save models in.'''
        return os.path.join(self.args.results_dir, 'checkpoint')

    def model_checkpoint_filename(self, epoch):
        return os.path.join(self.model_checkpoint_dir(),
                            'cp_{}.pth'.format(epoch))

    def create_environment(self, env_type, env_name=None):
        if 'grid' in env_type:
            self.transition_func = TransitionFunction(self.width,
                                                      self.height,
                                                      obstacle_movement)
        elif env_type == 'mujoco':
            assert(env_name is not None)
            self.env = gym.make(env_name)

    def set_expert(self, expert):
        assert self.expert is None, "Trying to set non-None expert"
        self.expert = expert
        assert expert.obstacles is not None, "Obstacles cannot be None"
        self.obstacles = expert.obstacles
        assert expert.set_diff is not None, "set_diff in h5 file cannot be None"
        self.set_diff = expert.set_diff

    def convert_models_to_type(self, dtype):
        self.vae_model = self.vae_model.type(dtype)
        if self.use_rnn_goal_predictor:
          self.Q_model = self.Q_model.type(dtype)
          self.Q_2_model = self.Q_2_model.type(dtype)
          self.Q_model_linear = self.Q_model_linear.type(dtype)


    def loss_function_using_goal(self, recon_using_goal, x, epoch):
        if self.args.discrete_action:
            _, label = torch.max(x, 1)
            loss = F.cross_entropy(recon_using_goal, label)
        else:
            loss = F.mse_loss(recon_using_goal, x)
        return loss

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x1, recon_x2, x, vae_posterior_output, epoch):
        lambda_loss1 = 1.0
        th_epochs = 0.5*args.num_epochs
        #lambda_kld = max(0.1, 0.1 + (lambda_loss1 - 0.1) \
        #        * ((epoch - th_epochs)/(args.num_epochs-th_epochs)))
        lambda_kld = 0.00001
        lambda_loss2 = 10.0

        if self.args.discrete_action:
            _, label = torch.max(x, 1)
            loss1 = F.cross_entropy(recon_x1, label)
            if self.args.use_separate_goal_policy:
                loss2 = F.cross_entropy(recon_x2, label)
        else:
            loss1 = F.mse_loss(recon_x1, x)
            if self.args.use_separate_goal_policy:
                loss2 = F.mse_loss(recon_x2, x)

        if self.args.use_discrete_vae:
            # logits is the un-normalized log probability for belonging to a class
            logits = vae_posterior_output[0]
            num_q_classes = self.vae_model.posterior_latent_size
            # q_prob is (N, C)
            q_prob = F.softmax(logits, dim=1)  # q_prob
            log_q_prob = torch.log(q_prob + 1e-10)  # log q_prob
            prior_prob = Variable(torch.Tensor([1.0 / num_q_classes])).type(
                    logits.data.type())
            batch_size = logits.size(0)
            KLD = torch.sum(q_prob * (log_q_prob - torch.log(prior_prob))) / batch_size
            # print("q_prob: {}".format(q_prob))
        else:
            mu, logvar = vae_posterior_output[0], vae_posterior_output[1]
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            batch_size = mu.size(0)
            KLD =-0.5 * (torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size)


        #return MSE + KLD
        if self.args.use_separate_goal_policy:
            return lambda_loss1*loss1 + lambda_loss2*loss2 + lambda_kld*KLD, \
                    loss1, loss2, KLD
        else:
            return lambda_loss1*loss1 + lambda_kld*KLD, loss1, None, KLD

    def log_model_to_tensorboard(
            self,
            models_to_log=(
                'vae_policy',
                'goal_policy',
                'vae_posterior',
                'Q_model_linear',
                'Q_model',
                )):
        '''Log weights and gradients of network to Tensorboard.

        models_to_log: Tuple of model names to log to tensorboard.
        '''
        if 'vae_policy' in models_to_log:
            vae_model_l2_norm, vae_model_grad_l2_norm = \
                get_weight_norm_for_network(self.vae_model.policy)
            self.logger.summary_writer.add_scalar(
                            'weight/policy',
                             vae_model_l2_norm,
                             self.train_step_count)
            self.logger.summary_writer.add_scalar(
                            'grad/policy',
                             vae_model_grad_l2_norm,
                             self.train_step_count)

        if self.vae_model.policy_goal is not None and \
                'goal_policy' in models_to_log:
            vae_model_l2_norm, vae_model_grad_l2_norm = \
                            get_weight_norm_for_network(self.vae_model.policy_goal)
            self.logger.summary_writer.add_scalar(
                            'weight/policy_goal',
                             vae_model_l2_norm,
                             self.train_step_count)
            self.logger.summary_writer.add_scalar(
                            'grad/policy_goal',
                             vae_model_grad_l2_norm,
                             self.train_step_count)

        if 'vae_posterior' in models_to_log:
            vae_model_l2_norm, vae_model_grad_l2_norm = \
                            get_weight_norm_for_network(self.vae_model.posterior)
            self.logger.summary_writer.add_scalar(
                            'weight/posterior',
                             vae_model_l2_norm,
                             self.train_step_count)
            self.logger.summary_writer.add_scalar(
                            'grad/posterior',
                             vae_model_grad_l2_norm,
                             self.train_step_count)


        if self.use_rnn_goal_predictor and 'Q_model_linear' in models_to_log:
            Q_model_l2_norm, Q_model_l2_grad_norm = \
                            get_weight_norm_for_network(self.Q_model_linear)
            self.logger.summary_writer.add_scalar(
                            'weight/Q_model_linear_l2',
                             Q_model_l2_norm,
                             self.train_step_count)
            self.logger.summary_writer.add_scalar(
                            'grad/Q_model_linear_l2',
                             Q_model_l2_grad_norm,
                             self.train_step_count)

        if self.use_rnn_goal_predictor and 'Q_model' in models_to_log:
            Q_model_l2_norm, Q_model_l2_grad_norm = \
                            get_weight_norm_for_network(self.Q_model)
            self.logger.summary_writer.add_scalar(
                            'weight/Q_model_l2',
                             Q_model_l2_norm,
                             self.train_step_count)
            self.logger.summary_writer.add_scalar(
                            'grad/Q_model_l2',
                             Q_model_l2_grad_norm,
                             self.train_step_count)

    def set_models_to_train(self):
        if self.use_rnn_goal_predictor:
          self.Q_model.train()
          self.Q_2_model.train()
          self.Q_model_linear.train()

        self.vae_model.train()

    def save_checkpoint(self, epoch):
        model_data = {
                'vae_model': self.vae_model.state_dict(),
                'Q_model': self.Q_model.state_dict(),
                'Q_2_model': self.Q_2_model.state_dict(),
                'Q_model_linear': self.Q_model_linear.state_dict(),
        }
        torch.save(model_data, self.model_checkpoint_filename(epoch))

    def load_checkpoint(self, checkpoint_path):
        '''Load models from checkpoint.'''
        checkpoint_models = torch.load(checkpoint_path)
        self.vae_model.load_state_dict(checkpoint_models['vae_model'])
        self.Q_model.load_state_dict(checkpoint_models['Q_model'])
        if checkpoint_models.get('Q_2_model') is not None:
            self.Q_2_model.load_state_dict(checkpoint_models['Q_2_model'])
        self.Q_model_linear.load_state_dict(checkpoint_models['Q_model_linear'])

    def load_checkpoint_goal_policy(self, checkpoint_path):
        '''Load models from checkpoint.'''
        checkpoint_models = torch.load(checkpoint_path)
        self.vae_model.load_state_dict(checkpoint_models['vae_model'])
        self.Q_model.load_state_dict(checkpoint_models['Q_model'])
        self.Q_2_model.load_state_dict(checkpoint_models['Q_2_model'])
        self.Q_model_linear.load_state_dict(checkpoint_models['Q_model_linear'])

    def get_state_features(self, state_obj, use_state_features):
        if use_state_features:
            feat = np.array(state_obj.get_features(), dtype=np.float32)
        else:
            feat = np.array(state_obj.coordinates, dtype=np.float32)
        return feat

    def get_history_features(self, state, use_velocity=False):
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
                new_state[one_idx, t, :] = \
                        state[one_idx, t+1, :] - state[one_idx, t, :]

            new_state[:, history_size-1, :] = state[:, history_size-1, :]

            return new_state
        else:
            return state

    def get_context_at_state(self, x, c):
        '''Get context variable c_t for given x_t, c_{t-1}.

        x: State at time t. (x_t)
        c: Context at time t-1. (c_{t-1})
        '''
        if self.args.use_discrete_vae:
            logits = self.vae_model.encode(x, c)
            return self.vae_model.reparameterize(logits,
                                                 self.vae_model.temperature)
        else:
            mu, logvar = self.vae_model.encode(x, c)
            return self.vae_model.reparameterize(mu, logvar)

    def train(self, expert, num_epochs, batch_size,
              train_goal_policy_only=False):
        final_train_stats = {
            'train_loss': [],
            'goal_pred_conf_arr': [],
            'temperature': [],
        }
        self.train_step_count = 0
        # Convert models to right type.
        self.convert_models_to_type(self.dtype)

        # Create the checkpoint directory.
        if not os.path.exists(self.model_checkpoint_dir()):
            os.makedirs(self.model_checkpoint_dir())
        # Save runtime arguments to pickle file
        args_pkl_filepath = os.path.join(self.args.results_dir, 'args.pkl')
        with open(args_pkl_filepath, 'wb') as args_pkl_f:
            pickle.dump(self.args, args_pkl_f, protocol=2)

        for epoch in range(1, num_epochs+1):
            self.vae_model.update_temperature(epoch-1)
            train_stats = self.train_fixed_length_epoch(
                    epoch,
                    expert,
                    batch_size,
                    train_goal_policy_only=train_goal_policy_only,
                    episode_len=self.args.episode_len)

            # Update stats for epoch
            final_train_stats['train_loss'].append(train_stats['train_loss'])
            if self.args.use_discrete_vae:
                final_train_stats['temperature'].append(
                        self.vae_model.temperature)

            if epoch % 1 == 0:
                results_pkl_path = os.path.join(self.args.results_dir,
                                                'results.pkl')
                self.test_models(expert, results_pkl_path=None,
                                 num_test_samples=5,   
                                 test_goal_policy_only=train_goal_policy_only)

            if epoch % self.args.checkpoint_every_epoch == 0:
                if self.dtype != torch.FloatTensor:
                    self.convert_models_to_type(torch.FloatTensor)
                self.save_checkpoint(epoch)
                if self.dtype != torch.FloatTensor:
                    self.convert_models_to_type(self.dtype)

        results_pkl_path = os.path.join(self.args.results_dir, 'results.pkl')
        self.test_models(expert, results_pkl_path=results_pkl_path,
                         num_test_samples=2,
                         other_results_dict={'train_stats': final_train_stats},
                         test_goal_policy_only=train_goal_policy_only)



    # TODO: Add option to not save gradients for backward pass when not needed
    def predict_goal(self, ep_state, ep_action, ep_c, ep_mask, num_goals):
        '''Predicts goal for 1 expert sample.

        Forward pass through the Q network.

        Return:
            final_goal: Average of all goals predicted at every step.
            pred_goal: Goal predicted at every step of the RNN.
        '''
        # Predict goal for the entire episode i.e., forward prop through Q
        batch_size, episode_len = ep_state.shape[0], ep_state.shape[1]

        ht = Variable(torch.zeros(batch_size, 64).type(self.dtype),
                      requires_grad=True)
        ct = Variable(torch.zeros(batch_size, 64).type(self.dtype),
                      requires_grad=True)
        ht_2 = Variable(torch.zeros(batch_size, 64).type(self.dtype),
                        requires_grad=True)
        ct_2 = Variable(torch.zeros(batch_size, 64).type(self.dtype),
                        requires_grad=True)
        final_goal = Variable(torch.zeros(batch_size, num_goals).type(
            self.dtype))
        pred_goal = []
        for t in range(episode_len):
            if 'grid' in self.env_type:
                state_obj = StateVector(ep_state[:, t, :], self.obstacles)
                # state_tensor is (N, F)
                state_tensor = torch.from_numpy(self.get_state_features(
                    state_obj, self.args.use_state_features)).type(self.dtype)
            elif self.env_type == 'mujoco':
                state_tensor = torch.from_numpy(ep_state[:, t, :]).type(self.dtype)

            # action_tensor is (N, A)
            action_tensor = torch.from_numpy(ep_action[:, t, :]).type(self.dtype)

            # NOTE: Input to LSTM cell needs to be of shape (N, F) where N can
            # be 1. inp_tensor is (N, F+A)
            inp_tensor = torch.cat((state_tensor, action_tensor), 1)

            ht, ct = self.Q_model(Variable(inp_tensor), (ht, ct))
            if self.args.stacked_lstm == 1:
                ht_2, ct_2 = self.Q_2_model(ht, (ht_2, ct_2))
                # output = self.Q_model_linear(ht_2)
                output = ht_2
            else:
                # output = self.Q_model_linear(ht)
                output = ht

            if self.args.flag_goal_pred == 'sum_all_hidden':
                final_goal = final_goal + self.Q_model_linear(output)
            elif self.args.flag_goal_pred == 'last_hidden':
                final_goal = output
            else:
                raise ValueError("Invalid goal pred flag {}".format(
                    self.args.flag_goal_pred))

            pred_goal.append(output)

        # final goal should be (N, G)
        if self.args.flag_goal_pred == 'sum_all_hidden':
            final_goal = F.softmax(final_goal / episode_len, dim=1)
        elif self.args.flag_goal_pred == 'last_hidden':
            final_goal = F.softmax(self.Q_model_linear(final_goal), dim=1)
        else:
            raise ValueError("Invalid goal pred flag {}".format(
                    self.args.flag_goal_pred))

        return final_goal, pred_goal

    def test_generate_trajectory_variable_length(self,
                                                 expert,
                                                 num_test_samples=10,
                                                 test_goal_policy_only=False):
        '''Test trajectory generation from VAE.

        Use expert trajectories for trajectory generation.

        expert: Expert object.
        num_test_samples: Number of trajectories to sample.
        test_goal_policy_only: True if only testing trajectory reconstruction
            from goal policy.
        '''
        self.vae_model.eval()
        if self.use_rnn_goal_predictor:
            self.Q_model.eval()
            self.Q_model_linear.eval()

        history_size, batch_size = self.vae_model.history_size, 1

        results = {
                'true_goal': [], 'pred_goal': [],
                'true_traj_state': [], 'true_traj_action': [],
                'pred_traj_state': [], 'pred_traj_action': [],
                'pred_traj_goal': [],
                'pred_context': []}

        # We need to sample expert trajectories to get (s, a) pairs which
        # are required for goal prediction.
        for e in range(num_test_samples):
            batch = expert.sample(batch_size)
            ep_state, ep_action, ep_c, ep_mask = batch
            ep_state = np.array(ep_state, dtype=np.float32)
            ep_action = np.array(ep_action, dtype=np.float32)
            ep_c = np.array(ep_c, dtype=np.int32)

            episode_len = ep_state.shape[1]
            if self.use_rnn_goal_predictor:
                final_goal, pred_goal = self.predict_goal(ep_state,
                                                          ep_action,
                                                          ep_c,
                                                          ep_mask,
                                                          self.num_goals)

                final_goal_numpy = final_goal.data.cpu().numpy().reshape((-1))
                results['pred_goal'].append(final_goal_numpy)

            if self.env_type == 'grid_room':
                true_goal_numpy = np.copy(ep_c)
            else:
                true_goal_numpy = np.zeros((ep_c.shape[0], self.num_goals))
                true_goal_numpy[np.arange(ep_c.shape[0]), ep_c[:, 0]] = 1
            true_goal = Variable(torch.from_numpy(true_goal_numpy).type(
                self.dtype))

            results['true_goal'].append(true_goal_numpy)

            # ep_action is tuple of arrays
            action_var = Variable(
                    torch.from_numpy(np.array(ep_action)).type(self.dtype))

            # Get the initial state
            c = -1 * np.ones((1, self.vae_model.posterior_latent_size),
                             dtype=np.float32)
            if 'grid' in self.env_type:
                x_state_obj = StateVector(ep_state[:, 0, :], self.obstacles)
                x_feat = self.get_state_features(x_state_obj,
                                                 self.args.use_state_features)
            elif self.env_type == 'mujoco':
                x_feat = ep_state[:, 0, :]
                dummy_state = self.env.reset()
                self.env.env.set_state(np.concatenate(
                    (np.array([0.0]), x_feat[0, :8]), axis=0), x_feat[0, 8:17])
                dummy_state = x_feat

            # x is (N, F)
            x = x_feat

            # Add history to state
            if history_size > 1:
                x_hist = -1 * np.ones((x.shape[0], history_size, x.shape[1]),
                                 dtype=np.float32)
                x_hist[:, history_size - 1, :] = x_feat

                x = self.get_history_features(
                        x_hist, self.args.use_velocity_features)

            true_traj, pred_traj = [], []
            pred_traj_goal, pred_context = [], []
            curr_state_arr = ep_state[:, 0, :]

            # Store list of losses to backprop later.
            for t in range(episode_len):
                x_var = Variable(torch.from_numpy(
                    x.reshape((batch_size, -1))).type(self.dtype))

                if self.use_rnn_goal_predictor:
                    if test_goal_policy_only:
                        c_var = None
                    else:
                        c_var = torch.cat([
                            final_goal,
                            Variable(torch.from_numpy(c).type(self.dtype))],
                            dim=1)
                else:
                    c_var = Variable(torch.from_numpy(c).type(self.dtype))
                    if len(true_goal.size()) == 2:
                        c_var = torch.cat([true_goal, c_var], dim=1)
                    elif len(true_goal.size()) == 3:
                        c_var = torch.cat([true_goal[:, t, :], c_var], dim=1)
                    else:
                        raise ValueError("incorrect true goal size")

                if self.args.use_rnn_goal:
                    vae_output = self.vae_model(
                            x_var,
                            c_var,
                            final_goal,
                            only_goal_policy=test_goal_policy_only)
                else:
                    if len(true_goal.size()) == 2:
                        vae_output = self.vae_model(
                                x_var, c_var, true_goal,
                                only_goal_policy=test_goal_policy_only)
                    else:
                        vae_output = self.vae_model(
                                x_var, c_var, true_goal[:, t, ],
                                only_goal_policy=test_goal_policy_only)

                if test_goal_policy_only:
                    # No reparametization happens when training goal policy only
                    vae_reparam_input = None
                elif self.args.use_discrete_vae:
                    # logits
                    vae_reparam_input = (vae_output[2],
                                         self.vae_model.temperature)
                else:
                    # mu, logvar
                    vae_reparam_input = (vae_output[2], vae_output[3])

                # store latent variables (logits or mu)
                if not test_goal_policy_only:
                    pred_context.append(vae_output[2].data.cpu().numpy())

                pred_actions_numpy = vae_output[0].data.cpu().numpy()

                # Store the "true" state
                true_traj.append((ep_state[:, t, :], ep_action[:, t, :]))
                pred_traj.append((curr_state_arr, pred_actions_numpy))
                if (not test_goal_policy_only and
                        self.args.use_separate_goal_policy):
                    pred_actions_2_numpy = vae_output[1].data.cpu().numpy()
                    pred_traj_goal.append(
                            (curr_state_arr, pred_actions_2_numpy.reshape((-1))))

                # Add history
                if history_size > 1:
                    x_hist[:, :(history_size-1), :] = x_hist[:,1:, :]

                if 'grid' in self.env_type:
                    # Get next state from action
                    action = ActionVector(np.argmax(pred_actions_numpy, axis=1))
                    # Get current state object
                    state = StateVector(curr_state_arr, self.obstacles)
                    # Get next state
                    next_state = self.transition_func(state, action, 0)

                    # Update x
                    next_state_features = self.get_state_features(
                            next_state, self.args.use_state_features)

                    if history_size > 1:
                        x_hist[:, history_size - 1, :] = next_state_features
                        x = self.get_history_features(x_hist, self.args.use_velocity_features)
                    else:
                        x[:] = next_state_features

                    # Update current state
                    curr_state_arr = np.array(next_state.coordinates,
                                              dtype=np.float32)
                elif self.env_type == 'mujoco':
                    action = pred_actions_numpy[0, :]
                    next_state, _, done, _ = self.env.step(action)
                    if done:
                        break

                    next_state = np.concatenate(
                            (next_state,
                                np.ones((batch_size, 1)) * (t+1)/(episode_len+1)), axis=1)

                    if history_size > 1:
                        x_hist[:, history_size - 1, :] = next_state
                        x = self.get_history_features(x_hist, self.args.use_velocity_features)
                    else:
                        x[:] = next_state

                    # Update current state
                    curr_state_arr = np.reshape(next_state, (batch_size, -1))

                # update c
                if not test_goal_policy_only:
                    c[:, -self.vae_model.posterior_latent_size:] = \
                            self.vae_model.reparameterize(
                                    *vae_reparam_input).data.cpu()

            '''
            ===== Print predicted trajectories while debugging ====
            def get_traj_from_tuple(x):
                traj = []
                for i in x:
                    traj.append((i[0][0], i[0][1]))
                return traj

            print('True: {}'.format(get_traj_from_tuple(true_traj)))
            print('Pred: {}'.format(get_traj_from_tuple(pred_traj)))
            '''
            true_traj_state_arr = np.array([x[0] for x in true_traj])
            true_traj_action_arr = np.array([x[1] for x in true_traj])
            pred_traj_state_arr = np.array([x[0] for x in pred_traj])
            pred_traj_action_arr = np.array([x[1] for x in pred_traj])

            results['true_traj_state'].append(true_traj_state_arr)
            results['true_traj_action'].append(true_traj_action_arr)
            results['pred_traj_state'].append(pred_traj_state_arr)
            results['pred_traj_action'].append(pred_traj_action_arr)
            results['pred_traj_goal'].append(np.array(pred_traj_goal))
            results['pred_context'].append(np.array(pred_context))

        return results

    def train_fixed_length_epoch(self, epoch, expert, batch_size=1,
                                 episode_len=6, train_goal_policy_only=False):
        '''Train VAE with variable length expert samples.
        '''
        self.set_models_to_train()
        history_size = self.vae_model.history_size
        train_stats = {
            'train_loss': [],
        }

        # TODO: The current sampling process can retrain on a single trajectory
        # multiple times. Will fix it later.
        num_batches = len(expert) // batch_size
        total_epoch_loss, total_epoch_per_step_loss = 0.0, 0.0

        for batch_idx in range(num_batches):
            # Train loss for this batch
            train_loss, train_policy_loss = 0.0, 0.0
            train_KLD_loss, train_policy2_loss = 0.0, 0.0
            ep_timesteps = 0
            true_return = 0.0
            batch = expert.sample(batch_size)

            self.vae_opt.zero_grad()
            if self.use_rnn_goal_predictor:
                self.Q_model_opt.zero_grad()

            ep_state, ep_action, ep_c, ep_mask = batch
            ep_state = np.array(ep_state, dtype=np.float32)
            ep_action = np.array(ep_action, dtype=np.float32)
            ep_c = np.array(ep_c, dtype=np.int32)

            if self.env_type == 'grid_room':
                true_goal_numpy = np.copy(ep_c)
            else: 
                true_goal_numpy = np.zeros((ep_c.shape[0], self.num_goals))
                true_goal_numpy[np.arange(ep_c.shape[0]), ep_c[:, 0]] = 1
            true_goal = Variable(torch.from_numpy(true_goal_numpy)).type(
                    self.dtype)

            if self.use_rnn_goal_predictor:
                final_goal, pred_goal = self.predict_goal(ep_state,
                                                          ep_action,
                                                          ep_c,
                                                          ep_mask,
                                                          self.num_goals)

            # Predict actions i.e. forward prop through q (posterior) and
            # policy network.

            # ep_action will be (N, A)
            action_var = Variable(
                    torch.from_numpy(ep_action).type(self.dtype))

            # Get the initial state (N, C)
            c = -1 * np.ones((batch_size, self.vae_model.posterior_latent_size),
                             dtype=np.float32)
            if 'grid' in self.env_type:
                x_state_obj = StateVector(ep_state[:, 0, :], self.obstacles)
                x_feat = self.get_state_features(x_state_obj,
                                                 self.args.use_state_features)
            elif self.env_type == 'mujoco':
                x_feat = ep_state[:, 0, :]
                self.env.reset()
                self.env.env.set_state(np.concatenate(
                    (np.array([0.0]), x_feat[0, :8]), axis=0), x_feat[0, 8:17])

            # x is (N, F)
            x = x_feat

            # Add history to state
            if history_size > 1:
                x_hist = -1 * np.ones((x.shape[0], history_size, x.shape[1]),
                                      dtype=np.float32)
                x_hist[:, history_size - 1, :] = x_feat
                x = self.get_history_features(
                        x_hist, self.args.use_velocity_features)

            # Store list of losses to backprop later.
            ep_loss, curr_state_arr = [], ep_state[:, 0, :]
            for t in range(episode_len):
                ep_timesteps += 1
                x_var = Variable(torch.from_numpy(
                    x.reshape((batch_size, -1))).type(self.dtype))

                # Append 'c' at the end.
                if args.use_rnn_goal:
                    if train_goal_policy_only:
                        c_var = None
                    else:
                        c_var = torch.cat([final_goal,
                            Variable(torch.from_numpy(c).type(self.dtype))],
                                dim=1)

                    vae_output = self.vae_model(
                            x_var, c_var, final_goal,
                            only_goal_policy=train_goal_policy_only)
                else:
                    if train_goal_policy_only:
                        c_var = None
                    else:
                        c_var = Variable(torch.from_numpy(c).type(self.dtype))
                        if len(true_goal.size()) == 2:
                            c_var = torch.cat([true_goal, c_var], dim=1)
                        elif len(true_goal.size()) == 3:
                            c_var = torch.cat([true_goal[:, t, :], c_var], dim=1)
                        else:
                            raise ValueError("incorrect true goal size")

                    if len(true_goal.size()) == 2:
                        vae_output = self.vae_model(
                                x_var, c_var, true_goal,
                                only_goal_policy=train_goal_policy_only)
                    else:
                        vae_output = self.vae_model(
                                x_var, c_var, true_goal[:, t, ],
                                only_goal_policy=train_goal_policy_only)

                expert_action_var = action_var[:, t, :].clone()
                if train_goal_policy_only:
                    vae_reparam_input = None
                elif self.args.use_discrete_vae:
                    vae_reparam_input = (vae_output[2],
                                         self.vae_model.temperature)
                else:
                    vae_reparam_input = (vae_output[2], vae_output[3])

                if train_goal_policy_only:
                    loss = self.loss_function_using_goal(vae_output[0],
                                                         expert_action_var,
                                                         epoch)
                else:
                    loss, policy_loss, policy2_loss, KLD_loss = \
                            self.loss_function(vae_output[0],
                                               vae_output[1],
                                               expert_action_var,
                                               vae_output[2:],
                                               epoch)

                    train_policy_loss += policy_loss.data[0]
                    if self.args.use_separate_goal_policy:
                        train_policy2_loss += policy2_loss.data[0]
                    train_KLD_loss += KLD_loss.data[0]

                ep_loss.append(loss)
                train_loss += loss.data[0]

                # pred_actions_numpy is (N, A)
                pred_actions_numpy = vae_output[0].data.cpu().numpy()

                if history_size > 1:
                    x_hist[:, :(history_size-1), :] = x_hist[:, 1:, :]

                if 'grid' in self.env_type:
                    # Get next state from action (N,)
                    action = ActionVector(np.argmax(pred_actions_numpy, axis=1))
                    # Get current state
                    state = StateVector(curr_state_arr, self.obstacles)
                    # Get next state
                    #next_state = self.transition_func(state, action, 0)
                    if t < episode_len-1:
                        next_state = StateVector(ep_state[:, t+1, :], self.obstacles)
                    else:
                       break

                    if history_size > 1:
                        x_hist[:, history_size-1] = self.get_state_features(
                                next_state, self.args.use_state_features)
                        x = self.get_history_features(
                                x_hist, self.args.use_velocity_features)
                    else:
                        x[:] = self.get_state_features(
                                next_state, self.args.use_state_features)
                    # Update current state
                    curr_state_arr = np.array(next_state.coordinates,
                                              dtype=np.float32)

                elif self.env_type == 'mujoco':
                    action = pred_actions_numpy[0, :]

                    if ep_c[0, 0] == 0:
                        self.env.env.mode = 'walk'
                    elif ep_c[0, 0] == 1:
                        self.env.env.mode = 'walkback'
                    elif ep_c[0, 0] == 2:
                        self.env.env.mode = 'jump'
                    elif ep_c[0, 0] == 3:
                        if t < episode_len/2:
                            self.env.env.mode = 'jump'
                        else:
                            self.env.env.mode = 'walk'
                    elif ep_c[0, 0] == 4:
                        if t < episode_len/2:
                            self.env.env.mode = 'jump'
                        else:
                            self.env.env.mode = 'walkback'
                    elif ep_c[0, 0] == 5:
                        if t < episode_len/2:
                            self.env.env.mode = 'walk'
                        else:
                            self.env.env.mode = 'jump'
                    elif ep_c[0, 0] == 6:
                        if t < episode_len/2:
                            self.env.env.mode = 'walkback'
                        else:
                            self.env.env.mode = 'jump'

                    #next_state, true_reward, done, _ = self.env.step(action)
                    if t < episode_len-1:
                        next_state = ep_state[:, t+1, :17]
                    else:
                        break
                    #true_return += true_reward
                    #if done:
                    #    break

                    next_state = np.concatenate((
                        next_state, np.ones((batch_size, 1)) * (t+1)/(episode_len+1)), axis=1)

                    if history_size > 1:
                        x_hist[:, history_size-1] = next_state
                        x = self.get_history_features(
                                x_hist, self.args.use_velocity_features)
                    else:
                        x[:] = next_state

                    curr_state_arr = np.reshape(next_state, (batch_size, -1))


                # update c
                if not train_goal_policy_only:
                    c[:, -self.vae_model.posterior_latent_size:] = \
                        self.vae_model.reparameterize(
                                *vae_reparam_input).data.cpu()

            # Calculate the total loss.
            total_loss = ep_loss[0]
            for t in range(1, len(ep_loss)):
                total_loss = total_loss + ep_loss[t]
            total_loss.backward()

            self.vae_opt.step()
            if self.use_rnn_goal_predictor:
                self.Q_model_opt.step()

            # Get the gradients and network weights
            if self.args.log_gradients_tensorboard:
                if train_goal_policy_only:
                    self.log_model_to_tensorboard(models_to_log=(
                            'goal_policy', 'Q_model_linear', 'Q_model'))
                else:
                    self.log_model_to_tensorboard()


            # Update stats
            total_epoch_loss += train_loss
            total_epoch_per_step_loss += (train_loss / episode_len)
            train_stats['train_loss'].append(train_loss)
            self.logger.summary_writer.add_scalar('loss/per_sample',
                                                   train_loss,
                                                   self.train_step_count)
            if not train_goal_policy_only:
                self.logger.summary_writer.add_scalar(
                        'loss/policy_loss_per_sample',
                        train_policy_loss,
                        self.train_step_count)
                self.logger.summary_writer.add_scalar(
                        'loss/KLD_per_sample',
                        train_KLD_loss,
                        self.train_step_count)

                if self.args.use_separate_goal_policy:
                    self.logger.summary_writer.add_scalar(
                            'loss/policy2_loss_per_sample',
                            train_policy2_loss,
                            self.train_step_count)

            if batch_idx % self.args.log_interval == 0:
                if train_goal_policy_only:
                    print('Train Epoch: {} [{}/{}] \t Loss: {:.3f} \t ' \
                          'Timesteps: {} \t Return: {:.3f}'.format(
                        epoch, batch_idx, num_batches, train_loss,
                        ep_timesteps, true_return))
                elif self.args.use_separate_goal_policy:
                    print('Train Epoch: {} [{}/{}] \t Loss: {:.3f} \t ' \
                          'Policy Loss: {:.2f}, \t Policy Loss 2: {:.2f}, \t '\
                          'KLD: {:.2f}, \t Timesteps: {} \t Return: {:.3f}'.format(
                        epoch, batch_idx, num_batches, train_loss,
                        train_policy_loss, train_policy2_loss, train_KLD_loss,
                        ep_timesteps, true_return))
                else:
                    print('Train Epoch: {} [{}/{}] \t Loss: {:.3f} \t ' \
                            'Policy Loss: {:.2f}, \t KLD: {:.2f}, \t ' \
                            'Timesteps: {} \t Return: {:.3f}'.format(
                        epoch, batch_idx, num_batches, train_loss,
                        train_policy_loss, train_KLD_loss, ep_timesteps, true_return))

            self.train_step_count += 1

        # Add other data to logger
        self.logger.summary_writer.add_scalar('loss/per_epoch_all_step',
                                               total_epoch_loss / num_batches,
                                               self.train_step_count)
        self.logger.summary_writer.add_scalar(
                'loss/per_epoch_per_step',
                total_epoch_per_step_loss  / num_batches,
                self.train_step_count)

        return train_stats

    def train_variable_length_epoch(self, epoch, expert, batch_size=1):
        '''Train VAE with variable length expert samples.
        '''
        self.set_models_to_train()
        history_size = self.vae_model.history_size
        train_stats = {
            'train_loss': [],
        }

        # TODO: The current sampling process can retrain on a single trajectory
        # multiple times. Will fix it later.
        num_batches = len(expert) // batch_size
        total_epoch_loss, total_epoch_per_step_loss = 0.0, 0.0

        for batch_idx in range(num_batches):
            # Train loss for this batch
            train_loss, train_policy_loss = 0.0, 0.0
            train_KLD_loss, train_policy2_loss = 0.0, 0.0
            ep_timesteps = 0
            batch = expert.sample(batch_size)

            self.vae_opt.zero_grad()
            if self.use_rnn_goal_predictor:
                self.Q_model_opt.zero_grad()

            ep_state, ep_action, ep_c, ep_mask = batch
            episode_len = len(ep_state[0])

            # After below operation ep_state, ep_action will be a tuple of
            # states, tuple of actions
            ep_state = (ep_state[0])
            ep_action = (ep_action[0])
            ep_c = (ep_c[0])[np.newaxis, :]
            ep_mask = (ep_mask[0])[np.newaxis, :]

            if self.env_type == 'grid_room':
                true_goal_numpy = np.copy(ep_c)
            else:
                true_goal_numpy = np.zeros((self.num_goals))
                true_goal_numpy[int(ep_c[0][0])] = 1
            true_goal = Variable(torch.from_numpy(true_goal_numpy).unsqueeze(
                0).type(self.dtype))


            if self.use_rnn_goal_predictor:
                final_goal, pred_goal = self.predict_goal(ep_state,
                                                          ep_action,
                                                          ep_c,
                                                          ep_mask,
                                                          self.num_goals)

            # Predict actions i.e. forward prop through q (posterior) and
            # policy network.

            # ep_action is tuple of arrays
            action_var = Variable(
                    torch.from_numpy(np.array(ep_action)).type(self.dtype))

            # Get the initial state
            c = -1 * np.ones((1, self.vae_model.posterior_latent_size),
                             dtype=np.float32)
            if 'grid' in self.env_type:
                x_state_obj = State(ep_state[0].tolist(), self.obstacles)
                x_feat = self.get_state_features(x_state_obj,
                                                 self.args.use_state_features)
            elif self.env_type == 'mujoco':
                x_feat = ep_state[0]
                self.env.reset()
                self.env.env.set_state(np.concatenate(
                    (np.array([0.0]), x_feat[:8]), axis=0), x_feat[8:17])

            x = np.reshape(x_feat, (1, -1))

            # Add history to state
            if history_size > 1:
                x_hist = -1 * np.ones((x.shape[0], history_size, x.shape[1]),
                                 dtype=np.float32)
                x_hist[:, history_size - 1, :] = x_feat
                x = self.get_history_features(
                        x_hist, self.args.use_velocity_features)

            # Store list of losses to backprop later.
            ep_loss, curr_state_arr = [], ep_state[0]
            for t in range(episode_len):
                ep_timesteps += 1
                x_var = Variable(torch.from_numpy(
                    x.reshape((1, -1))).type(self.dtype))

                # Append 'c' at the end.
                if args.use_rnn_goal:
                    c_var = torch.cat(
                            [final_goal,
                                Variable(torch.from_numpy(c).type(self.dtype))],
                            dim=1)

                    vae_output = self.vae_model(x_var, c_var, final_goal)
                else:
                    c_var = torch.cat(
                            [true_goal,
                                Variable(torch.from_numpy(c).type(self.dtype))],
                            dim=1)

                    vae_output = self.vae_model(x_var, c_var, true_goal)


                expert_action_var = action_var[t].clone().unsqueeze(0)
                if self.args.use_discrete_vae:
                    vae_reparam_input = (vae_output[2],
                                         self.vae_model.temperature)
                else:
                    vae_reparam_input = (vae_output[2], vae_output[3])

                loss, policy_loss, policy2_loss, KLD_loss = self.loss_function(
                        vae_output[0],
                        vae_output[1],
                        expert_action_var,
                        vae_output[2:],
                        epoch)

                ep_loss.append(loss)
                #loss.backward()
                train_loss += loss.data[0]
                train_policy_loss += policy_loss.data[0]
                if self.args.use_separate_goal_policy:
                    train_policy2_loss += policy2_loss.data[0]
                train_KLD_loss += KLD_loss.data[0]

                pred_actions_numpy = vae_output[0].data.cpu().numpy()

                if history_size > 1:
                    x_hist[:,:(history_size-1),:] = x_hist[:,1:,:]

                if 'grid' in self.env_type:
                    # Get next state from action
                    action = Action(np.argmax(pred_actions_numpy[0, :]))
                    # Get current state
                    state = State(curr_state_arr.tolist(), self.obstacles)
                    # Get next state
                    next_state = self.transition_func(state, action, 0)

                    if history_size > 1:
                        x_hist[:, history_size-1] = self.get_state_features(
                                next_state, self.args.use_state_features)
                        x = self.get_history_features(x_hist, self.args.use_velocity_features)
                    else:
                        x[:] = self.get_state_features(
                                next_state, self.args.use_state_features)
                    # Update current state
                    curr_state_arr = np.array(next_state.coordinates,
                                              dtype=np.float32)

                elif self.env_type == 'mujoco':
                    action = pred_actions_numpy[0, :]
                    next_state, _, done, _ = self.env.step(action)
                    if done:
                        break

                    next_state = np.concatenate((
                        next_state, np.ones((batch_size, 1)) * (t+1)/(episode_len+1)), axis=1)

                    if history_size > 1:
                        x_hist[:, history_size-1] = next_state
                        x = self.get_history_features(
                                x_hist, self.args.use_velocity_features)
                    else:
                        x[:] = next_state

                    curr_state_arr = np.reshape(next_state, (batch_size, -1))


                # update c
                c[:, -self.vae_model.posterior_latent_size:] = \
                    self.vae_model.reparameterize(*vae_reparam_input).data.cpu()

            # Calculate the total loss.
            total_loss = ep_loss[0]
            for t in range(1, len(ep_loss)):
                total_loss = total_loss + ep_loss[t]
            total_loss.backward()

            # Get the gradients and network weights
            if self.args.log_gradients_tensorboard:
                self.log_model_to_tensorboard()

            self.vae_opt.step()
            if self.use_rnn_goal_predictor:
                self.Q_model_opt.step()


            # Update stats
            total_epoch_loss += train_loss
            total_epoch_per_step_loss += (train_loss / episode_len)
            train_stats['train_loss'].append(train_loss)
            self.logger.summary_writer.add_scalar('loss/per_sample',
                                                   train_loss,
                                                   self.train_step_count)
            self.logger.summary_writer.add_scalar('loss/policy_loss_per_sample',
                                                   train_policy_loss,
                                                   self.train_step_count)
            self.logger.summary_writer.add_scalar('loss/KLD_per_sample',
                                                   train_KLD_loss,
                                                   self.train_step_count)
            if self.args.use_separate_goal_policy:
                self.logger.summary_writer.add_scalar(
                        'loss/policy2_loss_per_sample',
                        train_policy2_loss,
                        self.train_step_count)


            if batch_idx % self.args.log_interval == 0:
                if self.args.use_separate_goal_policy:
                    print('Train Epoch: {} [{}/{}] \t Loss: {:.3f} \t ' \
                          'Policy Loss: {:.2f}, \t Policy Loss 2: {:.2f}, \t '\
                          'KLD: {:.2f}, \t Timesteps: {}'.format(
                        epoch, batch_idx, num_batches, train_loss,
                        train_policy_loss, train_policy2_loss, train_KLD_loss,
                        ep_timesteps))
                else:
                    print('Train Epoch: {} [{}/{}] \t Loss: {:.3f} \t ' \
                            'Policy Loss: {:.2f}, \t KLD: {:.2f}, \t ' \
                            'Timesteps: {}'.format(
                        epoch, batch_idx, num_batches, train_loss,
                        train_policy_loss, train_KLD_loss, ep_timesteps))

            self.train_step_count += 1

        # Add other data to logger
        self.logger.summary_writer.add_scalar('loss/per_epoch_all_step',
                                               total_epoch_loss / num_batches,
                                               self.train_step_count)
        self.logger.summary_writer.add_scalar(
                'loss/per_epoch_per_step',
                total_epoch_per_step_loss  / num_batches,
                self.train_step_count)

        return train_stats


    def sample_start_location(self):
        set_diff = list(set(product(tuple(range(7, 13)),
                                    tuple(range(7, 13)))) - set(obstacles))

        return sample_start(set_diff)

    def test_goal_prediction(self, expert, num_test_samples=10):
        '''Test Goal prediction, i.e. is the Q-network (RNN) predicting the
        goal correctly.
        '''
        if self.use_rnn_goal_predictor:
            self.Q_model.eval()
            self.Q_model_linear.eval()
        history_size, batch_size = self.vae_model.history_size, 1

        results = {'true_goal': [], 'pred_goal': []}

        # We need to sample expert trajectories to get (s, a) pairs which
        # are required for goal prediction.
        for e in range(num_test_samples):
            batch = expert.sample(batch_size)
            ep_state, ep_action, ep_c, ep_mask = batch
            # After below operation ep_state, ep_action will be a tuple of
            # states, tuple of actions
            ep_state, ep_action = ep_state[0], ep_action[0]
            ep_c, ep_mask = ep_c[0], ep_mask[0]

            if self.use_rnn_goal_predictor:
                final_goal, pred_goal = self.predict_goal(ep_state,
                                                          ep_action,
                                                          ep_c,
                                                          ep_mask,
                                                          self.num_goals)

                results['pred_goal'].append(final_goal_numpy)

            true_goal_numpy = ep_c[0]
            final_goal_numpy = final_goal.data.cpu().numpy().reshape((-1))

            results['true_goal'].append(true_goal_numpy)

        return results

    def test_models(self, expert, results_pkl_path=None,
                    other_results_dict=None, num_test_samples=100,
                    test_goal_policy_only=False):
        '''Test models by generating expert trajectories.'''
        self.convert_models_to_type(self.dtype)

        results = self.test_generate_trajectory_variable_length(
                expert,
                num_test_samples=num_test_samples,
                test_goal_policy_only=test_goal_policy_only)


        if self.use_rnn_goal_predictor:
            goal_pred_conf_arr = np.zeros((self.num_goals, self.num_goals))
            for i in range(len(results['true_goal'])):
                row = np.argmax(results['true_goal'][i])
                col = np.argmax(results['pred_goal'][i])
                goal_pred_conf_arr[row, col] += 1
            results['goal_pred_conf_arr'] = goal_pred_conf_arr

            #print("Goal prediction confusion matrix:")
            #print(np.array_str(goal_pred_conf_arr, precision=0))

        if other_results_dict is not None:
            # Copy other results dict into the main results
            for k, v in other_results_dict.items():
                results[k] = v

        # Save results in pickle file
        if results_pkl_path is not None:
            with open(results_pkl_path, 'wb') as results_f:
                pickle.dump(results, results_f, protocol=2)
                print('Did save results to {}'.format(results_pkl_path))

def main(args):

    # Create Logger
    if not os.path.exists(os.path.join(args.results_dir, 'log')):
        os.makedirs(os.path.join(args.results_dir, 'log'))
    logger = TensorboardXLogger(os.path.join(args.results_dir, 'log'))

    dtype = torch.FloatTensor
    if args.cuda:
        dtype = torch.cuda.FloatTensor
    vae_train = VAETrain(
        args,
        logger,
        width=11,
        height=15,
        state_size=args.vae_state_size,
        action_size=args.vae_action_size,
        history_size=args.vae_history_size,
        num_goals=args.vae_goal_size,
        use_rnn_goal_predictor=args.use_rnn_goal,
        dtype=dtype,
        env_type=args.env_type,
        env_name=args.env_name
    )

    expert = ExpertHDF5(args.expert_path, args.vae_state_size)
    if 'grid' in args.env_type:
        expert.push(only_coordinates_in_state=True, one_hot_action=True)
    elif args.env_type == 'mujoco':
        expert.push(only_coordinates_in_state=False, one_hot_action=False)
    vae_train.set_expert(expert)

    # expert = Expert(args.expert_path, 2)
    # expert.push()

    if args.run_mode == 'test' or args.run_mode == 'test_goal_pred':
        if args.use_discrete_vae:
            vae_train.vae_model.temperature = 0.1
        assert len(args.checkpoint_path) > 0, \
                'No checkpoint provided for testing'
        vae_train.load_checkpoint(args.checkpoint_path)
        print("Did load models at: {}".format(args.checkpoint_path))
        results_pkl_path = os.path.join(
            args.results_dir,
            'results_' + os.path.basename(args.checkpoint_path))
        # Replace pth file extension with pkl
        results_pkl_path = results_pkl_path[:-4] + '.pkl'
        test_goal_policy_only = True if args.run_mode == 'test_goal_pred' else \
                False
        vae_train.test_models(expert, results_pkl_path=results_pkl_path,
                              num_test_samples=300,
                              test_goal_policy_only=test_goal_policy_only)
    elif args.run_mode == 'train_goal_pred':
        assert args.use_separate_goal_policy == 1, \
                'use_separate_goal_policy flag should be set for goal prediction'
        if len(args.finetune_path) > 0:
            vae_train.load_checkpoint(args.finetune_path)
            assert os.path.dirname(os.path.realpath(args.finetune_path)) != \
                    os.path.dirname(os.path.realpath(args.results_dir)), \
                    "Do not save new results in finetune dir."
        vae_train.train(expert, args.num_epochs, args.batch_size,
                        train_goal_policy_only=True)
    elif args.run_mode == 'train':
        if len(args.finetune_path) > 0:
            vae_train.load_checkpoint(args.finetune_path)
            assert os.path.dirname(os.path.realpath(args.finetune_path)) != \
                    os.path.dirname(os.path.realpath(args.results_dir)), \
                    "Do not save new results in finetune dir."
        vae_train.train(expert, args.num_epochs, args.batch_size)
    else:
        raise ValueError('Incorrect mode to run in.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--checkpoint_every_epoch', type=int, default=10,
                        help='Save models after ever N epochs.')
    # Run on GPU
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='enables CUDA training')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                        help='Disable CUDA training')
    parser.set_defaults(cuda=False)

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging ' \
                              'training status')
    parser.add_argument('--expert-path', default='L_expert_trajectories/',
                        metavar='G',
                        help='path to the expert trajectory files')
    parser.add_argument('--use_rnn_goal', type=int, default=1, choices=[0, 1],
                        help='Use RNN as Q network to predict the goal.')

    parser.add_argument('--use_goal_in_policy', type=int, default=0,
                        choices=[0, 1],
                        help='Give goal to policy network.')

    parser.add_argument('--use_separate_goal_policy', type=int, default=1,
                        choices=[0, 1],
                        help='Use another decoder with goal input.')
    #
    parser.add_argument('--use_history_in_policy', type=int, default=0,
                        choices=[0, 1],
                        help='Use history of states in policy.')

    # Arguments for VAE training
    parser.add_argument('--use_discrete_vae', dest='use_discrete_vae',
                        action='store_true', help='Use Discrete VAE.')
    parser.add_argument('--no-use_discrete_vae', dest='use_discrete_vae',
                        action='store_false', help='Do not Use Discrete VAE.')
    parser.set_defaults(use_discrete_vae=False)

    parser.add_argument('--vae_state_size', type=int, default=2,
                        help='State size for VAE.')
    parser.add_argument('--vae_action_size', type=int, default=4,
                        help='Action size for VAE.')
    parser.add_argument('--vae_goal_size', type=int, default=4,
                        help='Goal size for VAE.')
    parser.add_argument('--vae_history_size', type=int, default=1,
                        help='State history size to use in VAE.')
    parser.add_argument('--vae_context_size', type=int, default=1,
                        help='Context size for VAE.')

    # Temperature
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Discrete VAE temperature')

    # Goal prediction
    parser.add_argument('--flag_goal_pred', type=str, default='last_hidden',
                        choices=['last_hidden', 'sum_all_hidden'],
                        help='Type of network to use for goal prediction')
    parser.add_argument('--train_goal_pred_only', type=int, default=0,
                        choices=[0, 1],
                        help='Train goal prediction part of network only.')

    # Use features
    parser.add_argument('--use_state_features', dest='use_state_features',
                        action='store_true',
                        help='Use features instead of direct (x,y) values in VAE')
    parser.add_argument('--no-use_state_features', dest='use_state_features',
                        action='store_false',
                        help='Do not use features instead of direct (x,y) ' \
                              'values in VAE')
    parser.set_defaults(use_state_features=False)

    # Use velocity in history
    parser.add_argument('--use_velocity_features', dest='use_velocity_features',
                        action='store_true',
                        help='Use velocity instead of direct (x,y) history values in VAE')
    parser.add_argument('--no-use_velocity_features', dest='use_velocity_features',
                        action='store_false',
                        help='Do not use velocity instead of direct (x,y) history' \
                              'values in VAE')
    parser.set_defaults(use_velocity_features=False)

    # Logging flags
    parser.add_argument('--log_gradients_tensorboard',
                        dest='log_gradients_tensorboard', action='store_true',
                        help='Log network weights and grads in tensorboard.')
    parser.add_argument('--no-log_gradients_tensorboard',
                        dest='log_gradients_tensorboard', action='store_true',
                        help='Log network weights and grads in tensorboard.')
    parser.set_defaults(log_gradients_tensorboard=True)

    # Results dir
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory to save final results in.')
    # Checkpoint directory to load pre-trained models.
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Checkpoint path to load pre-trained models.')
    parser.add_argument('--finetune_path', type=str, default='',
                        help='pre-trained models to finetune.')

    # Model arguments
    parser.add_argument('--stacked_lstm', type=int, choices=[0, 1], default=1,
                        help='Use stacked LSTM for Q network.')

    # Action - discrete or continuous
    parser.add_argument('--discrete_action', dest='discrete_action',
                        action='store_true',
                        help='actions are discrete, use BCE loss')
    parser.add_argument('--continuous_action', dest='discrete_action',
                        action='store_false',
                        help='actions are continuous, use MSE loss')
    parser.set_defaults(discrete_action=False)

    # Environment - Grid or Mujoco
    parser.add_argument('--env-type', default='grid', 
                        choices=['grid', 'grid_room', 'mujoco'],
                        help='Environment type Grid or Mujoco.')
    parser.add_argument('--env-name', default=None,
                        help='Environment name if Mujoco.')

    # Mode to run algorithm
    parser.add_argument('--run_mode', type=str, default='train',
                        choices=['train', 'test',
                                 'train_goal_pred', 'test_goal_pred'],
                        help='Mode to run in.')
    # Expert batch episode length
    parser.add_argument('--episode_len', type=int, default=6,
                        help='Fixed episode length for batch training.')

    args = parser.parse_args()
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    main(args)
