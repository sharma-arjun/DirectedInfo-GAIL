import numpy as np
import argparse
import h5py
import os
import pdb
import pickle
import torch

from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

from load_expert_traj import Expert, ExpertHDF5
from load_expert_traj import recursively_save_dict_contents_to_group
from grid_world import State, Action, TransitionFunction
from grid_world import RewardFunction, RewardFunction_SR2
from grid_world import create_obstacles, obstacle_movement, sample_start
from itertools import product
from models import Policy, Posterior

from utils.logger import Logger, TensorboardXLogger
from utils.torch_utils import get_norm_scalar_for_layer
from utils.torch_utils import get_norm_inf_scalar_for_layer

#-----Environment-----#

#if args.expert_path == 'SR2_expert_trajectories/':
#    R = RewardFunction_SR2(-1.0,1.0,width)
#else:
#    R = RewardFunction(-1.0,1.0)


class VAE(nn.Module):
    def __init__(self,
                 state_size=1,
                 action_size=1,
                 latent_size=1,
                 output_size=1,
                 history_size=1,
                 hidden_size=64):
        super(VAE, self).__init__()
        self.history_size = history_size
        self.policy = Policy(state_size=state_size*self.history_size,
                             action_size=action_size,
                             latent_size=latent_size,
                             output_size=output_size,
                             hidden_size=hidden_size,
                             output_activation='sigmoid')

        self.posterior = Posterior(state_size=state_size*self.history_size,
                                   action_size=action_size,
                                   latent_size=latent_size,
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

    def decode(self, x, c):
        action_mean, action_log_std, action_std = self.policy(
                torch.cat((x, c), 1))
        return action_mean

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        c[:,0] = self.reparameterize(mu, logvar)
        return self.decode(x, c), mu, logvar

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
                 dtype=torch.FloatTensor):
        self.args = args
        self.logger = logger
        self.width, self.height = width, height
        self.state_size = state_size
        self.action_size = action_size
        self.history_size = history_size
        self.num_goals = num_goals
        self.dtype = dtype

        self.train_step_count = 0

        # Create models
        self.Q_model = nn.LSTMCell(self.state_size + action_size, 64)
        # Output of linear model num_goals = 4
        self.Q_model_linear = nn.Linear(64, num_goals)
        self.Q_model_linear_softmax = nn.Softmax(dim=1)

        # action_size is 0
        # Hack -- VAE input dim (s + a + latent).
        self.vae_model = VAE(state_size=state_size,
                             action_size=0,
                             latent_size=num_goals+args.vae_context_size,
                             output_size=action_size,
                             history_size=history_size,
                             hidden_size=64)

        self.obstacles, self.transition_func = None, None

        if use_rnn_goal_predictor:
            self.vae_opt = optim.Adam(self.vae_model.parameters(), lr=1e-3)
            self.Q_model_opt = optim.Adam([
                    {'params': self.Q_model.parameters()},
                    {'params': self.Q_model_linear.parameters()},
                ],
                lr=1e-3)
        else:
            self.vae_opt = optim.Adam(self.vae_model.parameters(), lr=1e-3)

        self.create_environment()
        self.expert = None
        self.obstacles, self.set_diff = None, None

    def model_checkpoint_dir(self):
        '''Return the directory to save models in.'''
        return os.path.join(self.args.results_dir, 'checkpoint')

    def model_checkpoint_filename(self, epoch):
        return os.path.join(self.model_checkpoint_dir(),
                            'cp_{}.pth'.format(epoch))

    def create_environment(self):
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

    def convert_models_to_type(self, dtype):
        self.vae_model = self.vae_model.type(dtype)
        self.Q_model = self.Q_model.type(dtype)
        self.Q_model_linear = self.Q_model_linear.type(dtype)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        #MSE = F.mse_loss(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #KLD = 0.5 * torch.sum(mu.pow(2))

        #return MSE + KLD
        return BCE + KLD, BCE, KLD

    def set_models_to_train(self):
        self.Q_model.train()
        self.Q_model_linear.train()
        self.vae_model.train()

    def load_checkpoint(self, checkpoint_path):
        '''Load models from checkpoint.'''
        checkpoint_models = torch.load(checkpoint_path)
        self.vae_model = checkpoint_models['vae_model']
        self.Q_model = checkpoint_models['Q_model']
        self.Q_model_linear = checkpoint_models['Q_model_linear']

    def get_state_features(self, state_obj, use_state_features):
        if use_state_features:
            feat = np.array(state_obj.get_features(), dtype=np.float32)
        else:
            feat = np.array(state_obj.coordinates, dtype=np.float32)
        return feat

    def train(self, expert, num_epochs, batch_size):
        final_train_stats = {
            'train_loss': [],
            'goal_pred_conf_arr': [],
        }
        self.train_step_count = 0
        # Convert models to right type.
        self.convert_models_to_type(self.dtype)

        # Create the checkpoint directory.
        if not os.path.exists(self.model_checkpoint_dir()):
            os.makedirs(self.model_checkpoint_dir())

        for epoch in range(1, num_epochs+1):
            # self.train_epoch(epoch, expert)
            train_stats = self.train_variable_length_epoch(epoch,
                                                           expert,
                                                           batch_size)
            # Update stats for epoch
            final_train_stats['train_loss'].append(train_stats['train_loss'])

            if epoch % 1 == 0:
                results_pkl_path = os.path.join(self.args.results_dir,
                                                'results.pkl')
                self.test_models(expert, results_pkl_path=None)

            if epoch % self.args.checkpoint_every_epoch == 0:
                if self.dtype != torch.FloatTensor:
                    self.convert_models_to_type(torch.FloatTensor)

                 # Loading opt in mac leads to CUDA error?
                model_data = {
                    'vae_model': self.vae_model,
                    'Q_model': self.Q_model,
                    'Q_model_linear': self.Q_model_linear,
                }

                torch.save(model_data, self.model_checkpoint_filename(epoch))
                print("Did save checkpoint file: {}".format(
                    self.model_checkpoint_filename(epoch)))

                if self.dtype != torch.FloatTensor:
                    self.convert_models_to_type(self.dtype)
        
        results_pkl_path = os.path.join(self.args.results_dir, 'results.pkl')
        self.test_models(expert, results_pkl_path=results_pkl_path,
                         other_results_dict={'train_stats': final_train_stats})


    # TODO: Add option to not save gradients for backward pass when not needed
    def predict_goal(self, ep_state, ep_action, ep_c, ep_mask, num_goals):
        '''Predicts goal for 1 expert sample.

        Forward pass through the Q network.

        Return:
            final_goal: Average of all goals predicted at every step.
            pred_goal: Goal predicted at every step of the RNN.
        '''
        # Predict goal for the entire episode i.e., forward prop through Q
        episode_len = len(ep_state)
        ht = Variable(torch.zeros(1, 64).type(self.dtype), requires_grad=True)
        ct = Variable(torch.zeros(1, 64).type(self.dtype), requires_grad=True)
        final_goal = Variable(torch.zeros(1, num_goals).type(self.dtype))
        pred_goal = []
        for t in range(episode_len):
            state_obj = State(ep_state[t].tolist(), self.obstacles)
            state_tensor = torch.from_numpy(self.get_state_features(
                state_obj, self.args.use_state_features)).type(self.dtype)
            action_tensor = torch.from_numpy(ep_action[t]).type(self.dtype)

            # NOTE: Input to LSTM cell needs to be of shape (N, F) where N can
            # be 1.
            inp_tensor = torch.cat((state_tensor, action_tensor), 0).unsqueeze(0)

            ht, ct = self.Q_model(Variable(inp_tensor), (ht, ct))

            output = self.Q_model_linear(ht)
            final_goal = final_goal + output

            # pred_goal.append(Q_model_linear_softmax(output))
            pred_goal.append(output)

        # final_goal = final_goal / episode_len
        final_goal = self.Q_model_linear_softmax(final_goal)

        return final_goal, pred_goal

    def test_generate_trajectory_variable_length(self, expert,
                                                 num_test_samples=10):
        '''Test trajectory generation from VAE.

        Use expert trajectories for trajectory generation.
        '''
        self.vae_model.eval()
        self.Q_model.eval()
        self.Q_model_linear.eval()
        history_size, batch_size = self.vae_model.history_size, 1

        results = {'true_goal': [], 'pred_goal': [],
                   'true_traj': [], 'pred_traj': []}

        # We need to sample expert trajectories to get (s, a) pairs which
        # are required for goal prediction.
        for e in range(num_test_samples):
            batch = expert.sample(batch_size)
            ep_state, ep_action, ep_c, ep_mask = batch

            # After below operation ep_state, ep_action will be a tuple of
            # states, tuple of actions
            ep_state, ep_action = ep_state[0], ep_action[0]
            ep_c, ep_mask = ep_c[0], ep_mask[0]

            episode_len = len(ep_state)
            final_goal, pred_goal = self.predict_goal(ep_state,
                                                      ep_action,
                                                      ep_c,
                                                      ep_mask,
                                                      self.num_goals)
            true_goal_numpy = np.zeros((self.num_goals))
            true_goal_numpy[int(ep_c[0])] = 1
            final_goal_numpy = final_goal.data.cpu().numpy().reshape((-1))

            results['true_goal'].append(true_goal_numpy)
            results['pred_goal'].append(final_goal_numpy)

            # Generate trajectories using VAE.

            # ep_action is tuple of arrays
            action_var = Variable(
                    torch.from_numpy(np.array(ep_action)).type(self.dtype))

            # Get the initial state
            c = -1 * np.ones((1, 1), dtype=np.float32)
            x_state_obj = State(ep_state[0].tolist(), self.obstacles)
            x_feat = self.get_state_features(x_state_obj,
                                             self.args.use_state_features)
            x = np.reshape(x_feat, (1, -1))

            # Add history to state
            if history_size > 1:
                x = -1 * np.ones((x.shape[0], history_size, x.shape[1]),
                                 dtype=np.float32)
                x[:, history_size - 1, :] = x_feat

            true_traj, pred_traj = [], []
            curr_state_arr = ep_state[0]

            # Store list of losses to backprop later.
            for t in range(episode_len):
                x_var = Variable(torch.from_numpy(
                    x.reshape((1, -1))).type(self.dtype))
                c_var = torch.cat([
                    final_goal,
                    Variable(torch.from_numpy(c).type(self.dtype))], dim=1)

                pred_actions_tensor, mu, logvar = self.vae_model(x_var, c_var)
                pred_actions_numpy = pred_actions_tensor.data.cpu().numpy()

                # Store the "true" state
                true_traj.append((ep_state[t], ep_action[t]))
                pred_traj.append((curr_state_arr,
                                  pred_actions_numpy.reshape((-1))))

                if history_size > 1:
                    x[:, :(history_size-1), :] = x[:,1:, :]

                # Get next state from action
                action = Action(np.argmax(pred_actions_numpy[0, :]))
                # Get current state object
                state = State(curr_state_arr.tolist(), self.obstacles)
                # Get next state
                next_state = self.transition_func(state, action, 0)

                # Update x
                next_state_features = self.get_state_features(
                        next_state, self.args.use_state_features)
                if history_size > 1:
                    x[:, history_size - 1, :] = next_state_features
                else:
                    x[:] = next_state_features

                # update c
                c[:, 0] = self.vae_model.reparameterize(mu, logvar).data.cpu()

                # Update current state
                curr_state_arr = np.array(next_state.coordinates,
                                          dtype=np.float32)

            results['true_traj'].append(np.array(true_traj))
            results['pred_traj'].append(np.array(pred_traj))
        return results

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
        batch_size = 1
        num_batches = len(expert) // batch_size
        total_epoch_loss, total_epoch_per_step_loss = 0.0, 0.0

        for batch_idx in range(num_batches):
            # Train loss for this batch
            train_loss, train_BCE_loss, train_KLD_loss = 0.0, 0.0, 0.0
            batch = expert.sample(batch_size)

            self.vae_opt.zero_grad()
            self.Q_model_opt.zero_grad()

            ep_state, ep_action, ep_c, ep_mask = batch
            episode_len = len(ep_state[0])

            # After below operation ep_state, ep_action will be a tuple of
            # states, tuple of actions
            ep_state = (ep_state[0])
            ep_action = (ep_action[0])
            ep_c = (ep_c[0])[np.newaxis, :]
            ep_mask = (ep_mask[0])[np.newaxis, :]

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
            c = -1 * np.ones((1, 1), dtype=np.float32)
            x_state_obj = State(ep_state[0].tolist(), self.obstacles)
            x_feat = self.get_state_features(x_state_obj,
                                             self.args.use_state_features)
            x = np.reshape(x_feat, (1, -1))

            # Add history to state
            if history_size > 1:
                x = -1 * np.ones((x.shape[0], history_size, x.shape[1]),
                                 dtype=np.float32)
                x[:, history_size - 1, :] = x_feat

            # Store list of losses to backprop later.
            ep_loss, curr_state_arr = [], ep_state[0]
            for t in range(episode_len):
                x_var = Variable(torch.from_numpy(
                    x.reshape((1, -1))).type(self.dtype))
                c_var = torch.cat(
                        [final_goal,
                         Variable(torch.from_numpy(c).type(self.dtype))], dim=1)

                pred_actions_tensor, mu, logvar = self.vae_model(x_var, c_var)

                expert_action_var = action_var[t].clone().unsqueeze(0)

                loss, BCE_loss, KLD_loss = self.loss_function(
                        pred_actions_tensor,
                        expert_action_var,
                        mu,
                        logvar)
                ep_loss.append(loss)
                train_loss += loss.data[0]
                train_BCE_loss += BCE_loss.data[0]
                train_KLD_loss += KLD_loss.data[0]

                pred_actions_numpy = pred_actions_tensor.data.cpu().numpy()

                if history_size > 1:
                    x[:,:(history_size-1),:] = x[:,1:,:]

                # Get next state from action
                action = Action(np.argmax(pred_actions_numpy[0, :]))
                # Get current state
                state = State(curr_state_arr.tolist(), self.obstacles)
                # Get next state
                next_state = self.transition_func(state, action, 0)

                if history_size > 1:
                    x[:, history_size-1] = self.get_state_features(
                            next_state, self.args.use_state_features)
                else:
                    x[:] = self.get_state_features(next_state,
                                                   self.args.use_state_features)
                # Update current state
                curr_state_arr = np.array(next_state.coordinates,
                                          dtype=np.float32)

                # update c
                c[:, 0] = self.vae_model.reparameterize(mu, logvar).data.cpu()

            # Calculate the total loss.
            total_loss = ep_loss[0]
            for t in range(1, len(ep_loss)):
                total_loss = total_loss + ep_loss[t]
            total_loss.backward()

            # Get the gradients and network weights
            # TODO: ADd gradients
            if self.args.log_gradients_tensorboard:
                vae_model_l2_norm, vae_model_grad_l2_norm = -10000.0, -10000.0
                for param in self.vae_model.parameters():
                    vae_model_l2_norm = np.maximum(
                            vae_model_l2_norm,
                            get_norm_scalar_for_layer(param, 2))
                    if param.grad is not None:
                        vae_model_grad_l2_norm = np.maximum(
                                vae_model_grad_l2_norm,
                                get_norm_inf_scalar_for_layer(param.grad))
                self.logger.summary_writer.add_scalar(
                        'weight/vae_model_l2',
                         vae_model_l2_norm,
                         self.train_step_count)
                self.logger.summary_writer.add_scalar(
                        'grad/vae_model_l2',
                         vae_model_grad_l2_norm,
                         self.train_step_count)

                Q_model_l2_norm, Q_model_l2_grad_norm = -10000.0, -10000.0
                for param in self.Q_model_linear.parameters():
                    Q_model_l2_norm = np.maximum(
                            Q_model_l2_norm,
                            get_norm_scalar_for_layer(param, 2))
                    if param.grad is not None:
                        Q_model_l2_grad_norm = np.maximum(
                                Q_model_l2_grad_norm,
                                get_norm_inf_scalar_for_layer(param.grad))
                self.logger.summary_writer.add_scalar(
                        'weight/Q_model_l2',
                         Q_model_l2_norm,
                         self.train_step_count)
                self.logger.summary_writer.add_scalar(
                        'grad/Q_model_l2',
                         Q_model_l2_grad_norm,
                         self.train_step_count)


            self.vae_opt.step()
            self.Q_model_opt.step()

            # pdb.set_trace()

            # Update stats
            total_epoch_loss += train_loss
            total_epoch_per_step_loss += (train_loss / episode_len)
            train_stats['train_loss'].append(train_loss)
            self.logger.summary_writer.add_scalar('loss/per_sample',
                                                   train_loss,
                                                   self.train_step_count)
            self.logger.summary_writer.add_scalar('loss/BCE_per_sample',
                                                   train_BCE_loss,
                                                   self.train_step_count)
            self.logger.summary_writer.add_scalar('loss/KLD_per_sample',
                                                   train_KLD_loss,
                                                   self.train_step_count)


            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{}] \t Loss: {:.3f}   ' \
                        'BCE Loss: {:.2f},    KLD: {:.2f}'.format(
                    epoch, batch_idx, num_batches, train_loss,
                    train_BCE_loss, train_KLD_loss))

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


    def train_batch_epoch(self, epoch, batch_size=10):
        self.vae_model.train()
        history_size = model.history_size
        train_loss = 0
        for batch_idx in range(10): # 10 batches per epoch
            batch = self.expert.sample(batch_size)
            print("Batch len: {}".format(len(batch.state[0])))
            print("Batch len: {}".format(len(batch.state[1])))
            x_data = torch.Tensor(np.array(batch.state))
            N = x_data.size(1)
            x = -1*torch.ones(x_data.size(0), history_size, x_data.size(2))
            x[:, (history_size-1), :] = x_data[:, 0, :]

            a = Variable(torch.Tensor(np.array(batch.action)))

            # Context variable is in one-hot, convert it to integer
            _, c2 = torch.Tensor(np.array(batch.c)).max(2) # , (N, T)
            c2 = c2.float()[:,0].unsqueeze(1)
            c1 = -1*torch.ones(c2.size())
            c = torch.cat((c1, c2), 1)

            #c_t0 = Variable(c[:,0].clone().view(c.size(0), 1))

            if self.args.cuda:
                a = a.cuda()
                #c_t0 = c_t0.cuda()

            self.vae_opt.zero_grad()
            for t in range(N):
                #x_t0 = Variable(x[:,0,:].clone().view(x.size(0), x.size(2)))
                #x_t1 = Variable(x[:,1,:].clone().view(x.size(0), x.size(2)))
                #x_t2 = Variable(x[:,2,:].clone().view(x.size(0), x.size(2)))
                #x_t3 = Variable(x[:,3,:].clone().view(x.size(0), x.size(2)))
                input_x = Variable(x[:,:,:].view(x.size(0),
                                   history_size*x.size(2)).clone())
                c_t0 = Variable(c)

                if self.args.cuda:
                    input_x = input_x.cuda()
                    #x_t0 = x_t0.cuda()
                    #x_t1 = x_t1.cuda()
                    #x_t2 = x_t2.cuda()
                    #x_t3 = x_t3.cuda()
                    c_t0 = c_t0.cuda()


                recon_batch, mu, logvar = self.vae_model(input_x, c_t0)
                loss, BCE_loss, KLD_loss = self.loss_function(
                        recon_batch, a[:,t,:], mu, logvar)
                loss.backward()
                train_loss += loss.data[0]

                pred_actions = recon_batch.data.cpu().numpy()

                x[:,:3,:] = x[:,1:,:]
                # get next state and update x
                for b_id in range(pred_actions.shape[0]):
                    action = Action(np.argmax(pred_actions[b_id,:]))
                    state = State(x[b_id,3,:].cpu().numpy(), self.obstacles)
                    next_state = self.transition_func(state, action, 0)
                    x[b_id,3,:] = torch.Tensor(next_state.state)

                # update c
                c[:,0] = self.vae_model.reparameterize(mu, logvar).data.cpu()

            self.vae_opt.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * self.args.batch_size, 200.0,
                    100. * batch_idx / 20.0,
                    loss.data[0] / self.args.batch_size))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / 200.0))

    def sample_start_location(self):
        set_diff = list(set(product(tuple(range(7, 13)),
                                    tuple(range(7, 13)))) - set(obstacles))

        return sample_start(set_diff)

    def test_goal_prediction(self, expert, num_test_samples=10):
        '''Test Goal prediction, i.e. is the Q-network (RNN) predicting the
        goal correctly.
        '''
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

            final_goal, pred_goal = self.predict_goal(ep_state,
                                                      ep_action,
                                                      ep_c,
                                                      ep_mask,
                                                      self.num_goals)
            true_goal_numpy = ep_c[0]
            final_goal_numpy = final_goal.data.cpu().numpy().reshape((-1))

            results['true_goal'].append(true_goal_numpy)
            results['pred_goal'].append(final_goal_numpy)
        return results

    def test_models(self, expert, results_pkl_path=None,
                    other_results_dict=None):
        '''Test models by generating expert trajectories.'''
        results = self.test_generate_trajectory_variable_length(
                expert, num_test_samples=100)

        goal_pred_conf_arr = np.zeros((self.num_goals, self.num_goals))
        for i in range(len(results['true_goal'])):
            row = np.argmax(results['true_goal'][i])
            col = np.argmax(results['pred_goal'][i])
            goal_pred_conf_arr[row, col] += 1
        final_train_stats['goal_pred_conf_arr'].append(goal_pred_conf_arr)

        print("Goal prediction confusion matrix:")
        print(np.array_str(goal_pred_conf_arr, precision=0))

        if other_results_dict is not None:
            # Copy other results dict into the main results
            for k, v in other_results_dict.items():
                results[k] = v

        # Save results in pickle file
        if results_pkl_path is not None:
            with open(results_pkl_path, 'wb') as results_f:
                pickle.dump(results, results_f, protocol=2)
                print('Did save results to {}'.format(results_pkl_path))

    # I'm pretty sure this doesn't work.
    def test(self, expert):
        self.vae_model.eval()
        history_size = self.vae_model.history_size
        #test_loss = 0

        for _ in range(20):
            c = expert.sample_c()
            N = c.shape[0]
            c = np.argmax(c[0,:])

            #if args.expert_path == 'SR_expert_trajectories/':
            #    if c == 1:
            #        half = 0
            #    elif c == 3:
            #        half = 1
            #elif args.expert_path == 'SR2_expert_trajectories/':
            #    half = c
            #if args.expert_path == 'SR_expert_trajectories/' or \
            #        args.expert_path == 'SR2_expert_trajectories/':
            #    if half == 0: # left half
            #        set_diff = list(set(product(tuple(range(0, (width/2)-3)),
            #                           tuple(range(1, height)))) - set(obstacles))
            #    elif half == 1: # right half
            #        set_diff = list(set(product(
            #           tuple(range(width/2, width-2)),
            #           tuple(range(2, height)))) - set(obstacles))
            #else:
            #    set_diff = list(set(product(tuple(range(3, width-3)), repeat=2)) \
            #           - set(obstacles))
            start_loc = self.sample_start_location()
            s = State(start_loc, self.obstacles)
            R.reset()
            c = torch.from_numpy(np.array([-1.0,c])).unsqueeze(0).float()

            print('c is '.format(c[0,1]))
            c = Variable(c)

            x = -1*torch.ones(1, history_size, 2)

            if args.cuda:
                x = x.cuda()
                c = c.cuda()

            for t in range(N):

                x[:,:(history_size-1),:] = x[:,1:,:]
                curr_x = torch.from_numpy(s.state).unsqueeze(0)
                if args.cuda:
                    curr_x = curr_x.cuda()

                x[:,(history_size-1),:] = curr_x

                #x_t0 = Variable(x[:,0,:])
                #x_t1 = Variable(x[:,1,:])
                #x_t2 = Variable(x[:,2,:])
                #x_t3 = Variable(x[:,3,:])

                input_x = Variable(x[:,:,:].view(x.size(0),
                                                 history_size*x.size(2)).clone())

                mu, logvar = model.encode(input_x, c)
                c[:,0] = model.reparameterize(mu, logvar)
                pred_a = model.decode(input_x, c).data.cpu().numpy()
                pred_a = np.argmax(pred_a)
                print("Pred a {}".format(pred_a))
                next_s = Transition(s, Action(pred_a), R.t)

                s = next_s

                #test_loss += loss_function(recon_batch, data, mu, logvar).data[0]


        #test_loss /= len(test_loader.dataset)
        #print('====> Test set loss: {:.4f}'.format(test_loss))


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
        width=21,
        height=21,
        state_size=args.vae_state_size,
        action_size=args.vae_action_size,
        history_size=args.vae_history_size,
        num_goals=4,
        use_rnn_goal_predictor=args.use_rnn_goal,
        dtype=dtype
    )

    expert = ExpertHDF5(args.expert_path, 2)
    expert.push(only_coordinates_in_state=True, one_hot_action=True)
    vae_train.set_expert(expert)

    # expert = Expert(args.expert_path, 2)
    # expert.push()

    if len(args.checkpoint_path) > 0:
        vae_train.load_checkpoint(args.checkpoint_path)
        print("Did load models at: {}".format(args.checkpoint_path))
        results_pkl_path = os.path.join(
                args.results_dir,
                'results_' + os.path.basename(args.checkpoint_path))
        # Replace pth file extension with pkl
        results_pkl_path = results_pkl_path[:-4] + '.pkl'
        vae_train.test_models(expert, results_pkl_path=results_pkl_path)
    else:
        vae_train.train(expert, args.num_epochs, args.batch_size)

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

    # Arguments for VAE training
    parser.add_argument('--vae_state_size', type=int, default=2,
                        help='State size for VAE.')
    parser.add_argument('--vae_action_size', type=int, default=4,
                        help='Action size for VAE.')
    parser.add_argument('--vae_history_size', type=int, default=1,
                        help='State history size to use in VAE.')
    parser.add_argument('--vae_context_size', type=int, default=1,
                        help='Context size for VAE.')

    # Use features
    parser.add_argument('--use_state_features', dest='use_state_features',
                        action='store_true',
                        help='Use features instead of direct (x,y) values in VAE')
    parser.add_argument('--no-use_state_features', dest='use_state_features',
                        action='store_false',
                        help='Do not use features instead of direct (x,y) ' \
                              'values in VAE')
    parser.set_defaults(use_state_features=False)

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
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Checkpoint path to load pre-trained models.')

    args = parser.parse_args()
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    main(args)
