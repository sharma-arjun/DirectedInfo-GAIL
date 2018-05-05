import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
from utils.math import normal_log_density

def init_fanin(tensor):
    fanin = tensor.size(1)
    v = 1.0 / np.sqrt(fanin)
    init.uniform(tensor, -v, v)


class PhasePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128, 128), n_layers=2, activation='tanh', log_std=0):
        super(PhasePolicy, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size_1 = hidden_size[0]
        self.hidden_size_2 = hidden_size[-1]
        self.n_layers = n_layers
        self.is_disc_action = False

        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.control_hidden_list = []
        self.control_h2o_list = []

    
        self.l_00 = nn.Linear(state_dim, self.hidden_size_1)
        self.h2o_0 = nn.Linear(self.hidden_size_2, action_dim)

        self.l_10 = nn.Linear(state_dim, self.hidden_size_1)
        self.h2o_1 = nn.Linear(self.hidden_size_2, action_dim)


        init_fanin(self.l_00.weight)
        init_fanin(self.l_10.weight)

        init.uniform(self.h2o_0.weight,-3e-3, 3e-3)
        init.uniform(self.h2o_0.bias,-3e-3, 3e-3)
        init.uniform(self.h2o_1.weight,-3e-3, 3e-3)
        init.uniform(self.h2o_1.bias,-3e-3, 3e-3)

        if n_layers == 2:
            

            self.l_01 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
            self.l_11 = nn.Linear(self.hidden_size_1, self.hidden_size_2)

            init_fanin(self.l_01.weight)
            init_fanin(self.l_11.weight)


        self.control_hidden_list.append([self.l_00, self.l_10])
        if n_layers == 2:
            self.control_hidden_list.append([self.l_01, self.l_11])

        self.control_h2o_list = [self.h2o_0, self.h2o_1]

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)


    def forward(self, x, phase):

        phase = phase.data
        control_hidden_list = self.control_hidden_list
        control_h2o_list = self.control_h2o_list

        w0_h1 = Variable(phase.repeat(1, self.hidden_size_1))
        w1_h1 = Variable((1.0 - phase).repeat(1, self.hidden_size_1))

        if self.n_layers == 2:
            w0_h2 = Variable(phase.repeat(1, self.hidden_size_2))
            w1_h2 = Variable((1.0 - phase).repeat(1, self.hidden_size_2))


        w0_o = Variable(phase.repeat(1, self.action_dim))
        w1_o = Variable((1.0 - phase).repeat(1, self.action_dim))
                
        h_0 = self.activation(w0_h1*control_hidden_list[0][0](x) + w1_h1*control_hidden_list[0][1](x))

        if self.n_layers == 2:
            h_1 = self.activation(w0_h2*control_hidden_list[1][0](h_0) + w1_h2*control_hidden_list[1][1](h_0))
            action_mean = w0_o*control_h2o_list[0](h_1) + w1_o*control_h2o_list[1](h_1)

        else:
            action_mean = w0_o*control_h2o_list[0](h_0) + w1_o*control_h2o_list[1](h_0)

        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, x, phase):
        action_mean, _, action_std = self.forward(x, phase)
        action = torch.normal(action_mean, action_std)

        return action.data

    def get_log_prob(self, x, phase, actions):
        action_mean, action_log_std, action_std = self.forward(x, phase)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def convert_to_cuda(self):
        for n in range(self.n_layers):
            for weight in self.control_hidden_list[n]:
                weight.cuda()

        for weight in self.control_h2o_list:
            weight.cuda()

        self.action_log_std = nn.Parameter(self.action_log_std.cuda().data)

    def convert_to_cpu(self):
        for n in range(self.n_layers):
            for weight in self.control_hidden_list[n]:
                weight.cpu()

        for weight in self.control_h2o_list:
            weight.cpu()

        self.action_log_std = nn.Parameter(self.action_log_std.cpu().data)
