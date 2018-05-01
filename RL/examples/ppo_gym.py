import argparse
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from torch.autograd import Variable
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent

Tensor = DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch PPO example')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--mode-list', nargs='+', required=True,
                    help='mode-list - permutation of walk, walkback, jump')
parser.add_argument('--num-steps-per-mode', type=int, default=333, metavar='N',
                    help='number of steps per mode (default: 333)')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=0, metavar='G',
                    help='log std for the policy (default: 0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='learning rate (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--policy-list', nargs='*',
                    help="policies to use with mixed expert trajectory generation")
parser.add_argument('--state-type', default="no_context", metavar='G',
                    help='Type of state - no context, context, decayed context')
parser.add_argument('--traj-save-dir', metavar='G',
                    help='save directory for expert h5')
parser.add_argument('--save-model-path', metavar='G', default='learned_models',
                    help='save directory for expert h5')
parser.add_argument('--jump-thresh', type=float, default=1.3, metavar='N',
                    help='threshold for jump reward')
args = parser.parse_args()


def env_factory(thread_id, mode=None):
    env = gym.make(args.env_name)
    env.seed(args.seed + thread_id)
    if hasattr(env.env, 'mode'):
        print('Setting mode to', mode)
        env.env.mode = mode
    if hasattr(env.env, 'jump_thresh'):
        env.env.jump_thresh = args.jump_thresh
    return env


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)

env_dummy = env_factory(0)
state_dim = env_dummy.observation_space.shape[0]
is_disc_action = len(env_dummy.action_space.shape) == 0
ActionTensor = LongTensor if is_disc_action else DoubleTensor

if args.traj_save_dir:
    if not os.path.exists(args.traj_save_dir):
        os.makedirs(args.traj_save_dir)

if args.save_model_path:
    if not os.path.exists(os.path.join(assets_dir(), args.save_model_path)):
        os.makedirs(os.path.join(assets_dir(), args.save_model_path))

if args.state_type == 'decayed_context':
    extra_dim = 2
elif args.state_type == 'context':
    extra_dim = 1
else:
    extra_dim = 0

running_state_list = []
for _ in range(len(args.mode_list)):
    running_state = ZFilter((state_dim+extra_dim,), clip=5)
    running_state_list.append(running_state)

# running_reward = ZFilter((1,), demean=False, clip=10)

"""define actor and critic"""
if args.policy_list is not None:
    # load policy nets here
    policy_list = []
    running_state_list = []
    for path_name in args.policy_list:
        with open(path_name, "rb") as f_weight_file:
            policy_net, value_net, running_state = pickle.load(f_weight_file)
        if use_gpu:
            policy_net = policy_net.cuda()
        policy_list.append(policy_net)
        running_state_list.append(running_state)

else:
    policy_list = []
    value_net_list = []

    if args.model_path is None:
        if is_disc_action:
            for _ in range(len(args.mode_list)):
                policy_list.append(DiscretePolicy(state_dim+extra_dim, env_dummy.action_space.n))
        else:
            for _ in range(len(args.mode_list)):
                policy_net = Policy(state_dim+extra_dim, env_dummy.action_space.shape[0], log_std=args.log_std)

        for _ in range(len(args.mode_list)):
            value_net_list.append(Value(state_dim+extra_dim))
    else:
        running_state_list = []
        for model_path in args.model_path_list:
            policy_net, value_net, running_state = pickle.load(open(model_path, "rb"))
            policy_list.append(policy_list)
            value_net_list.append(value_net)
            running_state_list.append(running_state)
    if use_gpu:
        for policy_net, value_net in zip(policy_list, value_net_list):
            policy_net = policy_net.cuda()
            value_net = value_net.cuda()

    """create agent"""
    agent = Agent(env_factory, policy_list, running_state_list=running_state_list, render=args.render,
                 num_threads=args.num_threads, mode_list=args.mode_list, state_type=args.state_type, 
                 num_steps_per_mode=args.num_steps_per_mode)

    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)

del env_dummy

# optimization epoch number and batch size for PPO
optim_epochs = 5
optim_batch_size = 512


def update_params(batch, i_iter):
    states = torch.from_numpy(np.stack(batch.state))
    actions = torch.from_numpy(np.stack(batch.action))
    rewards = torch.from_numpy(np.stack(batch.reward))
    masks = torch.from_numpy(np.stack(batch.mask).astype(np.float64))
    if use_gpu:
        states, actions, rewards, masks = states.cuda(), actions.cuda(), rewards.cuda(), masks.cuda()
    values = value_net(Variable(states, volatile=True)).data
    fixed_log_probs = policy_net.get_log_prob(Variable(states, volatile=True), Variable(actions)).data

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, use_gpu)

    lr_mult = max(1.0 - float(i_iter) / args.max_iter_num, 0)

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).cuda() if use_gpu else LongTensor(perm)

        states, actions, returns, advantages, fixed_log_probs = \
            states[perm], actions[perm], returns[perm], advantages[perm], fixed_log_probs[perm]

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, lr_mult, args.learning_rate, args.clip_epsilon, args.l2_reg)


def train_loop():
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size)
        t0 = time.time()
        update_params(batch, i_iter)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            if use_gpu:
                policy_net.cpu(), value_net.cpu()
            pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(assets_dir(), args.save_model_path + '/{}_{}_{}_{}_{}_ppo.p'.format(args.env_name, '_'.join(args.mode_list),
                             str(args.jump_thresh), str(i_iter), str(log['avg_reward']))), 'wb'))
            if use_gpu:
                policy_net.cuda(), value_net.cuda()

def gen_traj_loop():
    n = 1
    agent = Agent(env_factory, policy_list[0], running_state=running_state_list[0], render=args.render,
                  num_threads=args.num_threads, mode_list=args.mode_list, state_type=args.state_type,
                  num_steps_per_mode=args.num_steps_per_mode)
    
    env_data_dict = {'num_goals': 3}
    expert_data_dict = {}
    i_iter = 0
    print('Writing to h5 file ...')

    while i_iter < n:
        #vid_folder = str(i_iter)
        vid_folder = None
        path_key = str(i_iter) + '_0'
        returned_dict, save_flag = agent.generate_mixed_expert_trajs(policy_list, running_state_list, vid_folder=vid_folder)
        if save_flag:
            expert_data_dict[path_key] = returned_dict
            i_iter += 1
            print(i_iter)

    save_expert_traj_dict_to_h5(expert_data_dict, args.traj_save_dir)

    

if args.policy_list is None:
    train_loop()
else:
    gen_traj_loop()
