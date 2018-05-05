import multiprocessing
from utils.replay_memory import Memory
from utils.torch import *
from torch.autograd import Variable
import math
import time
from gym import wrappers


def activity_map(mode):
    activity_dict = {'walk': 0.0, 'walkback': 1.0, 'jump': 2.0}
    #activity_dict = {'walk': 0.0, 'walkback': 1.0, 'jump': 2.0, 'rest': 3.0}
    #activity_dict = {'forward': 0.0, 'backward': 1.0}
    one_hot = np.zeros((len(activity_dict),))
    one_hot[int(activity_dict[mode])] = 1.0

    #return activity_dict[mode]
    return one_hot


def collect_samples(pid, queue, env, policy, custom_reward, mean_action, tensor,
                    render, running_state, update_rs, min_batch_size, mode_list, state_type,
                    num_steps_per_mode, use_phase):
    torch.randn(pid, )
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0
    max_t = num_steps_per_mode * len(mode_list) - 1

    while num_steps < min_batch_size:
        state = env.reset()
        if state_type == 'decayed_context':
            state = np.concatenate((state, np.array([1.0]),
                                    activity_map(mode_list[0]),
                                    activity_map(mode_list[min(1, len(mode_list)-1)])), axis=0)
        elif state_type == 'context':
            state = np.concatenate((state, activity_map(mode_list[0]),
                                    activity_map(mode_list[min(1, len(mode_list)-1)])), axis=0)

        if running_state is not None:
            state = running_state(state, update=update_rs)
        reward_episode = 0

        for t in range(10000):
            phase = t/max_t
            curr_mode_id = t // num_steps_per_mode
            if t % num_steps_per_mode == 0:
                if hasattr(env.env, 'mode'):
                    env.env.mode = mode_list[curr_mode_id]
            state_var = Variable(tensor(state).unsqueeze(0), volatile=True)
            phase_var = Variable(tensor(np.array([phase])), volatile=True)
            if mean_action:
                if use_phase:
                    action = policy(state_var, phase_var)[0].data[0].numpy()
                else:
                    action = policy(state_var)[0].data[0].numpy()
            else:
                if use_phase:
                    action = policy.select_action(state_var, phase_var)[0].numpy()
                else:
                    action = policy.select_action(state_var)[0].numpy()

            action = int(action) if policy.is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            reward_episode += reward

            next_mode_id = min(t+1, max_t) // num_steps_per_mode

            if state_type == 'decayed_context':
                next_state = np.concatenate((next_state, 
                                             np.array([1/((t % num_steps_per_mode) + 1)]),
                                             activity_map(mode_list[next_mode_id]),
                                             activity_map(mode_list[min(next_mode_id+1, len(mode_list)-1)])), axis=0)
            elif state_type == 'context':
                next_state = np.concatenate((next_state,
                                             activity_map(mode_list[next_mode_id]),
                                             activity_map(mode_list[min(next_mode_id+1, len(mode_list)-1)])), axis=0)

            if running_state is not None:
                next_state = running_state(next_state, update=update_rs)

            if custom_reward is not None:
                reward = custom_reward(state, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            if t == num_steps_per_mode * len(mode_list) - 1:
                done = True

            mask = 0 if done else 1

            memory.push(state, action, mask, next_state, reward, np.array([phase]))

            if render:
                env.render()
            if done:
                break

            state = next_state

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, env_factory, policy, custom_reward=None, mean_action=False, render=False,
                 tensor_type=torch.DoubleTensor, running_state=None, num_threads=1, mode_list=None,
                 state_type=None, num_steps_per_mode=333, use_phase=False):
        self.env_factory = env_factory
        self.policy = policy
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.tensor = tensor_type
        self.num_threads = num_threads
        self.mode_list = mode_list
        self.state_type = state_type
        self.num_steps_per_mode = num_steps_per_mode
        self.use_phase = use_phase
        self.env_list = []
        for i in range(num_threads):
            if mode_list:
                self.env_list.append(self.env_factory(i, mode_list[0]))
            else:
                self.env_list.append(self.env_factory(i))

    def collect_samples(self, min_batch_size):
        t_start = time.time()
        if use_gpu:
            if self.use_phase:
                self.policy.convert_to_cpu()
            else:
                self.policy.cpu()
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env_list[i + 1], self.policy, self.custom_reward, self.mean_action, self.tensor,
                           False, self.running_state, False, thread_batch_size, self.mode_list, self.state_type,
                           self.num_steps_per_mode, self.use_phase)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env_list[0], self.policy, self.custom_reward, self.mean_action, self.tensor,
                                      self.render, self.running_state, True, thread_batch_size, self.mode_list, self.state_type,
                                      self.num_steps_per_mode, self.use_phase)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        if use_gpu:
            if self.use_phase:
                self.policy.convert_to_cuda()
            else:
                self.policy.cuda()
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log


    def generate_mixed_expert_trajs(self, policy_list, running_state_list, vid_folder=None):

        assert(len(policy_list) == len(self.mode_list))
        N = len(self.mode_list)
        num_steps_per_policy = self.num_steps_per_mode
        # currently not using multiprocessing for this part
        env = self.env_list[0]
        if vid_folder != None:
            env = wrappers.Monitor(env, '../videos/' + vid_folder, force=True)
            env_base = env.env.env
        else:
            env_base = env.env
        expert_dict = {'state': [], 'action': [], 'goal': []}

        state = env.reset()
        if self.state_type == 'decayed_context':
            state = np.concatenate((state, np.array([1.0]),
                                    activity_map(self.mode_list[0]),
                                    activity_map(self.mode_list[min(1, N-1)])), axis=0)
        elif self.state_type == 'context':
            state = np.concatenate((state, activity_map(self.mode_list[0]),
                                    activity_map(self.mode_list[min(1, N-1)])), axis=0)

        if self.running_state is not None:
            state = self.running_state(state, update=False)

        save_flag = True # is true when the episode does not end early (agent doesn't die)
    
        for i in range(len(self.mode_list)):
            if save_flag == False:
                break
            self.policy = policy_list[i]
            mode = self.mode_list[i]
            if use_gpu:
                if self.use_phase:
                    self.policy.convert_to_cuda()
                else:
                    self.policy.cuda()

            if hasattr(env_base, 'mode'):
                env_base.mode = mode

            for n in range(num_steps_per_policy):
                print(n)
                phase = (i * num_steps_per_policy +  n) / (N * num_steps_per_policy - 1)
                state_var = Variable(self.tensor(state).unsqueeze(0), volatile=True)
                phase_var = Variable(self.tensor(np.array([phase])), volatile=True)
                state_var = state_var.cuda()

                if use_phase:
                    action  = self.policy(state_var, phase_var)[0]
                else:
                    action  = self.policy(state_var)[0]

                if use_gpu:
                    action = action.data[0].cpu().numpy()
                else:
                    action = action.data[0].numpy()
                next_state, reward, done, _ = env.step(action)

                if self.state_type == 'decayed_context':
                    next_state = np.concatenate((next_state,
                                                 np.array([1/(n+1)]),
                                                 activity_map(self.mode_list[min(n+1, N * num_steps_per_policy-1) // num_steps_per_policy]),
                                                 activity_map(self.mode_list[min(i+1, N-1)])), axis=0)
                elif self.state_type == 'context':
                    next_state = np.concatenate((next_state,
                                                 activity_map(self.mode_list[min(n+1, N * num_steps_per_policy-1) // num_steps_per_policy]),
                                                 activity_map(self.mode_list[min(i+1, N-1)])), axis=0)

                if self.running_state is not None:
                    next_state = self.running_state(next_state, update=False)
                if self.render:
                    env.render()
                if done:
                    if i != len(self.mode_list) - 1 or n != num_steps_per_policy:
                        save_flag = False
                    break

                expert_dict['state'].append(state)
                expert_dict['action'].append(action)
                expert_dict['goal'].append(0)

                state = next_state

        return expert_dict, save_flag
