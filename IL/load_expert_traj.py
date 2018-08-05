import numpy as np
from collections import namedtuple
from utils.running_state import ZFilter
import h5py
import os
import pdb
import random

Trajectory = namedtuple('Trajectory', ('state', 'action', 'c', 'mask'))

def recursively_get_dict_from_group(group_or_data):
    d = {}
    if type(group_or_data) == h5py.Dataset:
        return np.array(group_or_data)

    # Else it's still a group
    for k in group_or_data.keys():
        v = recursively_get_dict_from_group(group_or_data[k])
        d[k] = v
    return d

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Take an already open HDF5 file and insert the contents of a dictionary
    at the current path location. Can call itself recursively to fill
    out HDF5 files with the contents of a dictionary.
    """
    assert type(dic) is type({}), "must provide a dictionary"
    assert type(path) is type(''), "path must be a string"
    assert type(h5file) is h5py._hl.files.File, "must be an open h5py file"

    for key in dic:
        assert type(key) is type(''), 'dict keys must be strings to save to hdf5'
        did_save_key = False

        if type(dic[key]) in (np.int64, np.float64, type(''), int, float):
            h5file[path + key] = dic[key]
            did_save_key = True
            assert h5file[path + key].value == dic[key], \
                'The data representation in the HDF5 file does not match the ' \
                'original dict.'
        if type(dic[key]) is type([]):
            h5file[path + key] = np.array(dic[key])
            did_save_key = True
        if type(dic[key]) is np.ndarray:
            h5file[path + key] = dic[key]
            did_save_key = True
            assert np.array_equal(h5file[path + key].value, dic[key]), \
                'The data representation in the HDF5 file does not match the ' \
                'original dict.'
        if type(dic[key]) is type({}):
            recursively_save_dict_contents_to_group(h5file,
                                                    path + key + '/',
                                                    dic[key])
            did_save_key = True
        if not did_save_key:
            print("Dropping key from h5 file: {}".format(path + key))

class Expert(object):
    def __init__(self, folder, num_inputs):
        self.memory = []
        self.pointer = 0
        self.n = len(os.listdir(folder))
        self.folder = folder
        self.data_files = os.listdir(folder)
        self.list_of_sample_c = []
        #self.running_state = ZFilter((num_inputs,), clip=5)

    def push(self):
        """Saves a (state, action, c, mask) tuple."""
        for filename in self.data_files:
            #f = open(self.folder + str(i) + '.txt', 'r')
            f = open(os.path.join(self.folder, filename), 'r')
            line_counter = 0
            temp_mem = []
            temp_c = []
            for line in f:
                if line_counter % 3 == 0:
                    if line_counter > 0:
                        temp_mem.append(Trajectory(s, a, c, 1))
                    #s = self.running_state(np.asarray(line.strip().split(), dtype='float'))
                    s = np.asarray(line.strip().split(), dtype=np.float32)
                elif line_counter % 3 == 1:
                    a = np.asarray(line.strip().split(), dtype=np.float32)
                elif line_counter % 3 == 2:
                    c = np.asarray(line.strip().split(), dtype=np.float32)
                    temp_c.append(c)

                line_counter += 1

            f.close()
            temp_mem.append(Trajectory(s, a, c, 0))
            self.memory.append(Trajectory(*zip(*temp_mem)))
            self.list_of_sample_c.append(np.array(temp_c))


    def sample(self, size=5):
        ind = np.random.randint(self.n, size=size)
        batch_list = []
        for i in ind:
            batch_list.append(self.memory[i])

        return Trajectory(*zip(*batch_list))

    def sample_as_list(self, size=5):
        ind = np.random.randint(self.n, size=size)
        batch_list = []
        for i in ind:
            batch_list.append(self.memory[i])

        return batch_list

    def sample_c(self):
        ind = random.randint(0, self.n-1)
        return self.list_of_sample_c[ind]

    #def sample_batch(self, batch_size):
    #    random_batch = random.sample(self.memory, batch_size)
    #    return Transition(*zip(*random_batch))

    def __len__(self):
        return len(self.memory)

class ExpertHDF5(Expert):
    def __init__(self, expert_dir, num_inputs):
        super(ExpertHDF5, self).__init__(expert_dir, num_inputs)
        self.expert_dir = expert_dir
        self.memory, self.memory_no_tuple = [], []
        self.pointer = 0
        self.list_of_sample_c = []

        self.obstacles = None
        self.set_diff = None
        self.num_goals, self.num_actions = 0, 0

    def push(self, only_coordinates_in_state=False, one_hot_action=True):
        h5_file = os.path.join(self.expert_dir, 'expert_traj.h5')
        assert os.path.exists(h5_file), \
                "hdf5 file does not exist {}".format(h5_file)
        h5f = h5py.File(h5_file, 'r')
        memory, memory_no_tuple = [], []

        self.num_goals = int(h5f['env_data']['num_goals'].value)
        self.num_actions = int(h5f['env_data']['num_actions'].value)

        for k in sorted(h5f['expert_traj'].keys()):
            state = np.array(h5f['expert_traj'][k]['state'], dtype=np.float32)
            # Just keep the (x, y) coordinates in state i.e. remove the features
            if only_coordinates_in_state:
                state = state[:, :2]
            action = np.array(h5f['expert_traj'][k]['action'], dtype=np.float32)
            if one_hot_action and len(action.shape) == 1:
                action_one_hot = np.zeros((action.shape[0], self.num_actions),
                                           dtype=np.float32)
                for a_idx in range(action.shape[0]):
                    action_one_hot[a_idx, int(action[a_idx])] = 1
                action = action_one_hot

            # context = h5f[k]['context']
            context = np.array(h5f['expert_traj'][k]['goal'], dtype=np.float32)
            mask = np.ones((action.shape[0]))
            mask[-1] = 0
            memory.append(Trajectory(state, action, context, mask))
            memory_no_tuple.append((state, action, context, mask))

        self.memory = memory
        self.memory_no_tuple = memory_no_tuple

        self.obstacles = np.array(h5f['obstacles'])
        self.set_diff = np.array(h5f['set_diff'])
        h5f.close()

    def sample(self, size=5, return_indices=False):
        ind = np.random.randint(len(self.memory), size=size)
        batch_list = []
        for i in ind:
            batch_list.append(self.memory[i])

        if return_indices:
            return Trajectory(*zip(*batch_list)), ind
        else:
            return Trajectory(*zip(*batch_list))

    def sample_as_list(self, size=5):
        ind = np.random.randint(len(self.memory), size=size)
        batch_list = []
        for i in ind:
            batch_list.append(self.memory[i])

        return batch_list

    def sample_c(self):
        ind = random.randint(0, len(self.memory)-1)
        sample_c = [data[i][2] for i in ind]
        return sample_c

class SeparateRoomTrajExpert(ExpertHDF5):
    def __init__(self, expert_dir, num_inputs):
        super(SeparateRoomTrajExpert, self).__init__(expert_dir, num_inputs)

    def push(self, only_coordinates_in_state=False, one_hot_action=True):
        h5_file = os.path.join(self.expert_dir, 'expert_traj.h5')
        assert os.path.exists(h5_file), \
                "hdf5 file does not exist {}".format(h5_file)
        h5f = h5py.File(h5_file, 'r')
        memory = []

        self.num_goals = int(h5f['env_data']['num_goals'].value)
        self.num_actions = int(h5f['env_data']['num_actions'].value)

        for k in sorted(h5f['expert_traj'].keys()):
            state = np.array(h5f['expert_traj'][k]['state'], dtype=np.float32)
            # Just keep the (x, y) coordinates in state i.e. remove the features
            if only_coordinates_in_state:
                state = state[:, :2]
            action = np.array(h5f['expert_traj'][k]['action'], dtype=np.float32)
            if one_hot_action and len(action.shape) == 1:
                action_one_hot = np.zeros((action.shape[0], self.num_actions),
                                           dtype=np.float32)
                for a_idx in range(action.shape[0]):
                    action_one_hot[a_idx, int(action[a_idx])] = 1
                action = action_one_hot

            # context = h5f[k]['context']
            context = np.array(h5f['expert_traj'][k]['goal'], dtype=np.float32)

            # Get the list of episodes for the meta-episode
            episode_list = self.get_list_of_episodes(state, action, context)
            for episode in episode_list:
                memory.append(Trajectory(episode[0],
                                         episode[1],
                                         episode[2],
                                         episode[3]))

        self.memory = memory

        self.obstacles = np.array(h5f['obstacles'])
        self.set_diff = np.array(h5f['set_diff'])
        h5f.close()

    def get_list_of_episodes(self, state_arr, action_arr, context_arr):
        '''
        Break the episode array which consists of multiple episodes into list
        of episodes.
        '''
        memory = []
        context_arr_int = context_arr.astype(np.int32)
        episode_len, last_state_idx = state_arr.shape[0], 0
        while last_state_idx < episode_len:
            curr_state_idx = last_state_idx + 1
            while curr_state_idx < episode_len:
                if (context_arr_int[last_state_idx].tolist() ==
                        context_arr_int[curr_state_idx].tolist()):
                    curr_state_idx += 1
                else:
                    break
            # Do not put very small episodes into memory.
            if curr_state_idx - last_state_idx > 1:
                episode_range = range(last_state_idx, curr_state_idx)
                mask_arr = np.ones((len(episode_range)))
                mask_arr[-1] = 0
                memory.append((state_arr[episode_range],
                               action_arr[episode_range],
                               context_arr[episode_range],
                               mask_arr))
            last_state_idx = curr_state_idx

        return memory
