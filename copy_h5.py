import h5py
import sys
import os
import numpy as np

def save_expert_traj_dict_to_h5(traj_data_dict, save_dir,
                                h5_filename='expert_traj.h5'):
    h5_f = h5py.File(os.path.join(save_dir, h5_filename), 'w')
    recursively_save_dict_contents_to_group(h5_f, '/', traj_data_dict)
    h5_f.flush()
    h5_f.close()
    print("Did save data to {}".format(os.path.join(save_dir, h5_filename)))

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


n_iter = 25
n_goals = 1

f = h5py.File(sys.argv[1], 'r')

expert_dict = {}
#expert_dict['env_data'] = {'num_goals': int(f['env_data']['num_goals'].value), 'num_actions': int(f['env_data']['num_actions'].value)}
expert_dict['env_data'] = {'num_goals': 1, 'num_actions': 1}
expert_dict['expert_traj'] = {}
expert_dict['obstacles'] = []
expert_dict['set_diff'] = []

for i in range(n_goals):
    for j in range(n_iter):
        iter_dict = {'state': [], 'action': [], 'goal': []}
        path_key = str(j) + '_' + str(i)
        #s = list(np.array(f['expert_traj'][path_key]['state']))
        #a = list(np.array(f['expert_traj'][path_key]['action']))
        #g = list(np.array(f['expert_traj'][path_key]['goal']))
        s = list(np.array(f[path_key]['state']))
        a = list(np.array(f[path_key]['action']))
        g = list(np.array(f[path_key]['goal']))

        iter_dict['state'] = s
        iter_dict['action'] = a
        iter_dict['goal'] = g

        expert_dict['expert_traj'][path_key] = iter_dict


save_expert_traj_dict_to_h5(expert_dict, './')
