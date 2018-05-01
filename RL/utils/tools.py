import os
import h5py
import numpy as np


def assets_dir():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets'))

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
