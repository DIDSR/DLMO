from torch.utils.data import Dataset
import numpy as np
import h5py
import torch
import os
import glob
import sys

def list_all_npy_files(super_path, cmpr_dtype=None, unity_normalize=False):
# Recursively reads all .npz/.npy files in a directory structure and combines them into a single numpy array.
#
# Args:
# super_path (str): The root directory to start searching for .npz files.
# cmpr_dtype (numpy dtype, optional): If provided, converts the arrays to this data type.
# unity_normalize (bool): If True, normalizes uint8 data to the range [0, 1].
#
# Returns:
# numpy.ndarray: A combined array of all .npz files found.

    arr_accumulator = []
    idx = 0
    for main_folder in os.listdir(super_path): #outermost folder
        for sub_folder in os.listdir(os.path.join(super_path, main_folder)): # sub folders
            inner_path= os.path.join(super_path, main_folder, sub_folder) # each inner folder
            for file in os.listdir(inner_path):
                if file.endswith('.npz'):
                    each_npz_file = os.path.join(inner_path, file)
                    each_npz_read = np.load(each_npz_file)
                    each_npz_arr  = each_npz_read.f.arr_0
                    each_npz_arr  = np.squeeze(each_npz_arr)
                    # change the datatype and normalize the 'uint8'
                    if cmpr_dtype!=None: each_npz_arr = each_npz_arr.astype(cmpr_dtype)
                    if unity_normalize:  each_npz_arr = each_npz_arr/255.0
                    if idx ==0:
                        arr_accumulator = each_npz_arr
                    else:
                        arr_accumulator = np.append (arr_accumulator, each_npz_arr, axis=0)
                    idx = idx +1
    return(arr_accumulator)


class DatasetFromHdf5(Dataset):
# A custom Dataset class for loading data from HDF5 files.
#
# This class inherits from torch.utils.data.Dataset and is designed to read
# image data stored in HDF5 format, with options for data size adjustment.

    def __init__(self, hvd, file_path, mod_num=1, drop_style='remove_last'):
    # Initialize the DatasetFromHdf5 object.
    #
    # Args:
    # hvd: Horovod object for distributed training.
    # file_path (str): Path to the HDF5 file containing the image data.
    # mod_num (int): Number used for ensuring the dataset size is divisible by this value.
    # drop_style (str): Strategy for handling dataset size when not divisible by mod_num.
    #                   Options: 'remove_last' or 'add_n_remove_last'.

        super(DatasetFromHdf5, self).__init__()
        shuffle_patches = False
        # shuffling patches at the Sampler Distribution is more efficient
        # for h5 files. so let this options be False for this subroutine
        if os.path.isfile(file_path):
            if (os.path.exists(file_path) == True):
                hf = h5py.File(file_path, mode='r')
                H0 = hf.get("H_0")
                H1 = hf.get("H_1")
                self.data = np.append(H0, H1, axis=0)
            else:
                if hvd.rank() == 0:
                    print('\n-------------------------------------------------------------')
                    print("ERROR! No training/tuning h5 files. Re-check input data paths.")
                    print('--------------------------------------------------------------')
                    sys.exit()
        else:
            if hvd.rank() == 0:
                    print('\n----------------------------------------------------------------------------')
                    print("ERROR! Issues related to training/tuning path. Re-check training-fname option.")
                    print('------------------------------------------------------------------------------')
                    sys.exit()
        if np.mod(self.data.shape[0], mod_num) != 0:
            if drop_style == 'remove_last':
                # this option removes last b patches where a (mod n) eq b
                remove_n = np.mod(self.data.shape[0], mod_num)
                self.data = self.data[:-remove_n,:,:,:]
                self.target = self.target[:-remove_n]
            else:
                # add_n_remove_last
                # this options first adds patches of len mod_num
                # these addition patches are randomly selected from
                # the range [0, data.shape[0]. Then from the subsequent
                # generated array last b patches are removed where a (mod n ) eq b
                add_length = np.mod(self.data.shape[0], mod_num)
                add_ind = np.random.randint(low=0, high=self.data.shape[0], size=add_length)
                add_ind = np.sort(add_ind)
                print(add_length)
                extra_input = self.data[add_ind,:,:,:]
                extra_target = self.target[add_ind]
                # gf.multi2dplots(3, 3, extra_target[:, 0,:,:], 0)
                # gf.multi2dplots(3, 3, extra_input[:, 0,:,:], 0)
                sys.exit()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        input = self.data[index,:,:,:]
        return torch.from_numpy(input).float()

