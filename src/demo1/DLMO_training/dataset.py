from torch.utils.data import Dataset
import numpy as np
import h5py
import torch
import os
import glob
import sys


class DatasetFromNpz(Dataset):
# A custom Dataset class for loading data from .npz or .npy files.
#
# This class inherits from torch.utils.data.Dataset and is designed to read
# image data stored in NumPy array format (.npz or .npy files).
#
# Attributes:
#     data (numpy.ndarray): The loaded image data.
#
# Methods:
#     __init__(file_path): Constructor that loads the data from the given file path.
#     __getitem__(index): Returns a single image from the dataset.
#     __len__(): Returns the total number of images in the dataset.

    def __init__(self, file_path):
        # Initialize the DatasetFromNpz object.
        #
        # Args:
        #     file_path (str): Path to the .npz or .npy file containing the image data.
        #
        # Output:
        #     None
        super(DatasetFromNpz, self).__init__()
        self.data = np.load(file_path)

    def __getitem__(self, index):
        input = self.data[index,:,:,:]
        return input

    def __len__(self):
        return self.data.shape[0]


class DatasetFromHdf5(Dataset):
# A custom Dataset class for loading data from HDF5 files.
#
# This class inherits from torch.utils.data.Dataset and is designed to read
# image data and corresponding labels stored in HDF5 format.
#
# Attributes:
#     data (numpy.ndarray): The loaded image data.
#     target (numpy.ndarray): The corresponding labels for the image data.
#
# Methods:
#     __init__(hvd, file_path, mod_num, drop_style): Constructor that loads and processes the data from the given HDF5 file.
#     __getitem__(index): Returns a single image and its label from the dataset.
#     __len__(): Returns the total number of images in the dataset.

    def __init__(self, hvd, file_path, mod_num=1, drop_style='remove_last'):
        # Initialize the DatasetFromHdf5 object.
        #
        # Args:
        #     hvd: Horovod object for distributed training (not used in this method but kept for printing loading message).
        #     file_path (str): Path to the HDF5 file containing the image data and labels.
        #     mod_num (int): Number used for ensuring the dataset size is divisible by this value.
        #     drop_style (str): Strategy for handling dataset size when not divisible by mod_num.
        #                       Options: 'remove_last' or 'add_n_remove_last'.
        #
        # Output:
        #     None
        #
        # Note:
        #     This method loads data from HDF5 file, combines signal-present and signal-absent samples,
        #     and adjusts the dataset size based on mod_num and drop_style.
        #     Targets are a numpy array with H0.shape[0] of "1"s, followed by H1.shape[0] of "0"s.
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
                self.target = np.concatenate((np.ones([H0.shape[0], 1], dtype=H0.dtype), np.zeros([H1.shape[0], 1], dtype=H1.dtype)), axis=0)
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
        trgt = self.target[index]
        return(torch.from_numpy(input).float(), torch.from_numpy(trgt).float())