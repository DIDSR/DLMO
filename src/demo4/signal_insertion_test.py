# Doulet and Singlet Signals Insersion Script
#
# This script inserts doulet and singlet signals into DDPM (Denoising Diffusion Probabilistic Models) generated objects.
# It saves the objects with signals in HDF5 format.
#
# Command-line Options:
#     acceleration (int): Acceleration factor for sparse sampling (2, 4, 6, or 8).
#
# Usage:
#     python signal_insertion_test.py [acceleration factor]
#
# Examples:
#     Run with acceleration factor 2:
#       python signal_insertion_test.py 2
#
# Note: Ensure that all required data files and directories are properly set up before running the script.
# To run, source the following environment
# source /home/prabhat.kc/anaconda3/base_env.sh
# source /home/prabhat.kc/anaconda3/horovod_sm80_env.sh


import torch
import numpy as np
import torch.nn as nn
import os
import utils
import sys
from dataset import DatasetFromNpz
import add_signals
import h5py

# ------------------------------------- CUDA for PyTorch ------------------------------------------#
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# ------------------------------ Some basic settings ----------------------------------------------#
acceleration = int(sys.argv[1])

test_data_path = "../demo1/synthetic_data_generation/examples/DDPM_obj/"
mr_acq_path = "../"

dim1, dim2 = 260, 311
n_std = 15  # # is the always 15 for all acceleration factors ? KL: Yes
n_coil = 8
cmpr_dtype = 'float32'
batch_size = 320

te_half_size = 4000
te_tot_size = 2 * te_half_size

if cmpr_dtype == 'float16':
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

output_path = "./objects/"
if not os.path.isdir(output_path): os.makedirs(output_path, exist_ok=True)

# ------------------------------ Signals info -------------------------------------------------#
loc = np.load(mr_acq_path + "mri_loc.npy")

A_dict = {'1':0.3, \
          '2':0.3, \
          '4':0.7, \
          '6':1.0, \
          '8':1.3}
A = A_dict[str(acceleration)]

wid = 1.75

doublet_L = [4, 5, 6, 7, 8, 9]

# ----------------------------Load sensitivity map-------------------------------------------------#
map_dir = "sensitivity_8coils.npy"
sensi_map = np.load(mr_acq_path + map_dir)  # shaped (8, 260, 311)
sensi_map = np.reshape(sensi_map, (1, -1, dim1, dim2))  # shaped (1, 8, 260, 311)
sensi_map = torch.tensor(sensi_map, dtype=torch_dtype)
sensi_map = sensi_map.to(device)
print('shape of the loaded sensitivity map: ', sensi_map.shape, 'and its dtype is', sensi_map.dtype, flush=True)

# ----------------------------------------------Loading the testing data--------------------------#
print("\nReading the test dataset ...", flush=True)

# testing_data = utils.list_all_npy_files(test_data_path, cmpr_dtype=cmpr_dtype, unity_normalize=True)

testing_data = np.load(test_data_path+'/samples_10000x260x311x1.npz')
testing_data  = testing_data.f.arr_0
testing_data  = np.squeeze(testing_data)

testing_data = testing_data.astype(cmpr_dtype)
testing_data = testing_data/255.0

testing_data = np.reshape(testing_data, (-1, dim1, dim2))
init_test_som = testing_data.shape[0]

testing_data = testing_data[:te_tot_size,:,:]

print('Shape of the loaded testing data is:', testing_data.shape, 'and its dtype is:', testing_data.dtype, flush=True)

testing_data = testing_data[:te_tot_size,:,:]
print('Out of %d SOMs, %d of them are used to test DLMO\n' % (init_test_som, testing_data.shape[0]), flush=True)  #
print('min and max in testing data: [%.4f, %.4f]' % (np.min(testing_data), np.max(testing_data)))

# ----------------------------------------------ADD SIGNALS--------------------------#
# testing data
print("\nAdding signal to testing objects ... ", flush=True)

loc_list, L_list, testing_data = add_signals.AddSignalRayleigh(testing_data, A, wid, doublet_L, loc, te_half_size, te_tot_size, dim1, dim2)

testing_data = np.reshape(testing_data, (te_tot_size, 1, dim1, dim2))  # second index stores info on coil

print('shape of loaded test data:', testing_data.shape, ', dtype of testing data:', \
testing_data.dtype, ', no. of doublet SOM in the test set:', te_half_size, flush=True)

print('min and max in testing data with signal: [%.4f, %.4f]' % (np.min(testing_data), np.max(testing_data)))

print("\nSaving hdf5 files to: " + output_path + "test_gt_acc" + str(acceleration) + "_rsos.hdf5")
f = h5py.File(output_path + "test_gt_acc" + str(acceleration) + "_rsos.hdf5", "w")
f.create_dataset('H_1', data=testing_data[:te_half_size,:,:,:], dtype=cmpr_dtype)
f.create_dataset('H_0', data=testing_data[te_half_size:,:,:,:], dtype=cmpr_dtype)
f.create_dataset('L_list', data=L_list, dtype=cmpr_dtype)

# a few samples
print("\nSampling some images to: " + output_path + "test_gt_sample0[1]_rsos.npy")
np.save(output_path + "test_gt_sample1_rsos.npy", testing_data[:10,:,:,:])
np.save(output_path + "test_gt_sample0_rsos.npy", testing_data[te_half_size:te_half_size + 10,:,:,:])
np.save(output_path + "test_gt_L_list_0_rsos.npy", L_list[:10])
np.save(output_path + "test_gt_L_list_1_rsos.npy", L_list[te_half_size:te_half_size + 10])