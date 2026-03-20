# Doublet and Singlet Signals Insertion Script
#
# This script inserts doublet and singlet signals into DDPM (Denoising Diffusion Probabilistic Models) generated objects.
# It saves the objects with signals in HDF5 format.
#
# Command-line Options:
#     acceleration (int):      Acceleration factor for sparse sampling (2, 4, 6, or 8).
#     contrast (float):        Signal amplitude/contrast value.
#     signal_lengths (str):    Comma-separated signal separation lengths, e.g. "4,5,6,7,8".
#     object_npz_path (str, optional): Path to the DDPM-generated objects from demo 1.
#
# Usage:
#     python signal_insertion_test.py [acceleration factor] [contrast] [signal_lengths] [object_npz_path]
#
# Examples:
#     Run with acceleration factor 4:
#       python signal_insertion_test.py 4 0.7 '4,5,6,7,8'
#
# Note: Ensure that all required data files and directories are properly set up before running the script.
# To run, source the following environment
# conda activate dlmo


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
use_cuda                       = torch.cuda.is_available()
device                         = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
demo                           = True
# ------------------------------ Some basic settings ----------------------------------------------#
if len(sys.argv) < 4:
    print("Usage: python signal_insertion_test.py [acceleration] [contrast] [signal_lengths] [object_npz_path]")
    sys.exit(1)

default_test_data_file = "../demo5/image_acquisition_and_reconstruction/examples/DDPM_obj/samples_10000x260x311x1.npz"
mr_acq_path    = "../"

dim1, dim2     = 260, 311
n_std          = 15  # # is the always 15 for all acceleration factors ? KL: Yes
n_coil         = 8
cmpr_dtype     = 'float32'
batch_size     = 320

te_half_size   = 40 if demo else 4000
te_tot_size    = 2 * te_half_size

if cmpr_dtype == 'float16':
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

output_path = "./objects/"
if not os.path.isdir(output_path): os.makedirs(output_path, exist_ok=True)

# ------------------------------ Signals info -------------------------------------------------#
acceleration = int(sys.argv[1])                         # acceleration factor
A            = float(sys.argv[2])                       # contrast value/amplitude
signal_L_str = sys.argv[3]                              # signal lengths for singlet or doublet insertion
signal_L     = list(map(int, signal_L_str.split(",")))  #  comma-separated signal lengths
test_data_file = sys.argv[4] if len(sys.argv) > 4 else default_test_data_file

# print(acceleration, A, signal_L, test_data_file)

loc = np.load(mr_acq_path + "mri_loc.npy")
wid = 1.75
"""
'''commenting this part as demo cmd input 
will include acceleration factor, contrast values and signal lengths. 
'''
A_dict = {'1':0.3, \
          '2':0.3, \
          '4':0.7, \
          '6':1.0, \
          '8':1.3}
A = A_dict[str(acceleration)]
doublet_L = [4, 5, 6, 7, 8, 9]

# ----------------------------Load sensitivity map-------------------------------------------------#
''' Loading MR sensitivity coils and applying it during the  
forward modeling is not need for this  signal insertion demo. 
Hence commenting this part
'''
map_dir = "sensitivity_8coils.npy"
sensi_map = np.load(mr_acq_path + map_dir)  # shaped (8, 260, 311)
sensi_map = np.reshape(sensi_map, (1, -1, dim1, dim2))  # shaped (1, 8, 260, 311)
sensi_map = torch.tensor(sensi_map, dtype=torch_dtype)
sensi_map = sensi_map.to(device)
print('shape of the loaded sensitivity map: ', sensi_map.shape, 'and its dtype is', sensi_map.dtype, flush=True)
"""
# ----------------------------------------------Loading the testing data--------------------------#
print("\nReading the test dataset ...", flush=True)

# testing_data = utils.list_all_npy_files(test_data_path, cmpr_dtype=cmpr_dtype, unity_normalize=True)

testing_data  = np.load(test_data_file)
testing_data  = testing_data.f.arr_0
testing_data  = np.squeeze(testing_data)

testing_data  = testing_data.astype(cmpr_dtype)
testing_data  = testing_data/255.0

testing_data  = np.reshape(testing_data, (-1, dim1, dim2))
init_test_som = testing_data.shape[0]

testing_data  = testing_data[:te_tot_size,:,:]

print('Shape of the loaded testing data is:', testing_data.shape, 'and its dtype is:', testing_data.dtype, flush=True)

testing_data = testing_data[:te_tot_size,:,:]
print('Out of %d SOMs, %d of them are used to test DLMO\n' % (init_test_som, testing_data.shape[0]), flush=True)  #
print('min and max in testing data: [%.4f, %.4f]' % (np.min(testing_data), np.max(testing_data)))


# ----------------------------------------------ADD SIGNALS--------------------------#
# testing data
print("\nAdding signal to testing objects ... ", flush=True)

loc_list, L_list, testing_data = add_signals.AddSignalRayleigh(testing_data, A, wid, signal_L, loc, te_half_size, te_tot_size, dim1, dim2)

testing_data = np.reshape(testing_data, (te_tot_size, 1, dim1, dim2))  # second index is added to store information on coil sensitivity
print('shape of saved test data:', testing_data.shape, ', dtype of the saved testing data:', \
testing_data.dtype, ', no. of doublet SOM in the test set:', te_half_size, flush=True)

print('min and max in testing data with signal: [%.4f, %.4f]' % (np.min(testing_data), np.max(testing_data)))


if demo: 
	# display demo plot ----------------------------------------------------------------------------------------------------------------------------
	num_L_list = [int(x[0]) for x in L_list]
	L_str_list = [str(x) for x in num_L_list]
	demo_h1_few = np.transpose(np.squeeze(testing_data[0:3]), axes=(0, 2, 1))
	demo_h0_few = np.transpose(np.squeeze(testing_data[(te_half_size+3):(te_half_size+6)]), axes=(0, 2, 1))
	
	print('signal seperation length list:', L_str_list)
	utils.multi2dplots(1, 3, demo_h1_few[:, ::-1, :], axis=0, passed_fig_att={'colorbar': False, 'suptitle':'L_list', 'split_title': L_str_list[0:3]})
	utils. multi2dplots(1, 3, demo_h0_few[:, ::-1, :], axis=0, passed_fig_att={'colorbar': False, 'suptitle':'L_list', 'split_title': L_str_list[te_half_size+3:(te_half_size+6)]})
	# save demo objects into a .npz file ----------------------------------------------------------------------------------------------------------------------------
	print("\nSampling some demo images to: " + output_path + "test_gt_sample0[1]_rsos.npy")
	np.save(output_path + "test_gt_sample1_rsos.npy", testing_data[:te_half_size,:,:,:])
	np.save(output_path + "test_gt_sample0_rsos.npy", testing_data[te_half_size:,:,:,:])
	np.save(output_path + "test_gt_L_list_0_rsos.npy", L_list[:te_half_size])
	np.save(output_path + "test_gt_L_list_1_rsos.npy", L_list[te_half_size: ])
else: 
	print("\nSaving hdf5 files to: " + output_path + "test_gt_acc" + str(acceleration) + "_rsos.hdf5")
	f = h5py.File(output_path + "test_gt_acc" + str(acceleration) + "_rsos.hdf5", "w")
	f.create_dataset('H_1', data=testing_data[:te_half_size,:,:,:], dtype=cmpr_dtype)
	f.create_dataset('H_0', data=testing_data[te_half_size:,:,:,:], dtype=cmpr_dtype)
	f.create_dataset('L_list', data=L_list, dtype=cmpr_dtype)