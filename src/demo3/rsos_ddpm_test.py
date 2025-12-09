# DDPM Object Forward Projection and Reconstruction Script
#
# This script performs forward projection and reconstruction of DDPM (Denoising Diffusion Probabilistic Models)
# generated objects using RSOS (Root Sum of Squares) method to create test dataset.
# It saves the reconstructions in HDF5 format.
#
# Command-line Options:
#     acceleration (int): Acceleration factor for sparse sampling (2, 4, 6, or 8).
#
# Usage:
#     python rsos_ddpm_test.py [acceleration factor]
#
# Examples:
#     Run with acceleration factor 2:
#       python rsos_ddpm_test.py 2
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

output_path = "./rsos_rec/"
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

# # a few samples
# print("\nSampling some images to: " + output_path + "test_gt_sample0[1]_rsos.npy")
# np.save(output_path + "test_gt_sample1_rsos.npy", testing_data[:10,:,:,:])
# np.save(output_path + "test_gt_sample0_rsos.npy", testing_data[te_half_size:te_half_size + 10,:,:,:])
# np.save(output_path + "test_gt_L_list_0_rsos.npy", L_list[:10])
# np.save(output_path + "test_gt_L_list_1_rsos.npy", L_list[te_half_size:te_half_size + 10])

# ------------------------------ Forward project test data -------------------------------------------------#

for acc in [acceleration]:
    cur_testing_data = np.empty(testing_data.shape)

    # ------------------------------ sparse sampling info -------------------------------------------------#
    if acc > 1:
        print('Acceleration factor: ' + str(acc) + 'x')
        # Load mask
        mask_dir = "masks/mask_Poisson_" + str(acc) + "_fold.npy"
        mask = np.load(mr_acq_path + mask_dir)  # shape of (260, 311)
        mask = np.reshape(mask, (1, 1, dim1, dim2))
        mask = torch.tensor(mask, dtype=torch.complex64)
        mask = mask.to(device)
        print("Mask shape: ", mask.shape, flush=True)  # shape of [1, 1, 260, 311]

    # -----------------------------------K-SPACE ACQUISITION AND RECONSTRUCTION--------------------------#

    # testing data
    for batch_index in range(int(te_tot_size / batch_size)):

        # shape of (320, 260, 311) -> 320 is batch size here
        local_batch = testing_data[batch_index * batch_size:(1 + batch_index) * batch_size,:,:,:]

        local_batch = np.repeat(local_batch, n_coil, axis=1)  # shaped (320, 8, 260, 311)

        # Transfer to gpu device and into k-space measurement with noise.
        local_batch = torch.tensor(local_batch, dtype=torch.float32).to(device)
        local_batch = local_batch * sensi_map
        local_batch = torch.fft.fftshift(input=torch.fft.fft2(local_batch), dim=(2, 3))
        local_batch = torch.normal(local_batch, n_std)

        # local_batch_k = torch.tensor(testing_data_k[batch_index * batch_size:(1 + batch_index) * batch_size,:,:,:]).to(device)
        if acc > 1:
            local_batch_k = local_batch * mask  # local batch has shape of ([320, 8, 260, 311]) and is complex array
        else:
            local_batch_k = local_batch
        local_batch_recs = torch.fft.ifft2(local_batch_k)

        local_batch_cat = torch.square(torch.abs(local_batch_recs))
        local_batch_cat = torch.sqrt(torch.sum(input=local_batch_cat, dim=1))
        local_batch_cat = torch.reshape(local_batch_cat, (batch_size, 1, dim1, dim2))
        cur_testing_data[batch_index * batch_size:(1 + batch_index) * batch_size,:,:,:] = local_batch_cat.cpu()

    # -----------------------------------Save to hdf5 files--------------------------#
    # test data

    print('min and max in reconstructed testing data with signal: [%.4f, %.4f]' % (np.min(cur_testing_data), np.max(cur_testing_data)))

    print("\nSaving hdf5 files to: " + output_path + "test_acc" + str(acc) + "_rsos.hdf5")
    f = h5py.File(output_path + "test_acc" + str(acc) + "_rsos.hdf5", "w")
    f.create_dataset('H_1', data=cur_testing_data[:te_half_size,:,:,:], dtype=cmpr_dtype)
    f.create_dataset('H_0', data=cur_testing_data[te_half_size:,:,:,:], dtype=cmpr_dtype)
    f.create_dataset('L_list', data=L_list, dtype=cmpr_dtype)
    f.close()

    # # a few samples
    # print("\nSampling some images to: " + output_path + "test_acc" + str(acc) + "_sample0[1]_rsos.npy")
    # np.save(output_path + "test_acc" + str(acc) + "_sample1_rsos.npy", cur_testing_data[:10,:,:,:])
    # np.save(output_path + "test_acc" + str(acc) + "_sample0_rsos.npy", cur_testing_data[te_half_size:te_half_size + 10,:,:,:])
    # np.save(output_path + "test_acc" + str(acc) + "_L_list_0_rsos.npy", L_list[:10])
    # np.save(output_path + "test_acc" + str(acc) + "_L_list_1_rsos.npy", L_list[te_half_size:te_half_size + 10])



# Acc x1
cur_testing_data = np.empty(testing_data.shape)
# -----------------------------------K-SPACE ACQUISITION AND RECONSTRUCTION--------------------------#

# testing data
for batch_index in range(int(te_tot_size / batch_size)):

    # shape of (320, 260, 311) -> 320 is batch size here
    local_batch = testing_data[batch_index * batch_size:(1 + batch_index) * batch_size,:,:,:]

    local_batch = np.repeat(local_batch, n_coil, axis=1)  # shaped (320, 8, 260, 311)

    # Transfer to gpu device and into k-space measurement with noise.
    local_batch = torch.tensor(local_batch, dtype=torch.float32).to(device)
    local_batch = local_batch * sensi_map
    local_batch = torch.fft.fftshift(input=torch.fft.fft2(local_batch), dim=(2, 3))
    local_batch = torch.normal(local_batch, n_std)

    # local_batch_k = torch.tensor(testing_data_k[batch_index * batch_size:(1 + batch_index) * batch_size,:,:,:]).to(device)
    local_batch_k = local_batch
    local_batch_recs = torch.fft.ifft2(local_batch_k)

    local_batch_cat = torch.square(torch.abs(local_batch_recs))
    local_batch_cat = torch.sqrt(torch.sum(input=local_batch_cat, dim=1))
    local_batch_cat = torch.reshape(local_batch_cat, (batch_size, 1, dim1, dim2))
    cur_testing_data[batch_index * batch_size:(1 + batch_index) * batch_size,:,:,:] = local_batch_cat.cpu()

# -----------------------------------Save to hdf5 files--------------------------#
# test data

print("\nSaving hdf5 files to: " + output_path + "test_acc" + str(acc) + "_at_acc1_rsos.hdf5")
f = h5py.File(output_path + "test_acc" + str(acc) + "_at_acc1_rsos.hdf5", "w")
f.create_dataset('H_1', data=cur_testing_data[:te_half_size,:,:,:], dtype=cmpr_dtype)
f.create_dataset('H_0', data=cur_testing_data[te_half_size:,:,:,:], dtype=cmpr_dtype)
f.create_dataset('L_list', data=L_list, dtype=cmpr_dtype)
f.close()