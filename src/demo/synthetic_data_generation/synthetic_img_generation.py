# DDPM Object Forward Projection and Reconstruction Script
#
# This script performs forward projection and reconstruction of DDPM (Denoising Diffusion Probabilistic Models)
# generated objects using RSOS (Root Sum of Squares) method to create a few examples of accelerated MR images.
# It saves the reconstructions in HDF5 format as well as png format.
#
# Command-line Options:
#     acceleration (int): Acceleration factor for sparse sampling (2, 4, 6, or 8).
#
# Usage:
#     python synthetic_img_generation.py [acceleration factor]
#
# Examples:
#     Run with acceleration factor 2:
#       python synthetic_img_generation.py 2
#
# Note: Ensure that all required data files and directories are properly set up before running the script.
# To run, source the following environment
# source /home/prabhat.kc/anaconda3/base_env.sh
# source /home/prabhat.kc/anaconda3/horovod_sm80_env.sh

import torch
import numpy as np
import torch.nn as nn
import os
import sys
from dataset import DatasetFromNpz
import add_signals
import h5py
from PIL import Image

# ------------------------------------- CUDA for PyTorch ------------------------------------------#
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# ------------------------------ Some basic settings ----------------------------------------------#
acceleration = int(sys.argv[1])

num_gen_img = int(sys.argv[2])

is_png = int(sys.argv[3])

DDPM_obj_path = "/projects01/didsr-aiml/zitong.yu/DLMO_demo/synthetic_data_generation/examples/DDPM_obj/"
mr_acq_path = "/projects01/didsr-aiml/zitong.yu/DLMO_demo/"

dim1, dim2 = 260, 311
n_std = 15
n_coil = 8
cmpr_dtype = 'float32'
batch_size = num_gen_img

te_half_size = int(num_gen_img/2)
te_tot_size = int(2 * te_half_size)

if cmpr_dtype == 'float16':
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

output_path = "/projects01/didsr-aiml/zitong.yu/DLMO_demo/synthetic_data_generation/examples/img_w_signal/"
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
print("\nReading DDPM-generated objects ...", flush=True)

imgs = np.load(DDPM_obj_path + "samples_10000x260x311x1.npz")
imgs = imgs.f.arr_0

imgs = imgs.astype(cmpr_dtype)
imgs = imgs / 255.0

imgs = np.reshape(imgs, (-1, dim1, dim2))
init_test_som = imgs.shape[0]

imgs = imgs[:te_tot_size,:,:]

print('Shape of the loaded testing data is:', imgs.shape, 'and its dtype is:', imgs.dtype, flush=True)

print('Out of %d SOMs, %d of them are used.\n' % (init_test_som, imgs.shape[0]), flush=True)  #
print('min and max in the data: [%.4f, %.4f]' % (np.min(imgs), np.max(imgs)))

# ----------------------------------------------ADD SIGNALS--------------------------#
print("\nAdding signal to objects ... ", flush=True)

loc_list, L_list, imgs = add_signals.AddSignalRayleigh(imgs, A, wid, doublet_L, loc, te_half_size, te_tot_size, dim1, dim2)

imgs = np.reshape(imgs, (te_tot_size, 1, dim1, dim2))  # second index stores info on coil

print('Shape of loaded data:', imgs.shape, ', dtype of data:', \
imgs.dtype, ', no. of doublet SOM in the data set:', te_half_size, flush=True)

print('min and max in data with signal: [%.4f, %.4f]' % (np.min(imgs), np.max(imgs)))

print("\nSaving hdf5 files to: " + output_path + "gt_acc" + str(acceleration) + "_rsos.hdf5")
f = h5py.File(output_path + "gt_acc" + str(acceleration) + "_rsos.hdf5", "w")
f.create_dataset('H_1', data=imgs[:te_half_size,:,:,:], dtype=cmpr_dtype)
f.create_dataset('H_0', data=imgs[te_half_size:,:,:,:], dtype=cmpr_dtype)
f.create_dataset('L_list', data=L_list, dtype=cmpr_dtype)

if is_png == 1:
    print("\nSaving images to: " + output_path + "gt_sample_singlet[doublet]_rsos_{index}.png")
    for i in range(te_half_size):
        im = np.squeeze(imgs[i,:,:])
        im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255
        im = Image.fromarray(np.uint8(im))
        im.save(output_path + "gt_sample_doublet_rsos_" + str(i) + ".png")
        im = np.squeeze(imgs[te_half_size + i,:,:])
        im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255
        im = Image.fromarray(np.uint8(im))
        im.save(output_path + "gt_sample_singlet_rsos_" + str(i) + ".png")

# ------------------------------ Forward project test data -------------------------------------------------#

for acc in [acceleration]:
    cur_data = np.empty(imgs.shape)

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
        local_batch = imgs[batch_index * batch_size:(1 + batch_index) * batch_size,:,:,:]

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
        cur_data[batch_index * batch_size:(1 + batch_index) * batch_size,:,:,:] = local_batch_cat.cpu()

    # -----------------------------------Save to hdf5 files--------------------------#
    # test data

    print('min and max in reconstructed images with signal: [%.4f, %.4f]' % (np.min(cur_data), np.max(cur_data)))

    print("\nSaving hdf5 files to: " + output_path + "accelerated_acc" + str(acc) + "_rsos.hdf5")
    f = h5py.File(output_path + "accelerated_acc" + str(acc) + "_rsos.hdf5", "w")
    f.create_dataset('H_1', data=cur_data[:te_half_size,:,:,:], dtype=cmpr_dtype)
    f.create_dataset('H_0', data=cur_data[te_half_size:,:,:,:], dtype=cmpr_dtype)
    f.create_dataset('L_list', data=L_list, dtype=cmpr_dtype)
    f.close()

    if is_png == 1:
        print("\nSaving images to: " + output_path + "accelerated_sample_singlet[doublet]_rsos_{index}.png")
        for i in range(te_half_size):
            im = np.squeeze(cur_data[i,:,:])
            im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255
            im = Image.fromarray(np.uint8(im))
            im.save(output_path + "accelerated_sample_doublet_rsos_" + str(i) + ".png")
            im = np.squeeze(cur_data[te_half_size + i,:,:])
            im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255
            im = Image.fromarray(np.uint8(im))
            im.save(output_path + "accelerated_sample_singlet_rsos_" + str(i) + ".png")

# Acc x1
cur_data = np.empty(imgs.shape)
# -----------------------------------K-SPACE ACQUISITION AND RECONSTRUCTION--------------------------#

# testing data
for batch_index in range(int(te_tot_size / batch_size)):

    # shape of (320, 260, 311) -> 320 is batch size here
    local_batch = imgs[batch_index * batch_size:(1 + batch_index) * batch_size,:,:,:]

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
    cur_data[batch_index * batch_size:(1 + batch_index) * batch_size,:,:,:] = local_batch_cat.cpu()

# -----------------------------------Save to hdf5 files--------------------------#
# test data

print("\nSaving hdf5 files to: " + output_path + "fully_sampled_acc" + str(acc) + "_rsos.hdf5")
f = h5py.File(output_path + "fully_sampled_acc" + str(acc) + "_rsos.hdf5", "w")
f.create_dataset('H_1', data=cur_data[:te_half_size,:,:,:], dtype=cmpr_dtype)
f.create_dataset('H_0', data=cur_data[te_half_size:,:,:,:], dtype=cmpr_dtype)
f.create_dataset('L_list', data=L_list, dtype=cmpr_dtype)
f.close()

if is_png == 1:
    print("\nSaving images to: " + output_path + "fully_sampled_singlet[doublet]_rsos_{index}.png")
    for i in range(te_half_size):
        im = np.squeeze(cur_data[i,:,:])
        im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255
        im = Image.fromarray(np.uint8(im))
        im.save(output_path + "fully_sampled_doublet_rsos_" + str(i) + ".png")
        im = np.squeeze(cur_data[te_half_size + i,:,:])
        im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255
        im = Image.fromarray(np.uint8(im))
        im.save(output_path + "fully_sampled_singlet_rsos_" + str(i) + ".png")
