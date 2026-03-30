#!/usr/bin/env python
"""
Run Step 4 model-observer evaluation on three HDF5 files and plot PC vs signal length.

This script runs dlmo_test_hvd.py three times for:
1) rSOS @ 4x  
2) rSOS @ 1x (fully sampled)
3) U-Net @ 4x

Then computes percent-correct (AUC) by signal length L={4,5,6,7,8} and plots the result.
"""

import argparse
import os
import subprocess
import sys
import numpy as np
import h5py
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Step 4: Generate evaluation figure")
    parser.add_argument("--h5-rsos-4", default="../image_acquisition_and_reconstruction/examples/img_w_signal/accelerated_acc4_rsos.hdf5",
                        help="rSOS 4x HDF5 file")
    parser.add_argument("--h5-rsos-1", default="../image_acquisition_and_reconstruction/examples/img_w_signal/fully_sampled_acc4_rsos.hdf5",
                        help="rSOS 1x (fully sampled) HDF5 file")
    parser.add_argument("--h5-unet-4", default="../AI_rec/ai_rec/test_acc4_unet.hdf5",
                        help="U-Net 4x HDF5 file")
    parser.add_argument("--l-list-file", default="../image_acquisition_and_reconstruction/examples/img_w_signal/accelerated_acc4_rsos.hdf5",
                        help="HDF5 file containing L_list (typically rSOS 4x)")
    parser.add_argument("--batch-size", type=int, default=6, help="Batch size for model observer")
    parser.add_argument("--output-png", default="./dlmo_predictions/step4.png", help="Output PNG path")
    parser.add_argument("--output-csv", default="./dlmo_predictions/step4.csv", help="Output CSV path")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output_png), exist_ok=True)

    # Run observer 3 times
    print("=" * 60)
    print("Running model observer on rSOS @ 4x ...")
    print("=" * 60)
    subprocess.run([
        sys.executable, "dlmo_test_hvd.py",
        "--task", "rayleigh",
        "--hdf5-file", args.h5_rsos_4,
        "--test-path", os.path.dirname(args.h5_rsos_4),
        "--acceleration", "4",
        "--batch-size", str(args.batch_size),
        "--pretrained-model-path", "./trained_model/mri_cnn_io_acc_4_hvd/hvd_cpts",
        "--pretrained-model-epoch", "50",
        "--preds-tag", "rsos_acc4"
    ], check=True)

    print("\n" + "=" * 60)
    print("Running model observer on rSOS @ 1x (fully sampled) ...")
    print("=" * 60)
    subprocess.run([
        sys.executable, "dlmo_test_hvd.py",
        "--task", "rayleigh",
        "--hdf5-file", args.h5_rsos_1,
        "--test-path", os.path.dirname(args.h5_rsos_1),
        "--acceleration", "1",
        "--batch-size", str(args.batch_size),
        "--pretrained-model-path", "./trained_model/mri_cnn_io_acc_1_hvd/hvd_cpts",
        "--pretrained-model-epoch", "170",
        "--preds-tag", "rsos_acc1"
    ], check=True)

    print("\n" + "=" * 60)
    print("Running model observer on U-Net @ 4x ...")
    print("=" * 60)
    subprocess.run([
        sys.executable, "dlmo_test_hvd.py",
        "--task", "rayleigh",
        "--hdf5-file", args.h5_unet_4,
        "--test-path", os.path.dirname(args.h5_unet_4),
        "--acceleration", "4",
        "--batch-size", str(args.batch_size),
        "--pretrained-model-path", "./trained_model/mri_cnn_io_acc_4_unet_hvd/hvd_cpts",
        "--pretrained-model-epoch", "50",
        "--preds-tag", "unet_acc4"
    ], check=True)

    # Load predictions and L_list
    print("\n" + "=" * 60)
    print("Computing per-length PC values and generating plot ...")
    print("=" * 60)

    preds_rsos_4 = np.load("./dlmo_predictions/acc4/preds_rsos_acc4.npy")
    preds_rsos_1 = np.load("./dlmo_predictions/acc1/preds_rsos_acc1.npy")
    preds_unet_4 = np.load("./dlmo_predictions/acc4/preds_unet_acc4.npy")

    # Load L_list
    with h5py.File(args.l_list_file, "r") as hf:
        l_list = np.array(hf["L_list"]).flatten().astype(int)

    # Ensure consistent size
    n = preds_rsos_4.size
    if preds_rsos_1.size != n or preds_unet_4.size != n:
        raise ValueError(f"Prediction size mismatch: {preds_rsos_4.size} vs {preds_rsos_1.size} vs {preds_unet_4.size}")
    if l_list.size != n and l_list.size == n // 2:
        l_list = np.concatenate([l_list, l_list])

    # Labels: first half = doublet (1), second half = singlet (0)
    half = n // 2
    labels = np.concatenate([np.ones(half), np.zeros(half)])

    # Compute PC by length
    lengths_to_eval = np.array([4, 5, 6, 7, 8])
    pc_rsos_4 = []
    pc_rsos_1 = []
    pc_unet_4 = []

    for lval in lengths_to_eval:
        mask = l_list == lval
        y = labels[mask]
        
        if len(np.unique(y)) < 2:
            pc_rsos_4.append(np.nan)
            pc_rsos_1.append(np.nan)
            pc_unet_4.append(np.nan)
        else:
            pc_rsos_4.append(roc_auc_score(y, preds_rsos_4[mask]))
            pc_rsos_1.append(roc_auc_score(y, preds_rsos_1[mask]))
            pc_unet_4.append(roc_auc_score(y, preds_unet_4[mask]))

    pc_rsos_4 = np.array(pc_rsos_4)
    pc_rsos_1 = np.array(pc_rsos_1)
    pc_unet_4 = np.array(pc_unet_4)

    # Save CSV
    with open(args.output_csv, "w") as f:
        f.write("L,PC_rSOS_1x,PC_rSOS_4x,PC_UNet_4x\n")
        for i, lval in enumerate(lengths_to_eval):
            f.write(f"{lval},{pc_rsos_1[i]:.6f},{pc_rsos_4[i]:.6f},{pc_unet_4[i]:.6f}\n")

    # Plot
    plt.figure(figsize=(7, 5.5))
    plt.plot(lengths_to_eval, pc_rsos_1, 'o-', color='royalblue', linewidth=2, markersize=8, label='rSOS @ 1x (fully-sampled)')
    plt.plot(lengths_to_eval, pc_rsos_4, 's-', color='orange', linewidth=2, markersize=8, label='rSOS @ 4x')
    plt.plot(lengths_to_eval, pc_unet_4, '^-', color='red', linewidth=2, markersize=8, label='U-Net @ 4x')
    
    plt.xlabel('Signal length', fontsize=12)
    plt.ylabel('PC (AUC)', fontsize=12)
    plt.ylim(0.5, 1.05)
    plt.xlim(3.5, 8.5)
    plt.xticks(lengths_to_eval)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=300)
    print(f"Figure saved to: {args.output_png}")

    # Print results
    print(f"Per-length PC values saved to: {args.output_csv}")
    print("\nResults:")
    print(f"{'L':>3} {'rSOS 1x':>10} {'rSOS 4x':>10} {'U-Net 4x':>10}")
    for i, lval in enumerate(lengths_to_eval):
        print(f"{lval:>3} {pc_rsos_1[i]:>10.4f} {pc_rsos_4[i]:>10.4f} {pc_unet_4[i]:>10.4f}")


if __name__ == "__main__":
    main()
