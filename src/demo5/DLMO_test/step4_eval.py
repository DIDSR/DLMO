#!/usr/bin/env python
"""
Run Step 4 by reading model-observer outputs and then plotting percent-correct (AUC) vs signal length.

This script's default options reads dlmo_test_hvd.py applied three times for:
1) rSOS @ 4x  
2) rSOS @ 1x (fully sampled)
3) U-Net @ 4x

Then computes AUC by signal length L={4,5,6,7,8} and plots the result.
"""
import argparse
import os
import sys
import numpy as np
import h5py
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Step 4: Final Percent correct (PC)-based output companring different reconstruction methods.")
    parser.add_argument("--dlmo-eval-on-ref", default='./dlmo_discrimination/acc1/preds_rsos.npy', \
                        help="DLMO-based output file on the discrimination/detection output from the reference method.")
    parser.add_argument("--dlmo-eval-on-acc-rsos", default='./dlmo_discrimination/acc4/preds_rsos.npy', \
                        help="DLMO-based output file on the discrimination/detection task evaluated using old method \
                        on accelerated acquisition.")
    parser.add_argument("--dlmo-eval-on-acc-ai-rec", default='./dlmo_discrimination/acc4/preds_unet.npy', \
                        help="DLMO-based output file on the discrimination/detection task evaluated using new AI method \
                        on accelerated acquisition.")
    parser.add_argument("--l-list-file", default="../../demo3/rsos_rec/test_acc4_at_acc1_rsos.hdf5",
                        help="HDF5 file containing L_list (typically rSOS 1x)")
    parser.add_argument("--recon-method-str", default="rSOS_1x,rSOS_4x,U-Net_4x",
                        help="Reconstruction method names for the reference method, accelerated legacy method, \
                        and accelerated AI-based method.")
    parser.add_argument('--signal-length-arr', nargs='+', type=int, help='Input an array of integers')
    parser.add_argument("--output-png", default="./dlmo_discrimination/step4.png", help="Output PNG path")
    parser.add_argument("--output-csv", default="./dlmo_discrimination/step4.csv", help="Output CSV path")
    args = parser.parse_args()

    # recons method types -------------------------------------------------------------
    recon_methods      = args.recon_method_str.split(',')
    ref_method_str     = recon_methods[0]
    acc_old_method_str = recon_methods[1]
    acc_ai_method_str  = recon_methods[2]

    # Reading the DLMO output on the discrimination task (i.e. ,singlet vs dublet signals)
    # for the three methods
    preds_rsos_acc = np.load(args.dlmo_eval_on_acc_rsos)
    preds_rsos_ref = np.load(args.dlmo_eval_on_ref)
    preds_ai_acc   = np.load(args.dlmo_eval_on_acc_ai_rec)
 
    # Load signal separation length list ---------------------------------------------------------------------------------
    with h5py.File(args.l_list_file, "r") as hf:
        l_list = np.array(hf["L_list"]).flatten().astype(int)

    # Ensure consistent size -----------------------------------------------------------------------------------------------
    n = preds_rsos_acc.size
    if preds_rsos_ref.size != n or preds_ai_acc.size != n:
        raise ValueError(f"Prediction size mismatch: {preds_rsos_acc.size} vs {preds_rsos_ref.size} vs {preds_ai_acc.size}")
    if l_list.size != n and l_list.size == n // 2:
        l_list = np.concatenate([l_list, l_list])
    # Labels: first half = doublet (0), second half = singlet (1)
    half = n // 2
    labels = np.concatenate([np.zeros(half), np.ones(half)])

    # Compute PC by length
    lengths_to_eval = np.array(args.signal_length_arr) #np.array([4, 5, 6, 7, 8])-4
    pc_rsos_acc     = []
    pc_rsos_ref     = []
    pc_ai_acc       = []
    print('\nSignal lengths considered in this analysis is:', lengths_to_eval)
    
    l_list= l_list+lengths_to_eval[0] # shifting L_list by the first index in the signal length array
    print('\nSignal seperation lengths corresponding to', n, 'reconstructions in this analysis:')
    print(l_list) 
    
    print('\nThe three reconstructions methods considered in this analysis are:', \
        ref_method_str, acc_old_method_str, acc_ai_method_str)
    for lval in lengths_to_eval:
        mask = l_list == lval
        y = labels[mask]
        
        if len(np.unique(y)) < 2:
            pc_rsos_acc.append(np.nan)
            pc_rsos_ref.append(np.nan)
            pc_ai_acc.append(np.nan)
        else:
            pc_rsos_acc.append(roc_auc_score(y, preds_rsos_acc[mask]))
            pc_rsos_ref.append(roc_auc_score(y, preds_rsos_ref[mask]))
            pc_ai_acc.append(roc_auc_score(y, preds_ai_acc[mask]))

    pc_rsos_acc = np.array(pc_rsos_acc)
    pc_rsos_ref = np.array(pc_rsos_ref)
    pc_ai_acc = np.array(pc_ai_acc)

    # Save CSV
    with open(args.output_csv, "w") as f:
        #f.write("L,PC_rSOS_1x,PC_rSOS_4x,PC_UNet_4x\n")
        f.write("L,PC_"+ref_method_str+",PC_"+acc_old_method_str+",PC_"+acc_ai_method_str+"\n")
        for i, lval in enumerate(lengths_to_eval):
            f.write(f"{lval},{pc_rsos_ref[i]:.6f},{pc_rsos_acc[i]:.6f},{pc_ai_acc[i]:.6f}\n")

    # Plot
    plt.figure(figsize=(7, 5.5))
    plt.plot(lengths_to_eval, pc_rsos_ref, 'o-', color='royalblue', linewidth=2, markersize=8, label=ref_method_str)
    plt.plot(lengths_to_eval, pc_rsos_acc, 's-', color='orange', linewidth=2, markersize=8, label=acc_old_method_str)
    plt.plot(lengths_to_eval, pc_ai_acc, '^-', color='red', linewidth=2, markersize=8, label=acc_ai_method_str)

    plt.xlabel('Signal length', fontsize=12)
    plt.ylabel('PC (AUC)', fontsize=12)
    plt.ylim(0.3, 1.05)
    plt.xlim(3.5, 8.5)
    plt.xticks(lengths_to_eval)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.show()
    plt.savefig(args.output_png, dpi=300)
    print(f"Figure saved to: {args.output_png}\n")

if __name__ == "__main__":
    main()


