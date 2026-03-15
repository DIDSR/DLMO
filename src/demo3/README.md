# MR acquisition and reconstruction

This example shows forward projection and reconstruction of DDPM-generated objects using the rSOS method to create a test dataset. It saves the reconstructions in HDF5 format.

Command-line Options:

```
Acceleration (int): Acceleration factor for sparse sampling (2, 4, 6, or 8).
Object NPZ path (optional): Path to the DDPM-generated `.npz` file from demo 1.
```

Usage:

```
python rsos_ddpm_test.py [acceleration factor] [object_npz_path]
```

Examples:
	Run with acceleration factor 4:

```
python rsos_ddpm_test.py 4
```

Input files are `.npz` files generated in the demo1. The reconstructions are saved in HDF5 format in the `./rsos_rec/` folder. Each HDF5 file contains a dataset named `H_0` which holds the singlet image reconstructions, `H_1` for doublet image reconstructions, and `L_list` for the signal length for each reconstruction.

If you omit the optional object path, the script expects the default demo 1 output:

```
../demo5/image_acquisition_and_reconstruction/examples/DDPM_obj/samples_10000x260x311x1.npz
```

Examples of the reconstructions are shown below. The left one is for acceleration factor 4, which contains a singlet signal, and the right one is for acceleration factor 8, which contains a doublet signal. The signal length is 3 for both cases. The reconstructions are noisy, and the doublet signal is not visually distinguishable from the singlet signal, which makes it a challenging task.
![acc4](../../docs/H0_rsos_0.png "Acceleration factor 4")
![acc8](../../docs/H1_rsos_0.png "Acceleration factor 8")