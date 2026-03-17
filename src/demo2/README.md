# Synthetic defect insertion

This script inserts doublet and singlet signals into DDPM generated MR images from demo 1. The singlet versus doublet signals for different signal lengths are determined using the 2AFC-based detection table provided below. This, in turn, means that singlet and doublet signals are set based on the acceleration factor, signal contrast, and signal length (in pixels). This code saves the objects with signals in HDF5 format.

Command-line Options:

```
Acceleration (int):           Acceleration factor for sparse sampling (2, 4, 6, or 8).
Amplitude (float):            Contrast value relevant to the acceleration factor.
Signal lengths (str):         Comma-separated signal separation lengths, e.g. "4,5,6,7,8".
Object NPZ path (optional):   Path to the DDPM-generated `.npz` file from demo 1.
```

Usage:

```
python signal_insertion_test.py [acceleration factor] [contrast] [signal_lengths] [object_npz_path]
```

Examples:
    Run with acceleration factor 4 corresponding to the 7th row in the 2-AFC table below (also employed for testing in our DLMO paper):

```
python signal_insertion_test.py 4 0.7 '4,5,6,7,8'
```


The output HDF5 files are saved to `./objects/`. Each file contains datasets `H_0` (singlet reconstructions), `H_1` (doublet reconstructions), and `L_list` (signal lengths).