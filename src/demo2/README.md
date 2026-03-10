# Synthetic defect insertion

This script inserts doublet and singlet signals into DDPM generated MR images from demo 1. The singlet versus doublet signals for different signal lengths are determined using the 2AFC-based detection table provided below. Accordingly, singlet and doublet signals are set based on the acceleration factor, signal contrast, and signal length (in pixels). This code saves the objects with signals in HDF5 format.

Command-line Options:

```
Acceleration (int): Acceleration factor for sparse sampling (2, 4, 6, or 8).
Amplitude (float): Contrast value relavent to the acceleration factor. 
signal lengths (list): Signal separation lengths as an array. 
```

Usage:

```
python signal_insertion_test.py [acceleration factor] [contrast] [signal_lengths_as_an_array]
```

Examples:
    Run with acceleration factor 1 corresponding to the 3rd row in the 2-AFC table below:

```
python signal_insertion_test.py 1 0.3 4:8
```