# MR acquisition and reconstruction

This example shows forward projection and reconstruction of DDPM generated objects using the rSOS method to create test dataset. It saves the reconstructions in HDF5 format.

Command-line Options:

```
Acceleration (int): Acceleration factor for sparse sampling (2, 4, 6, or 8).
```

Usage:

```
python rsos_ddpm_test.py [acceleration factor]
```

Examples:
	Run with acceleration factor 4:

```
python rsos_ddpm_test.py 4
```