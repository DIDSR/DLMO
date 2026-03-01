# Syntheric defect insertion

This script inserts doulet and singlet signals into DDPM generated objects. It saves the objects with signals in HDF5 format.

Command-line Options:

```
Acceleration (int): Acceleration factor for sparse sampling (2, 4, 6, or 8).
```

Usage:

```
python signal_insertion_test.py [acceleration factor]
```

Examples:
    Run with acceleration factor 4:

```
python signal_insertion_test.py 4
```