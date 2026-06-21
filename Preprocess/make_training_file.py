"""
Build a training-subset HDF5 by copying everything except COD from the full database.
The original file is left untouched as the backup.

Run after build_db.py finishes:
    python make_training_file.py
"""

import hdf5plugin  # registers ZFP + bitshuffle decompressors
import h5py
import os
import sys

SRC  = "I(q)@L=50.h5"
DST  = "I(q)@L=50_train.h5"
SKIP = {"COD"}  # add groups to skip here

if not os.path.exists(SRC):
    sys.exit(f"Source file not found: {SRC}")

if os.path.exists(DST):
    sys.exit(f"Destination already exists: {DST}  — delete it first if you want to rebuild.")

print(f"Source : {SRC}  ({os.path.getsize(SRC)/1e9:.1f} GB)")
print(f"Dest   : {DST}")
print(f"Skipping groups: {SKIP}")
print()

with h5py.File(SRC, "r") as src, h5py.File(DST, "w") as dst:
    # root attributes
    for k, v in src.attrs.items():
        dst.attrs[k] = v

    for name in src:
        if name in SKIP:
            print(f"  skip   {name}")
            continue
        item = src[name]
        n    = len(item) if isinstance(item, h5py.Group) else item.shape
        print(f"  copy   {name}  ({n})")
        src.copy(name, dst, name=name)

print()
print(f"Done.  {DST}  ({os.path.getsize(DST)/1e9:.1f} GB)")
print(f"Original {SRC} is your full backup.")
