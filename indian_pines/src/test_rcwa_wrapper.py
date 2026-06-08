"""
Quick smoke test for RCWAWrapper.
Run from indian_pines/src/:
    python test_rcwa_wrapper.py
"""

import numpy as np
from rcwa_wrapper import RCWAWrapper

wl_nm = np.linspace(400, 1000, 20)   # coarse grid for speed

rcwa = RCWAWrapper(wl_nm=wl_nm)

# Single pillar
print("Testing single pillar d=200 nm ...")
T = rcwa.transmission(200.0)
print(f"  T shape: {T.shape}  min={T.min():.4f}  max={T.max():.4f}")
assert T.shape == (20,), "wrong shape"
assert not np.any(np.isnan(T)), "NaN in T"
assert not np.any(T < -0.01), "T < 0"
assert not np.any(T > 1.01),  "T > 1"

# Cache hit (same diameter — should return instantly, no MATLAB call)
print("Testing cache hit ...")
T2 = rcwa.transmission(200.0)
assert np.array_equal(T, T2), "cache returned different result"
print("  Cache hit OK")

# Filter matrix for 3 diameters
print("Testing filter_matrix for [100, 200, 300] nm ...")
F = rcwa.filter_matrix([100.0, 200.0, 300.0])
print(f"  F shape: {F.shape}  min={F.min():.4f}  max={F.max():.4f}")
assert F.shape == (3, 20), "wrong shape"

rcwa.close()
print("\nAll checks passed.")
