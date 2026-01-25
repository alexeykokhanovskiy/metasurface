import numpy as np
import os
import metasurface
import h5py


from metasurface import Spectrum, Geometry, Material, Metasurface

# -----------------------
# ТЕСТ
# -----------------------

def test_metasurface_save_hdf5():

    # --- Geometry ---
    h = 0.1
    w = 0.2
    geom = Geometry(
        name="LISSP",
        parameters={
            "height": h,
            "width": w
        }
    )

    # --- Materials ---
    mat = Material(
        materials={
            "layer1": "SiO2",
            "layer2": "GST"
        }
    )

    # --- Spectrum ---
    wavelength = np.linspace(1.0, 2.0, 100)
    value = np.sin(wavelength)

    spec = metasurface.Spectrum(
        wavelength=wavelength,
        value=value,
        type="R",
        polarization="TM",
        incidence_angle=0.0
    )

    # --- Metasurface ---
    ms = Metasurface(
        geometry=geom,
        materials=mat,
        spectra={"R_TM": spec},
        metadata="test metasurface for HDF5 save"
    )

    # --- Save ---
    filename = "test_metasurface.h5"
    ms.save_hdf5(filename)

    # --- Simple check ---
    assert os.path.exists(filename), "HDF5 file was not created"

    print(f"[OK] Metasurface saved to {filename}")



if __name__ == "__main__":
    test_metasurface_save_hdf5()
