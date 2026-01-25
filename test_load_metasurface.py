import numpy as np
import os

from metasurface import Spectrum, Geometry, Material, Metasurface


def test_metasurface_save_load_hdf5():

    # --- Geometry ---
    geom = Geometry(
        name="LISSP",
        parameters={
            "height": 0.1,
            "width": 0.2
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
    wl = np.linspace(1.0, 2.0, 50)
    val = np.cos(wl)

    spec = Spectrum(
        wavelength=wl,
        value=val,
        type="R",
        polarization="TM",
        incidence_angle=15.0
    )

    ms = Metasurface(
        geometry=geom,
        materials=mat,
        spectra={"R_TM": spec},
        metadata="round-trip test"
    )

    filename = "roundtrip_test.h5"

    # --- Save ---
    ms.save_hdf5(filename)
    assert os.path.exists(filename)

    # --- Load ---
    ms_loaded = Metasurface.load_hdf5(filename)

    # --- Checks ---
    assert ms_loaded.metadata == ms.metadata

    assert ms_loaded.geometry.name == ms.geometry.name
    assert ms_loaded.geometry.parameters == ms.geometry.parameters

    assert ms_loaded.materials.materials == ms.materials.materials

    assert "R_TM" in ms_loaded.spectra
    spec_l = ms_loaded.spectra["R_TM"]

    assert spec_l.type == spec.type
    assert spec_l.polarization == spec.polarization
    assert spec_l.incidence_angle == spec.incidence_angle
    assert np.allclose(spec_l.wavelength, spec.wavelength)
    assert np.allclose(spec_l.value, spec.value)

    print("[OK] Metasurface save/load round-trip successful")


if __name__ == "__main__":
    test_metasurface_save_load_hdf5()
