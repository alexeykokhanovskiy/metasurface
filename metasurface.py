
from dataclasses import dataclass, field
import numpy as np
import h5py
@dataclass
class Spectrum:
    wavelength: np.ndarray
    value: np.ndarray
    type: str # R or T
    polarization: str
    incidence_angle: float = 0.0

    def interpolate(self, wl_new: np.ndarray) -> np.ndarray:
        return np.interp(wl_new, self.wavelength, self.value)

@dataclass
class Geometry:
    name: str
    parameters: dict[str, float]
    def get(self, name: str) -> float:
        return self.parameters[name]

    def set(self, name: str, value: float):
        self.parameters[name] = value

    def summary(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.name}: geometry {params}"

    def summary_detailed(self) -> str:
        """Return multi-line formatted summary with 4 decimal precision."""
        lines = [f"Geometry: {self.name}"]
        for key, val in self.parameters.items():
            lines.append(f"  {key} = {val:.4f}")
        return "\n".join(lines)

@dataclass
class Material:
    materials: dict[str,str]

    def summary(self) -> str:
        mater = ", ".join(f"{k}={v}" for k, v in self.materials.items())
        return f"{mater}"

@dataclass
class Metasurface:
    geometry: Geometry
    materials: Material
    spectra: dict[str, Spectrum] = field(default_factory=dict)
    metadata: str = None

    def add_spectrum(self, name:str, spectrum: Spectrum):
        self.spectra[name] = spectrum

    def spectra_summary(self):
        if not self.spectra:
            return '-'
        return ", ".join(self.spectra.keys())

    def save_hdf5(self, filename: str):
        with h5py.File(filename, "w") as f:
            self._write_to_group(f)

    def _write_to_group(self, group):
        """
        Write metasurface data to an HDF5 group.

        Args:
            group: h5py.Group or h5py.File object to write to
        """
        # ---------- metadata ----------
        if self.metadata is not None:
            group.attrs["metadata"] = self.metadata

        # ---------- geometry ----------
        g_geom = group.create_group("geometry")
        g_geom.attrs["name"] = self.geometry.name

        g_params = g_geom.create_group("parameters")
        for key, val in self.geometry.parameters.items():
            g_params.create_dataset(key, data=val)

        # ---------- materials ----------
        g_mat = group.create_group("materials")
        for key, val in self.materials.materials.items():
            g_mat.attrs[key] = val

        # ---------- spectra ----------
        g_spec = group.create_group("spectra")
        for name, spec in self.spectra.items():
            g_s = g_spec.create_group(name)

            g_s.create_dataset("wavelength", data=spec.wavelength)
            g_s.create_dataset("value", data=spec.value)

            g_s.attrs["type"] = spec.type
            g_s.attrs["polarization"] = spec.polarization
            g_s.attrs["incidence_angle"] = spec.incidence_angle

    @staticmethod
    def load_hdf5(filename: str) -> "Metasurface":
        with h5py.File(filename, "r") as f:
            return Metasurface._read_from_group(f)

    @staticmethod
    def _read_from_group(group) -> "Metasurface":
        """
        Read metasurface data from an HDF5 group.

        Args:
            group: h5py.Group or h5py.File object to read from

        Returns:
            Metasurface object
        """
        # ---------- metadata ----------
        metadata = group.attrs.get("metadata", None)

        # ---------- geometry ----------
        g_geom = group["geometry"]
        geom_name = g_geom.attrs["name"]

        params = {}
        for key, ds in g_geom["parameters"].items():
            params[key] = float(ds[()])

        geometry = Geometry(
            name=geom_name,
            parameters=params
        )

        # ---------- materials ----------
        materials = {}
        for key, val in group["materials"].attrs.items():
            materials[key] = val

        material = Material(materials=materials)

        # ---------- spectra ----------
        spectra = {}
        for name, g_s in group["spectra"].items():
            spec = Spectrum(
                wavelength=np.array(g_s["wavelength"]),
                value=np.array(g_s["value"]),
                type=g_s.attrs["type"],
                polarization=g_s.attrs["polarization"],
                incidence_angle=float(g_s.attrs["incidence_angle"])
            )
            spectra[name] = spec

        return Metasurface(
            geometry=geometry,
            materials=material,
            spectra=spectra,
            metadata=metadata
        )

if __name__ == "__main__":
    h = 0.1
    t = 0.2
    #
    geom = Geometry(name = 'LISSP', parameters= {'height': h, 'width': t  })
    mat = Material(materials = {'layer1': 'SiO2', 'layer2': 'GST'} )
    spec = Spectrum(wavelength=1.0, value = 1.0, type = 'R', polarization='TM', incidence_angle=0.0)
    #
    #
    ms = Metasurface(geometry = geom, materials = mat, spectra = {'R_TM': spec}, metadata=None)
    print(ms.spectra_summary())
