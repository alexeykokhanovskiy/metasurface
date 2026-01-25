from metasurface import Geometry, Material, Spectrum, Metasurface
import numpy as np

h = 0.1
t = 0.2
#
geom = Geometry(name = 'LISSP', parameters= {'height': h, 'width': t  })
mat = Material(materials = {'layer1': 'SiO2', 'layer2': 'GST'} )


wl = np.linspace(1.0,4.0, 100)
value = np.ones_like(wl)
spec = Spectrum(wavelength=wl, value = value, type = 'R', polarization='TM', incidence_angle=0.0)
value = np.zeros_like(wl)
spec2 = Spectrum(wavelength=wl, value = value, type = 'R', polarization='TM', incidence_angle=0.0)


#
#
ms = Metasurface(geometry = geom, materials = mat, spectra = {'R_TM': spec, 'T_TM': spec2}, metadata=None)
# ms.save_hdf5('testing.h5')

ms.load_hdf5('testing.h5')


print(ms.spectra_summary())
