import sys
import numpy as np
import matplotlib.pyplot as plt
from pyccl import halos
from gadgetutils import snapshot
from cosmology import *

plt.style.use(["science", "notebook"])


if __name__ == "__main__":
    fname = sys.argv[1]
    hc = snapshot.HaloCatalog(fname)
    print(hc.a)

    bins = np.logspace(14, 15.2, 15)
    hmf, errors = hc.calc_fof_hmf(bins)

    hmfunc_angulo = halos.hmfunc.MassFuncAngulo12(cosmo)
    hmfunc_press = halos.hmfunc.MassFuncPress74(cosmo)

    plt.errorbar(bins, hmf, yerr=errors, fmt='o', label="simulation")
    plt.plot(bins, hmfunc_angulo.get_mass_function(cosmo, bins, 1), label="Angulo 2012")
    plt.plot(bins, hmfunc_press.get_mass_function(cosmo, bins, 1), label="Press-Schecter 1974")
    plt.legend()
    plt.xlabel("Halo mass, $h^{-1}$ M$_\\odot$")
    plt.ylabel("n(>M), $h^3$ Mpc$^{-3}$")
    plt.loglog()
    plt.show()
