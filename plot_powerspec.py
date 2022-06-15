import sys
import numpy as np
import matplotlib.pyplot as plt
from cosmology import *

plt.style.use(["science", "notebook"])


if __name__ == "__main__":
    fname = sys.argv[1]

    data = np.loadtxt(fname, skiprows=1)

    k = np.logspace(-3, 3, 50)
    linear_power = cosmo.linear_power(k, a_start)

    #plt.plot(k, k**3 * cosmo.linear_matter_power(k, a_start) / (2*np.pi**2))
    #plt.plot(k, k**3 * cosmo.nonlin_matter_power(k, a_start) / (2*np.pi**2))
    #plt.plot(data[:,0], data[:,1] / (2*np.pi**2), 'o')

    plt.plot(k, cosmo.linear_matter_power(k, a_start))
    plt.plot(k, cosmo.nonlin_matter_power(k, a_start))
    plt.plot(data[:,0], data[:,1] / data[:,0]**3 / 50.42, '.')

    plt.axvline(2*np.pi/256)
    plt.axvline(np.pi*512/256)
    plt.loglog()
    plt.xlabel("$k$")
    plt.ylabel("$P(k)$")
    plt.show()
