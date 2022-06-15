import sys
import numpy as np
import matplotlib.pyplot as plt
from cosmology import *

plt.style.use(["science", "notebook"])


if __name__ == "__main__":
    fname = sys.argv[1]

    data = np.loadtxt(fname)

    a = np.linspace(0.01, 1, 50)
    growth = cosmo.growth_factor(a)

    D_ccl = cosmo.growth_factor(a_start)
    print(f"At a = {a_start:.6e}")
    print(D_ccl)
    print(data[0])
    print((data[0,1] - D_ccl) / D_ccl)

    plt.plot(data[:,0], data[:,1], 'o', label="GADGET")
    plt.plot(a, growth, label="CCL")
    plt.xlabel("scale factor $a$")
    plt.ylabel("linear growth factor $D_+(a)$")
    plt.legend()
    plt.loglog()
    plt.show()
