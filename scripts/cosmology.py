import numpy as np
import pyccl as ccl


# simulations start at z = 63
z_start = 63
a_start = 1 / (z_start + 1)


# Planck 2018 cosmology
cosmo = ccl.Cosmology(Omega_c=0.315, Omega_b=0.049, h=0.674, sigma8=0.811, n_s=0.965)
