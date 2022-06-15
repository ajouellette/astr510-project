# References:
# GADGET-4 docs
# PlotPowerSpec.pro from GADGET-4
# Springel et al 2018
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl


def read_powerspec_file(fname, pieces=3):
    with open(fname) as f:
        data = []
        for piece in range(pieces):
            a = float(f.readline())
            nbins = int(f.readline())
            box_size = float(f.readline())
            Ngrid = int(f.readline())
            Dplus = float(f.readline())

            kvals = np.zeros(nbins)
            delta = np.zeros(nbins)
            mode_count = np.zeros(nbins, dtype=int)
            shot_noise = np.zeros(nbins)

            for i in range(nbins):
                line = f.readline().split()
                kvals[i] = float(line[0])
                delta[i] = float(line[1])
                mode_count[i] = int(line[3])
                shot_noise[i] = float(line[4])
            data.append([kvals, delta, mode_count, shot_noise])

    return a, box_size, Ngrid, Dplus, data


def rebin_powerspec(kvals, mode_count, data, min_modes=10, max_bins=50):
    rebinned_k = []
    rebinned_count = []
    rebinned_data = []

    min_dlogk = (np.log10(np.max(kvals)) - np.log10(np.min(kvals))) / max_bins

    istart = 0
    ind = [istart,]
    while istart < len(kvals):
        istart += 1

        dlogk = np.log10(np.max(kvals[ind])) - np.log10(np.min(kvals[ind]))
        count = np.sum(mode_count[ind])
        if dlogk > min_dlogk and count > min_modes:
            k = np.sum(kvals[ind] * mode_count[ind]) / count
            new_data = np.sum(data[ind] * mode_count[ind]) / count
            rebinned_k.append(k)
            rebinned_data.append(new_data)
            rebinned_count.append(count)

            ind = [istart,]
        else:
            ind.append(istart)

    rebinned_k = np.array(rebinned_k)
    rebinned_count = np.array(rebinned_count)
    rebinned_data = np.array(rebinned_data)

    return rebinned_k, rebinned_count, rebinned_data


def get_full_powerspec(fname, fold_fac=16, min_count=10, max_bins=50):
    a, box_size, Ngrid, _, data = read_powerspec_file(fname)

    k_all = []
    delta_all = []

    for piece in range(len(data)):
        kvals, delta, mode_count, shot_noise = data[piece]

        pshot = np.mean(shot_noise / (kvals**3 / (2*np.pi**2)))

        if piece == 0:
            kmin = 2*np.pi / box_size
        else:
            kmin = np.pi / box_size * Ngrid / 4 * fold_fac**(piece-1)
        kmax = np.pi / box_size * Ngrid / 4 * fold_fac**piece

        # remove shot noise
        delta -= shot_noise

        # re-bin power spectrum to coarser bins, enforce minimum of 10 modes per bin
        rebinned_k, rebinned_count, rebinned_delta = rebin_powerspec(kvals, mode_count, delta, min_modes=min_count, max_bins=max_bins)

        # only keep data within kmin, kmax range and above 0.5 x shot noise level
        ind = (rebinned_k > kmin) * (rebinned_k < kmax) * (rebinned_delta > 0.5 * pshot * rebinned_k ** 3 / (2*np.pi**2))
        k_all.append(rebinned_k[ind])
        delta_all.append(rebinned_delta[ind])

    k = np.hstack(k_all)
    delta = np.hstack(delta_all)

    return k, delta


if __name__ == "__main__":
    run_dir = sys.argv[1].rstrip('/')
    save_dir = "summary_data/" + run_dir.split('/')[-1]
    os.makedirs(save_dir, exist_ok=True)

    if "plot" in sys.argv:
        k, delta = get_full_powerspec(run_dir + "/powerspecs/powerspec_002.txt")

        cosmo = ccl.Cosmology(Omega_c=0.315, Omega_b=0.049, h=1, sigma8=0.811, n_s=0.965)
        k_ccl = np.logspace(-2, 2, 50)
        nonlin_power = cosmo.nonlin_power(k_ccl, 1)
        delta_nonlin = k_ccl**3 * nonlin_power / (2*np.pi**2)

        plt.plot(k, delta, '.', label="GADGET")
        plt.plot(k_ccl, delta_nonlin, '--', alpha=0.7, label="CCL")
        plt.loglog()
        plt.xlabel("$k$")
        plt.ylabel("$\\Delta^2(k)$")
        plt.legend()
        plt.show()
    else:
        ps_files = glob.glob(run_dir + "/powerspecs/powerspec_*.txt")
        ps_files.sort()
        for f in ps_files:
            print(f"Starting {f}...")
            num = f.split('_')[-1].split('.')[0]
            k, delta = get_full_powerspec(f)

            save_file = save_dir + f"/powerspec_{num}.npz"
            print(f"Saving data to {save_file}...")
            np.savez(save_file, k=k, delta=delta)
        print("Done.")
