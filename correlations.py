import sys
import os
import glob
import numpy as np
from Corrfunc.theory.xi import xi
from gadgetutils import snapshot


if __name__ == "__main__":

    num_threads = 4

    run = sys.argv[1].rstrip('/')
    save_dir = "summary_data/" + run.split('/')[-1]
    os.makedirs(save_dir, exist_ok=True)

    group_files = glob.glob(run + "/fof_subhalo_*.hdf5")
    snap_files = glob.glob(run + "/snapshot_*.hdf5")
    group_files.sort()
    snap_files.sort()

    for i, f in enumerate(group_files):
        print(f"Starting {snap_files[i]}\n     and {f}")
        hc = snapshot.load(f, load_subhalos=True)
        pd = snapshot.load(snap_files[i])
        print(f"a = {pd.a:.2f}")

        r_max = np.log10(pd.box_size/4)
        r_bins = np.logspace(-1, r_max, 20)
        print("Using r bins", r_bins)

        print("Starting FoF groups...")
        if hc.n_halos > 2:
            results_fof = xi(pd.box_size, num_threads, r_bins, hc.pos[:,0], hc.pos[:,1], hc.pos[:,2],
                    output_ravg=True, verbose=True)
        else:
            print("No FoF groups present")
            results_fof = 0
        print("Starting subhalos...")
        if hc.n_subhalos > 2:
            results_sh = xi(pd.box_size, num_threads, r_bins, hc.pos_sh[:,0], hc.pos_sh[:,1], hc.pos_sh[:,2],
                    output_ravg=True, verbose=True)
        else:
            print("No subhalos present")
            results_sh = 0
        print("Starting DM particles...")
        down_sample = max(1, int(pd.n_parts/1e6))
        results_dm = xi(pd.box_size, num_threads, r_bins, pd.pos[::down_sample,0], pd.pos[::down_sample,1], pd.pos[::down_sample,2],
                output_ravg=True, verbose=True)

        save_file = save_dir + f"/correlations_{pd.snap_num:03}.npz"
        print(f"Saving data to {save_file}...")
        np.savez(save_file, r_bins=r_bins, results_fof=results_fof, results_sh=results_sh, results_dm=results_dm)
    print("Done.")
