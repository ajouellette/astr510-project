import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from gadgetutils import snapshot

plt.style.use(["science", "notebook"])


if __name__ == "__main__":
    directory = sys.argv[1].rstrip("/")
    slice_w = float(sys.argv[2])

    files = glob.glob(directory + "/snapshot_*.hdf5")
    files.sort()

    for f in files:
        print(f)
        pd = snapshot.ParticleData(f, load_vels=False, load_ids=False)
        zslice = np.abs(pd.pos[:,2]) < slice_w

        plt.figure(figsize=(12,9))
        plt.hexbin(pd.pos[zslice,0], pd.pos[zslice,1], norm=colors.LogNorm(), cmap="inferno")
        plt.xlabel("X, $h^{-1}$ Mpc")
        plt.ylabel("Y, $h^{-1}$ Mpc")
        plt.title(f"a = {pd.a:.2f}")
        plt.colorbar(label="projected density")
        plt.savefig(f"slice_{directory.split('/')[-1]}_{round(pd.n_parts**(1/3))}_{pd.snap_num:03}.png")
