import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import odr
from scipy.stats import gaussian_kde
from gadgetutils import snapshot, utils

plt.style.use(["science", "notebook"])


def nfw(params, r):
    """NFW density profile."""
    rho_0, R_s = np.abs(params)
    return rho_0 / (r/R_s * (1 + r/R_s)**2)


def nfw_param_grad(params, r):
    rho_0, r_s = params
    rho = nfw(params, r)
    return np.array([rho/rho_0, -rho/r_s * (1/(r/r_s) + 2/(1+r/r_s))])


nfw_model = odr.Model(nfw, fjacb=nfw_param_grad)


def get_sphere_samples(r, n):
    """Sample n points on a sphere with radius r.

    Uses fibonaci spiral
    http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/#more-3069
    """
    golden_ratio = (1 + 5**0.5)/2
    i = np.arange(n)
    theta = 2 * np.pi * i / golden_ratio
    phi = np.arccos(1 - 2*(i + 0.5)/n)
    return np.vstack((r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(phi))).T


if __name__ == "__main__":
    med_samples_per_r = 5
    min_samples_per_r = 3
    max_samples_per_r = 500

    run_dir = sys.argv[1].rstrip('/')
    save_dir = "summary_data/" + run_dir.split('/')[-1]
    os.makedirs(save_dir, exist_ok=True)

    snap_files = glob.glob(run_dir + "/snapshot_*.hdf5")
    snap_files.sort()
    group_files = glob.glob(run_dir + "/fof_subhalo_tab_*.hdf5")
    group_files.sort()
    for i, f in enumerate(snap_files):
        print(f"\nStarting snapshot {i}...")
        pd = snapshot.load(f, load_vels=False, load_ids=False, make_tree=True)
        hc = snapshot.load(group_files[i])

        if hc.n_halos < 10:
            print("No halos present")
            continue

        virial_density = 358 * pd.mean_matter_density/pd.part_mass

        radii = []
        radii_scaled = []
        densities = []
        densities_scaled = []

        for halo in range(10):
            print(f"Starting halo {halo}...")
            inds = hc.get_particles(halo)
            print(f"\t{len(inds)} particles")
            center = hc.pos[halo]
            pos = utils.center_box_pbc(pd.pos[inds], center, pd.box_size)
            radius = np.max(np.linalg.norm(pos, axis=1))
            pos = utils.center_box_pbc(pd.pos[pd.query_radius(center, radius)], center, pd.box_size)

            # use a gaussian KDE with bandwidth proportional to grav. softening
            kde = gaussian_kde(pos.T, bw_method=0.82*pd.softening)
            r_eval = np.logspace(-2, np.log10(radius), 20)
            # calculate number of samples at each r, should be proportional to area of sphere
            areas = 4 * np.pi * r_eval**2
            coeff = med_samples_per_r / np.median(areas)
            samples = np.round(np.clip(coeff * areas, min_samples_per_r, max_samples_per_r)).astype(int)

            prob_dens = np.zeros_like(r_eval)
            errors = np.zeros_like(r_eval)
            for i, r in enumerate(r_eval):
                pos_eval = get_sphere_samples(r, samples[i])
                # evaluate KDE model
                dens = kde.evaluate(pos_eval.T)
                prob_dens[i] = np.mean(dens)
                errors[i] = np.std(dens)

            # get estimate for virial radius
            r_centered = 0.5 * (r_eval[1:] + r_eval[:-1])
            avg_dens = np.cumsum(4*np.pi * r_centered**2 * (r_eval[1:] - r_eval[:-1]) * 0.5*(prob_dens[1:] + prob_dens[:-1])) / (4/3*np.pi * r_centered**3)
            i_vir = np.argmin(np.abs(avg_dens * len(pos) - virial_density))
            r_vir = r_centered[i_vir]

            # fit NFW profile to r < Rvir/2
            mask = r_eval < r_vir / 2
            print(f"\tVirial radius: {r_vir:.3f}", np.sum(mask), "particles inside R_vir")
            data = odr.RealData(r_eval[mask], prob_dens[mask], sy=errors[mask])
            #rho_0 = 0.5*(np.max(prob_dens) + prob_dens[i_vir]) / 10
            rho_0 = avg_dens[np.argmin(np.abs(r_centered - r_vir/6))]
            print(f"\tInitial guess for rho_0: {rho_0:.3f}, Initial guess for R_s: {r_vir/6:.3f}")
            odr_fit = odr.ODR(data, nfw_model, beta0=[rho_0, r_vir/6])
            odr_output = odr_fit.run()
            odr_output.pprint()
            print()

            # re-scale to unitless profile
            r_scaled = r_eval / abs(odr_output.beta[1])
            density_scaled = prob_dens / abs(odr_output.beta[0])

            radii.append(r_eval)
            radii_scaled.append(r_scaled)
            densities.append(prob_dens * len(pos))
            densities_scaled.append(density_scaled)

        radii = np.hstack(radii)
        radii_scaled = np.hstack(radii_scaled)
        densities = np.hstack(densities)
        densities_scaled = np.hstack(densities_scaled)

        num = f.split('_')[-1].split('.')[0]
        save_file = save_dir + f"/density_profiles_{num}.npz"
        np.savez(save_file, r=radii, r_scaled=radii_scaled, density=densities, density_scaled=densities_scaled)
