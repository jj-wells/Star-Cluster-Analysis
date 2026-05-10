#from isochrones.mist import MIST_Isochrone
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from astropy.table import Table
from pathlib import Path
import json
import numpy as np
from functions.v_disp import KMS_PER_AU_PER_YR, filter_catalog_for_dispersion
#import warnings

PROJECT_DIR = Path(__file__).resolve().parents[1]
PYUPMASK_OUTPUT_DIR = PROJECT_DIR / "pyUPMASK-main" / "output"
METRIC_TRIALS_DIR = PROJECT_DIR / "metric_trials"
MST_DIR = PROJECT_DIR / "MST"
ISOCHRONE_DIR = PROJECT_DIR / "isochrones"
FIGURES_CMD_DIR = PROJECT_DIR / "figures_cmd"
FIGURES_DENSITY_DIR = PROJECT_DIR / "figures_density"
FIGURES_STARS_DIR = PROJECT_DIR / "figures_stars"
FIGURES_VEL_DIR = PROJECT_DIR / "figures_vel"
FIGURES_VEL_KDE_DIR = PROJECT_DIR / "figures_vel_kde"
FIGURES_3D_DIR = PROJECT_DIR / "figures_3d"
FIGURES_MST_DIR = PROJECT_DIR / "figures_MST"
FIGURES_METRICS_DIR = PROJECT_DIR / "figures_metrics"
Q_VDISP_CLUSTER_LIST_FILE = FIGURES_METRICS_DIR / "q_vs_v_disp_clusters.json"
CMD_SHIFT_OVERRIDES = {
    "ic348": {"colour": 0.25, "mag": 0.6},
    "m4": {"colour": 0.45, "mag": 1.2},
}
DEFAULT_Q_VDISP_CLUSTERS = ["ic348_5", "m3_1", "m67_1", "m92_2", "plei_13"]


def parse_isochrone_names(isochrone_text):

    names = []
    for raw_name in isochrone_text.replace(";", ",").split(","):
        name = raw_name.strip()
        if not name:
            continue

        stem = Path(name).stem
        exact_file = ISOCHRONE_DIR / f"{stem}.dat"

        if exact_file.exists():
            names.append(stem)
            continue

        prefix_matches = sorted(ISOCHRONE_DIR.glob(f"{stem}_*.dat"))
        if prefix_matches:
            names.extend(path.stem for path in prefix_matches)
        else:
            names.append(stem)

    return list(dict.fromkeys(names))


def read_parsec_isochrone(iso_file):

    try:
        iso = Table.read(str(iso_file), format="ascii")
        if "Gmag" in iso.columns and "G_BPmag" in iso.columns and "G_RPmag" in iso.columns:
            return iso
    except Exception:
        pass

    lines = iso_file.read_text(encoding="utf-8").splitlines()
    header_index = None

    for index, line in enumerate(lines):
        if line.lstrip().startswith("#") and "Gmag" in line and "G_BPmag" in line and "G_RPmag" in line:
            header_index = index
            break

    if header_index is None:
        raise ValueError(f"No recognised PARSEC table header found in {iso_file.name}.")

    cleaned_lines = [lines[header_index].lstrip("#").strip()]
    cleaned_lines.extend(line for line in lines[header_index + 1:] if line.strip() and not line.lstrip().startswith("#"))

    return Table.read(cleaned_lines, format="ascii.basic", guess=False)


def shifted_parsec_isochrone(iso_name, dist_mod, shift):

    iso_file = ISOCHRONE_DIR / f"{iso_name}.dat"
    if not iso_file.exists():
        print(f"Upload a PARSEC isochrone for your cluster: {iso_file.name}")
        return None

    iso = read_parsec_isochrone(iso_file)

    if "G_BPmag" in iso.columns:
        bp = iso["G_BPmag"]
        rp = iso["G_RPmag"]
    elif "BP_mag" in iso.columns:
        bp = iso["BP_mag"]
        rp = iso["RP_mag"]
    else:
        raise ValueError(f"No recognised isochrone colour columns found in {iso_file.name}.")

    iso["bp_rp"] = bp - rp + shift["colour"]

    if "Gmag" in iso.columns:
        gmag = iso["Gmag"]
    elif "G_mag" in iso.columns:
        gmag = iso["G_mag"]
    else:
        raise ValueError(f"No recognised isochrone magnitude columns found in {iso_file.name}.")

    iso["G_app"] = gmag + dist_mod + shift["mag"]

    mask = (
        (iso["G_app"] <= 21) &
        np.isfinite(iso["bp_rp"]) &
        np.isfinite(iso["G_app"]) &
        (iso["bp_rp"] > -0.5) & (iso["bp_rp"] < 4.0)
    )

    iso2 = iso[mask]

    if "Mini" in iso2.colnames:
        mass_col = "Mini"
    elif "Mass" in iso2.colnames:
        mass_col = "Mass"
    elif "mass" in iso2.colnames:
        mass_col = "mass"
    else:
        raise ValueError(f"No recognised isochrone mass column found in {iso_file.name}.")

    iso2.sort(mass_col)

    col = np.array(iso2["bp_rp"])
    g = np.array(iso2["G_app"])

    jump = np.hypot(np.diff(col), np.diff(g))
    median_jump = np.nanmedian(jump)

    col2 = col.astype(float).copy()
    g2 = g.astype(float).copy()

    if np.isfinite(median_jump) and median_jump > 0:
        thr = 50 * median_jump
        breaks = np.where(jump > thr)[0]
        col2[breaks + 1] = np.nan
        g2[breaks + 1] = np.nan

    return col2, g2


def choose_pmin(file_name, pmin):

    data_file = PYUPMASK_OUTPUT_DIR / f"{file_name}.dat"

    data_all = Table.read(str(data_file), format="ascii")

    data_pmin = data_all[data_all["probs_final"] >= float(pmin)]

    return data_all, data_pmin


# --------------------------------------------------------------------------


def plot_CMD(file_name, pmin):

    tbl_all, tbl_mem = choose_pmin(file_name, pmin)

    colour_all = tbl_all["bp_rp"]

    phot_mag_all = tbl_all["phot_g_mean_mag"]

    colour_mem = tbl_mem["bp_rp"]

    phot_mag_mem = tbl_mem["phot_g_mean_mag"]


    # Calculates distance modulus to shift the isochrone data to match the apparent brightness from GAIA data
    plx = np.asarray(tbl_mem["parallax"], dtype=float)
    plx = plx[np.isfinite(plx) & (plx > 0)]
    if plx.size == 0:
        raise ValueError("No positive parallaxes available for CMD distance estimate.")

    mean_plx = np.median(plx)

    dist = 1000.0/mean_plx

    dist_mod = 5.0*np.log10(dist) - 5.0

    shift = CMD_SHIFT_OVERRIDES.get(file_name.split("_")[0].lower(), {"colour": 0.0, "mag": 0.0})


    # Making the isochrone using the MIST database 
    iso_text = input("Enter the name of your isochrone file(s), separated by commas: ")
    iso_names = parse_isochrone_names(iso_text)
    if not iso_names:
        print("No isochrone files entered.")
        return
#    warnings.filterwarnings("ignore", category=FutureWarning)
#
#    mist = MIST_Isochrone()
#
#    age = float(input("Enter the approx. age of the cluster: "))
#    feh = float(input("Enter the [FE/H] metallicity of the cluster: "))
#
#    iso = mist.isochrone(age, feh)
#    print("Rows: ", len(iso))
#
#    iso_name = input("Enter the name of your isochrone file: ")
#
#    iso_dir = ISOCHRONE_DIR
#    iso_dir.mkdir(parents=True, exist_ok=True)
#
#    iso_file = iso_dir / f"{iso_name}.dat"
#
#    iso.to_csv(
#        iso_file,
#        sep=" ",
#        index=False
#    )


    plt.figure(figsize=(6, 8))

    plt.scatter(colour_all, phot_mag_all,
                s=5, alpha=0.3, label="All stars")

    plt.scatter(colour_mem, phot_mag_mem,
                s=10, alpha=0.8, label=f"Members (P >= {pmin})")

    isochrone_colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plotted_isochrones = 0
    for index, iso_name in enumerate(iso_names):
        shifted_iso = shifted_parsec_isochrone(iso_name, dist_mod, shift)
        if shifted_iso is None:
            continue
        col2, g2 = shifted_iso
        plt.plot(
            col2,
            g2,
            color=isochrone_colours[index % len(isochrone_colours)],
            linewidth=1.7,
            label=f"PARSEC {iso_name}",
        )
        plotted_isochrones += 1

    if plotted_isochrones == 0:
        plt.close()
        print("No valid isochrone files were plotted.")
        return

#    plt.plot(iso2["bp_rp"], iso2["G_app"],        # use this to replace the line above in case you remove the interchangeable line break code
#             color="red",
#             linewidth=1.6,
#             label="PARSEC Isochrone")

    plt.gca().invert_yaxis()
    plt.xlabel(r"$G_{\rm BP} - G_{\rm RP}$")
    plt.ylabel(r"$G_{\rm mag}$")
    plt.title(file_name)
    plt.legend()
    plt.tight_layout()

    FIGURES_CMD_DIR.mkdir(parents=True, exist_ok=True)

    plt.savefig(FIGURES_CMD_DIR / f"{file_name}.png",
                dpi=300, bbox_inches="tight")

    plt.close()

#name = input("Enter the file_name: ")
#plot_CMD(name)

# -------------------------------------------------------------------------------------


def plot_density(file_name, pmin,
                 gridsize=200,
                 cmap='viridis',
                 bandwidth=None,
                 figsize=(7,6)):

    tbl_all, tbl_mem = choose_pmin(file_name, pmin)

    x = np.array(tbl_mem['ra'])
    y = np.array(tbl_mem['dec'])

    xi = np.linspace(x.min(), x.max(), gridsize)
    yi = np.linspace(y.min(), y.max(), gridsize)
    xx, yy = np.meshgrid(xi, yi)

    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=bandwidth)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()]))
    zz = zz.reshape(xx.shape)

    plt.figure(figsize=figsize)
    plt.imshow(zz, origin='lower',
               extent=[x.min(), x.max(), y.min(), y.max()],
               cmap=cmap,
               aspect='auto')
    
    plt.colorbar(label='Stellar Density')
    plt.xlabel('ra')
    plt.ylabel('dec')
    plt.title(file_name)
    plt.tight_layout()

    FIGURES_DENSITY_DIR.mkdir(parents=True, exist_ok=True)

    plt.savefig(FIGURES_DENSITY_DIR / f"{file_name}.png",
                dpi=300, bbox_inches="tight")

    plt.close()

#file_name = input("Enter file name: ")
#plot_density(file_name)


# -----------------------------------------------------------


def plot_stars(file_name, pmin):

    tbl_all, tbl_mem = choose_pmin(file_name, pmin)

    x = np.array(tbl_all["ra"])
    y = np.array(tbl_all["dec"])

    plt.figure(figsize=(7,7))
    plt.scatter(x, y, s=6, alpha=0.25, label="All stars")

    if "probs_final" in tbl_all.colnames if hasattr(tbl_all, "colnames") else "probs_final" in getattr(tbl_all, "columns", []):
        mask = np.array(tbl_all["probs_final"]) >= float(pmin)
        plt.scatter(x[mask], y[mask], s=10, alpha=0.9, label=f"Members (p >= {pmin})")

    plt.gca().invert_xaxis()  # sky convention: RA increases to the left
    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    plt.title(file_name)
    plt.legend()
    plt.tight_layout()
    
    FIGURES_STARS_DIR.mkdir(parents=True, exist_ok=True)

    plt.savefig(FIGURES_STARS_DIR / f"{file_name}.png",
                dpi=300, bbox_inches="tight")

    plt.close()

#file_name = input("Enter file name: ")
#plot_stars(file_name)


# -----------------------------------------------------------------------


def plot_probability_histogram(file_name):

    data_file = PYUPMASK_OUTPUT_DIR / f"{file_name}.dat"
    data_all = Table.read(str(data_file), format="ascii")

    probs_final = np.asarray(data_all["probs_final"], dtype=float)
    probs_final = probs_final[np.isfinite(probs_final)]

    bins = np.arange(0.0, 1.1, 0.1)

    plt.figure(figsize=(7, 5))
    plt.hist(probs_final, bins=bins, edgecolor="black")
    plt.xlabel("probs_final")
    plt.ylabel("Number of stars")
    plt.title(file_name)
    plt.xlim(0, 1)
    plt.xticks(bins)
    plt.tight_layout()

    output_dir = PROJECT_DIR / "figures_probability_histograms"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{file_name}_probability_histogram.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file


# -----------------------------------------------------------------------


def vel_heatmap(file_name, pmin):

    tbl_all, tbl_mem = choose_pmin(file_name, pmin)

    mu = np.sqrt(tbl_mem["pmra"]**2 + tbl_mem["pmdec"]**2)
    vt = 4.74 * mu / tbl_mem["parallax"]

    tbl_mem["vt"] = vt

    log_vt = np.log10(tbl_mem["vt"])

    plt.figure(figsize=(6,6))

    plt.scatter(tbl_mem["ra"], tbl_mem["dec"],
                c=log_vt, s=8, cmap="plasma")
    
    plt.colorbar(label="log10(Tangential velocity) (km/s)")
    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    plt.title(file_name)
    plt.gca().invert_xaxis()

    FIGURES_VEL_DIR.mkdir(parents=True, exist_ok=True)

    plt.savefig(FIGURES_VEL_DIR / f"{file_name}.png",
                dpi=300, bbox_inches="tight")

    plt.close()

#filename = input("Enter file name: ")
#vel_heatmap(filename)


# ----------------------------------------------------------------------------------


def layout3d(file_name, pmin):

    tbl_all, tbl_mem = choose_pmin(file_name, pmin)


    marker_size=5

    ra = np.asarray(tbl_mem["ra"], dtype=float)
    dec = np.asarray(tbl_mem["dec"], dtype=float)
    plx = np.asarray(tbl_mem["parallax"], dtype=float)

    m = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(plx) & (plx > 0)
    ra, dec, plx = ra[m], dec[m], plx[m]

    d = 1000.0 / plx

    d0 = 1000.0 / np.median(plx)

    ra0 = np.median(ra)
    dec0 = np.median(dec)
    d0 = np.median(d)

    ra_r = np.deg2rad(ra)
    dec_r = np.deg2rad(dec)
    ra0_r = np.deg2rad(ra0)
    dec0_r = np.deg2rad(dec0)


    x = d0 * np.cos(dec0_r) * (ra_r - ra0_r)
    y = d0 * (dec_r - dec0_r)
    z = d - d0

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z, s=marker_size, alpha=0.7)

    ax.set_xlabel("x (pc)")
    ax.set_ylabel("y (pc)")
    ax.set_zlabel("z (pc)")
    ax.set_title("3D Cluster Structure")

    max_range = np.array([x.max()-x.min(),
                      y.max()-y.min(),
                      z.max()-z.min()]).max() / 2.0

    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    FIGURES_3D_DIR.mkdir(parents=True, exist_ok=True)

    plt.savefig(FIGURES_3D_DIR / f"{file_name}.png",
                dpi=300, bbox_inches="tight")

    plt.close()

#filename = input("Enter file name: ")
#layout3d(filename)


# ----------------------------------------------------------------------------------


def _latest_metric_file(cluster_name, suffix):

    cluster_dir = METRIC_TRIALS_DIR / cluster_name
    candidates = sorted(cluster_dir.glob(f"{cluster_name}_{suffix}_pmin_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No {suffix} metric file found for {cluster_name} in {cluster_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _mst_data_file(file_name, pmin=None):

    cluster_dir = MST_DIR / file_name

    if pmin is not None:
        candidate = cluster_dir / f"{file_name}_mst_pmin_{str(pmin).replace('.', 'p')}.dat"
        if candidate.exists():
            return candidate

    candidates = sorted(cluster_dir.glob(f"{file_name}_mst_pmin_*.dat"))
    if not candidates:
        raise FileNotFoundError(f"No MST data file found for {file_name} in {cluster_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def plot_MST(file_name, pmin=None):

    mst_file = _mst_data_file(file_name, pmin=pmin)
    mst_table = Table.read(str(mst_file), format="ascii")

    kind = np.asarray(mst_table["kind"]).astype(str)

    node_mask = kind == "node"
    edge_mask = kind == "edge"

    x1_edges = np.asarray(mst_table["x1"][edge_mask], dtype=float)
    y1_edges = np.asarray(mst_table["y1"][edge_mask], dtype=float)
    x2_edges = np.asarray(mst_table["x2"][edge_mask], dtype=float)
    y2_edges = np.asarray(mst_table["y2"][edge_mask], dtype=float)

    plt.figure(figsize=(7, 7))

    for x1, y1, x2, y2 in zip(x1_edges, y1_edges, x2_edges, y2_edges):
        plt.plot([x1, x2], [y1, y2], color="black", linewidth=0.8, alpha=0.7, zorder=1)

    plt.xlabel("Projected x")
    plt.ylabel("Projected y")
    plt.title(f"{file_name} MST")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()

    FIGURES_MST_DIR.mkdir(parents=True, exist_ok=True)

    output_file = FIGURES_MST_DIR / f"{file_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file


def _latest_v_disp_pmin(cluster_name):

    v_file = _latest_metric_file(cluster_name, "v_disp")
    v_metrics = json.loads(v_file.read_text(encoding="utf-8"))
    return float(v_metrics["pmin"])


def tangential_velocity_sample(
    file_name,
    pmin=None,
    relative_to_mean=True,
):

    if pmin is None:
        pmin = _latest_v_disp_pmin(file_name)

    data_file = PYUPMASK_OUTPUT_DIR / f"{file_name}.dat"
    table = Table.read(str(data_file), format="ascii")
    selected = filter_catalog_for_dispersion(table, pmin=pmin)

    median_parallax = float(np.median(selected["parallax"]))
    conversion_factor = KMS_PER_AU_PER_YR / median_parallax

    v_ra = conversion_factor * selected["pmra"]
    v_dec = conversion_factor * selected["pmdec"]

    if relative_to_mean:
        v_ra = v_ra - np.mean(v_ra)
        v_dec = v_dec - np.mean(v_dec)

    v_tan = np.sqrt(v_ra**2 + v_dec**2)
    v_tan = v_tan[np.isfinite(v_tan)]

    if len(v_tan) < 2:
        raise ValueError(f"Not enough velocity samples for {file_name}.")

    return v_tan, float(pmin), len(v_tan)


def plot_tangential_velocity_kdes(
    cluster_names=None,
    pmin=None,
    relative_to_mean=True,
    bandwidth=None,
    combined=False,
    log_x=False,
):

    if cluster_names is None:
        cluster_names = load_q_vdisp_cluster_list()

    cluster_names = list(dict.fromkeys(cluster_names))

    samples = []
    for cluster_name in cluster_names:
        v_tan, cluster_pmin, n_stars = tangential_velocity_sample(
            cluster_name,
            pmin=pmin,
            relative_to_mean=relative_to_mean,
        )
        samples.append(
            {
                "name": cluster_name,
                "pmin": cluster_pmin,
                "n_stars": n_stars,
                "v_tan": v_tan,
            }
        )

    if relative_to_mean:
        xlabel = "Tangential velocity offset from cluster mean (km/s)"
        filename_suffix = "tangential_velocity_offset_kde"
    else:
        xlabel = "Tangential velocity (km/s)"
        filename_suffix = "tangential_velocity_kde"

    FIGURES_VEL_KDE_DIR.mkdir(parents=True, exist_ok=True)

    if combined:
        upper = max(np.percentile(sample["v_tan"], 99.0) for sample in samples)
        if log_x:
            positive_samples = np.concatenate([sample["v_tan"][sample["v_tan"] > 0] for sample in samples])
            lower = max(np.percentile(positive_samples, 1.0), 1e-3)
            x_grid = np.geomspace(lower, upper * 1.05, 400)
        else:
            x_grid = np.linspace(0.0, upper * 1.05, 400)

        plt.figure(figsize=(8, 6))
        cmap = plt.get_cmap("tab10")

        for index, sample in enumerate(samples):
            kde = gaussian_kde(sample["v_tan"], bw_method=bandwidth)
            label = f"{sample['name']} (N={sample['n_stars']}, p>={sample['pmin']:g})"
            plt.plot(
                x_grid,
                kde(x_grid),
                linewidth=1.8,
                color=cmap(index % 10),
                label=label,
            )

        title = "Tangential Velocity Offset Distributions" if relative_to_mean else "Tangential Velocity Distributions"
        filename = f"combined_{filename_suffix}s_logx.png" if log_x else f"combined_{filename_suffix}s.png"
        output_file = FIGURES_VEL_KDE_DIR / filename

        plt.xlabel(xlabel)
        plt.ylabel("Kernel density")
        plt.title(title)
        if log_x:
            plt.xscale("log")
        plt.legend(title="Cluster", loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    output_files = []
    for sample in samples:
        upper = np.percentile(sample["v_tan"], 99.0)
        x_grid = np.linspace(0.0, upper * 1.05, 400)
        kde = gaussian_kde(sample["v_tan"], bw_method=bandwidth)

        plt.figure(figsize=(7, 5))
        plt.plot(x_grid, kde(x_grid), linewidth=2.0)
        plt.xlabel(xlabel)
        plt.ylabel("Kernel density")
        plt.title(f"{sample['name']} Tangential Velocity KDE")
        plt.text(
            0.98,
            0.95,
            f"N={sample['n_stars']}\np>={sample['pmin']:g}",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )
        plt.tight_layout()

        output_file = FIGURES_VEL_KDE_DIR / f"{sample['name']}_{filename_suffix}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        output_files.append(output_file)

    return output_files


def load_q_vdisp_cluster_list():

    if Q_VDISP_CLUSTER_LIST_FILE.exists():
        cluster_names = json.loads(Q_VDISP_CLUSTER_LIST_FILE.read_text(encoding="utf-8"))
        return list(dict.fromkeys(cluster_names))

    return list(DEFAULT_Q_VDISP_CLUSTERS)


def save_q_vdisp_cluster_list(cluster_names):

    FIGURES_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    cluster_names = list(dict.fromkeys(cluster_names))
    Q_VDISP_CLUSTER_LIST_FILE.write_text(json.dumps(cluster_names, indent=2), encoding="utf-8")
    return cluster_names


def plot_q_vs_vdisp(cluster_names=None, include_cluster=None, persist_cluster_list=False):

    if cluster_names is None:
        cluster_names = load_q_vdisp_cluster_list()
    else:
        cluster_names = list(cluster_names)

    if include_cluster and include_cluster not in cluster_names:
        cluster_names.append(include_cluster)

    cluster_names = list(dict.fromkeys(cluster_names))

    if persist_cluster_list:
        cluster_names = save_q_vdisp_cluster_list(cluster_names)

    points = []
    for cluster_name in cluster_names:
        q_file = _latest_metric_file(cluster_name, "q_metrics")
        v_file = _latest_metric_file(cluster_name, "v_disp")

        q_metrics = json.loads(q_file.read_text(encoding="utf-8"))
        v_metrics = json.loads(v_file.read_text(encoding="utf-8"))

        points.append(
            {
                "name": cluster_name,
                "q": q_metrics["q_parameter"],
                "q_err_minus": q_metrics["q_err_minus"],
                "q_err_plus": q_metrics["q_err_plus"],
                "v_disp": v_metrics["sigma_2d_intrinsic_kms"],
                "v_err_minus": v_metrics["sigma_err_minus"],
                "v_err_plus": v_metrics["sigma_err_plus"],
            }
        )

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")

    for index, point in enumerate(points):
        color = cmap(index % 10)
        plt.errorbar(
            point["q"],
            point["v_disp"],
            xerr=np.array([[point["q_err_minus"]], [point["q_err_plus"]]]),
            yerr=np.array([[point["v_err_minus"]], [point["v_err_plus"]]]),
            fmt="o",
            capsize=4,
            markersize=6,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            label=point["name"],
        )

    plt.xlabel("Q parameter")
    plt.ylabel("Intrinsic 2D tangential velocity dispersion (km/s)")
    plt.title("Q Parameter vs Velocity Dispersion")
    plt.legend(title="Cluster", loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()

    FIGURES_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = FIGURES_METRICS_DIR / "q_vs_v_disp.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file
