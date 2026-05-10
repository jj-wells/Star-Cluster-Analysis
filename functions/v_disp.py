import json
from pathlib import Path

import numpy as np
from astropy.table import Table


PROJECT_DIR = Path(__file__).resolve().parents[1]
PYUPMASK_OUTPUT_DIR = PROJECT_DIR / "pyUPMASK-main" / "output"
METRIC_TRIALS_DIR = PROJECT_DIR / "metric_trials"
KMS_PER_AU_PER_YR = 4.74047


def load_cluster_table(file_name):

    data_file = PYUPMASK_OUTPUT_DIR / f"{file_name}.dat"
    return Table.read(str(data_file), format="ascii")


def filter_catalog_for_dispersion(
    table,
    prob_col="probs_final",
    pmin=0.95,
    pmra_col="pmra",
    pmdec_col="pmdec",
    pmra_error_col="pmra_error",
    pmdec_error_col="pmdec_error",
    parallax_col="parallax",
    parallax_error_col="parallax_error",
    ruwe_col="ruwe",
    max_pmra_error_masyr=0.5,
    max_pmdec_error_masyr=0.5,
    max_parallax_error_mas=0.5,
    max_ruwe=1.4,
):

    probabilities = np.asarray(table[prob_col], dtype=float)
    pmra = np.asarray(table[pmra_col], dtype=float)
    pmdec = np.asarray(table[pmdec_col], dtype=float)
    pmra_error = np.asarray(table[pmra_error_col], dtype=float)
    pmdec_error = np.asarray(table[pmdec_error_col], dtype=float)
    parallax = np.asarray(table[parallax_col], dtype=float)
    parallax_error = np.asarray(table[parallax_error_col], dtype=float)
    ruwe = np.asarray(table[ruwe_col], dtype=float)

    keep_mask = (  # membership and basic astrometric quality cuts
        (probabilities >= pmin)
        & np.isfinite(pmra)
        & np.isfinite(pmdec)
        & np.isfinite(pmra_error)
        & np.isfinite(pmdec_error)
        & np.isfinite(parallax)
        & np.isfinite(parallax_error)
        & np.isfinite(ruwe)
        & (parallax > 0.0)
        & (pmra_error <= max_pmra_error_masyr)
        & (pmdec_error <= max_pmdec_error_masyr)
        & (parallax_error <= max_parallax_error_mas)
        & (ruwe <= max_ruwe)
    )

    if np.count_nonzero(keep_mask) < 2:
        raise ValueError("Fewer than two stars satisfy the probability and quality cuts.")

    return {
        "pmra": pmra[keep_mask],
        "pmdec": pmdec[keep_mask],
        "pmra_error": pmra_error[keep_mask],
        "pmdec_error": pmdec_error[keep_mask],
        "parallax": parallax[keep_mask],
        "parallax_error": parallax_error[keep_mask],
        "ruwe": ruwe[keep_mask],
    }


def tangential_velocity_dispersion_from_arrays(
    pmra,
    pmdec,
    pmra_error,
    pmdec_error,
    parallax,
    parallax_error,
    ruwe,
    pmin=0.95,
    parallax_mode="median_cluster",
    max_pmra_error_masyr=0.5,
    max_pmdec_error_masyr=0.5,
    max_parallax_error_mas=0.5,
    max_ruwe=1.4,
):

    mean_parallax = float(np.mean(parallax))
    median_parallax = float(np.median(parallax))

    if parallax_mode == "individual":
        parallax_for_velocity = parallax  # use each star's own distance
    elif parallax_mode == "median_cluster":
        parallax_for_velocity = np.full_like(parallax, median_parallax)  # default: one cluster distance
    elif parallax_mode == "mean_cluster":
        parallax_for_velocity = np.full_like(parallax, mean_parallax)
    else:
        raise ValueError("parallax_mode must be 'individual', 'median_cluster', or 'mean_cluster'.")

    var_pmra_obs = np.var(pmra, ddof=1)
    var_pmdec_obs = np.var(pmdec, ddof=1)
    mean_err_var_pmra = float(np.mean(pmra_error**2))
    mean_err_var_pmdec = float(np.mean(pmdec_error**2))

    var_pmra_intrinsic = max(0.0, var_pmra_obs - mean_err_var_pmra)  # subtract measurement noise floor
    var_pmdec_intrinsic = max(0.0, var_pmdec_obs - mean_err_var_pmdec)

    sigma_pmra_intrinsic = float(np.sqrt(var_pmra_intrinsic))
    sigma_pmdec_intrinsic = float(np.sqrt(var_pmdec_intrinsic))
    sigma_pm_2d_intrinsic = float(np.sqrt(var_pmra_intrinsic + var_pmdec_intrinsic))

    conversion_factor = KMS_PER_AU_PER_YR / parallax_for_velocity  # mas/yr to km/s

    v_ra = conversion_factor * pmra
    v_dec = conversion_factor * pmdec
    v_tan = np.sqrt(v_ra**2 + v_dec**2)

    sigma_err_vra = conversion_factor * pmra_error
    sigma_err_vdec = conversion_factor * pmdec_error

    var_vra_obs = np.var(v_ra, ddof=1)
    var_vdec_obs = np.var(v_dec, ddof=1)

    cluster_parallax_for_conversion = median_parallax if parallax_mode == "median_cluster" else mean_parallax
    pm_to_kms = KMS_PER_AU_PER_YR / cluster_parallax_for_conversion

    sigma_vra_intrinsic = float(sigma_pmra_intrinsic * pm_to_kms)
    sigma_vdec_intrinsic = float(sigma_pmdec_intrinsic * pm_to_kms)
    sigma_2d_intrinsic = float(sigma_pm_2d_intrinsic * pm_to_kms)

    return {
        "file_name": "",
        "n_stars": float(len(v_ra)),
        "pmin": float(pmin),
        "parallax_mode": parallax_mode,
        "max_pmra_error_masyr": float(max_pmra_error_masyr),
        "max_pmdec_error_masyr": float(max_pmdec_error_masyr),
        "max_parallax_error_mas": float(max_parallax_error_mas),
        "max_ruwe": float(max_ruwe),
        "mean_parallax_mas": mean_parallax,
        "median_parallax_mas": median_parallax,
        "mean_parallax_error_mas": float(np.mean(parallax_error)),
        "mean_ruwe": float(np.mean(ruwe)),
        "sigma_pmra_masyr": float(np.sqrt(var_pmra_obs)),
        "sigma_pmdec_masyr": float(np.sqrt(var_pmdec_obs)),
        "sigma_pm_2d_masyr": float(np.sqrt(var_pmra_obs + var_pmdec_obs)),
        "mean_pmra_error_masyr": float(np.mean(pmra_error)),
        "mean_pmdec_error_masyr": float(np.mean(pmdec_error)),
        "sigma_pmra_intrinsic_masyr": sigma_pmra_intrinsic,
        "sigma_pmdec_intrinsic_masyr": sigma_pmdec_intrinsic,
        "sigma_pm_2d_intrinsic_masyr": sigma_pm_2d_intrinsic,
        "mean_vra_kms": float(np.mean(v_ra)),
        "mean_vdec_kms": float(np.mean(v_dec)),
        "mean_vtan_kms": float(np.mean(v_tan)),
        "sigma_vra_kms": float(np.sqrt(var_vra_obs)),
        "sigma_vdec_kms": float(np.sqrt(var_vdec_obs)),
        "sigma_vtan_kms": float(np.std(v_tan, ddof=1)),
        "sigma_2d_kms": float(np.sqrt(var_vra_obs + var_vdec_obs)),
        "mean_vra_error_kms": float(np.mean(sigma_err_vra)),
        "mean_vdec_error_kms": float(np.mean(sigma_err_vdec)),
        "sigma_vra_intrinsic_kms": sigma_vra_intrinsic,
        "sigma_vdec_intrinsic_kms": sigma_vdec_intrinsic,
        "sigma_2d_intrinsic_kms": sigma_2d_intrinsic,
    }


def bootstrap_tangential_velocity_dispersion(
    selected,
    sigma_2d_intrinsic_kms,
    pmin=0.95,
    parallax_mode="median_cluster",
    max_pmra_error_masyr=0.5,
    max_pmdec_error_masyr=0.5,
    max_parallax_error_mas=0.5,
    max_ruwe=1.4,
    n_bootstrap=200,
    random_seed=42,
):

    n_stars = len(selected["pmra"])
    if n_stars < 2:
        raise ValueError("At least two stars are required for dispersion bootstrapping.")

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")

    rng = np.random.default_rng(random_seed)
    sigma_bootstrap = np.empty(n_bootstrap, dtype=float)

    for index in range(n_bootstrap):
        sample_indices = rng.integers(0, n_stars, size=n_stars)  # sample with replacement
        sample_metrics = tangential_velocity_dispersion_from_arrays(
            pmra=selected["pmra"][sample_indices],
            pmdec=selected["pmdec"][sample_indices],
            pmra_error=selected["pmra_error"][sample_indices],
            pmdec_error=selected["pmdec_error"][sample_indices],
            parallax=selected["parallax"][sample_indices],
            parallax_error=selected["parallax_error"][sample_indices],
            ruwe=selected["ruwe"][sample_indices],
            pmin=pmin,
            parallax_mode=parallax_mode,
            max_pmra_error_masyr=max_pmra_error_masyr,
            max_pmdec_error_masyr=max_pmdec_error_masyr,
            max_parallax_error_mas=max_parallax_error_mas,
            max_ruwe=max_ruwe,
        )
        sigma_bootstrap[index] = sample_metrics["sigma_2d_intrinsic_kms"]

    sigma_ci16, sigma_ci84 = np.percentile(sigma_bootstrap, [16, 84])

    return {
        "sigma_bootstrap_method": "resample_with_replacement",
        "sigma_bootstrap_n": float(n_bootstrap),
        "sigma_bootstrap_random_seed": float(random_seed),
        "sigma_bootstrap_mean": float(np.mean(sigma_bootstrap)),
        "sigma_bootstrap_std": float(np.std(sigma_bootstrap, ddof=1)) if n_bootstrap > 1 else 0.0,
        "sigma_ci16": float(sigma_ci16),
        "sigma_ci84": float(sigma_ci84),
        "sigma_err_minus": float(abs(sigma_2d_intrinsic_kms - sigma_ci16)),
        "sigma_err_plus": float(abs(sigma_ci84 - sigma_2d_intrinsic_kms)),
    }


def tangential_velocity_dispersion(
    file_name,
    prob_col="probs_final",
    pmin=0.95,
    pmra_col="pmra",
    pmdec_col="pmdec",
    pmra_error_col="pmra_error",
    pmdec_error_col="pmdec_error",
    parallax_col="parallax",
    parallax_error_col="parallax_error",
    ruwe_col="ruwe",
    parallax_mode="median_cluster",
    max_pmra_error_masyr=0.5,
    max_pmdec_error_masyr=0.5,
    max_parallax_error_mas=0.5,
    max_ruwe=1.4,
    n_bootstrap=200,
    random_seed=42,
):

    table = load_cluster_table(file_name)
    selected = filter_catalog_for_dispersion(
        table=table,
        prob_col=prob_col,
        pmin=pmin,
        pmra_col=pmra_col,
        pmdec_col=pmdec_col,
        pmra_error_col=pmra_error_col,
        pmdec_error_col=pmdec_error_col,
        parallax_col=parallax_col,
        parallax_error_col=parallax_error_col,
        ruwe_col=ruwe_col,
        max_pmra_error_masyr=max_pmra_error_masyr,
        max_pmdec_error_masyr=max_pmdec_error_masyr,
        max_parallax_error_mas=max_parallax_error_mas,
        max_ruwe=max_ruwe,
    )

    metrics = tangential_velocity_dispersion_from_arrays(  # base estimate from the selected members
        pmra=selected["pmra"],
        pmdec=selected["pmdec"],
        pmra_error=selected["pmra_error"],
        pmdec_error=selected["pmdec_error"],
        parallax=selected["parallax"],
        parallax_error=selected["parallax_error"],
        ruwe=selected["ruwe"],
        pmin=pmin,
        parallax_mode=parallax_mode,
        max_pmra_error_masyr=max_pmra_error_masyr,
        max_pmdec_error_masyr=max_pmdec_error_masyr,
        max_parallax_error_mas=max_parallax_error_mas,
        max_ruwe=max_ruwe,
    )
    metrics.update(
        bootstrap_tangential_velocity_dispersion(
            selected=selected,
            sigma_2d_intrinsic_kms=metrics["sigma_2d_intrinsic_kms"],
            pmin=pmin,
            parallax_mode=parallax_mode,
            max_pmra_error_masyr=max_pmra_error_masyr,
            max_pmdec_error_masyr=max_pmdec_error_masyr,
            max_parallax_error_mas=max_parallax_error_mas,
            max_ruwe=max_ruwe,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
    )
    metrics["file_name"] = file_name
    return metrics


def write_metrics_to_file(metrics, output_path=None):

    if output_path is None:
        output_dir = METRIC_TRIALS_DIR / metrics["file_name"]
        output_path = output_dir / f"{metrics['file_name']}_v_disp_pmin_{str(metrics['pmin']).replace('.', 'p')}.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":

    file_name = input("Enter file name: ").strip()
    pmin = float(input("Enter minimum membership probability: ").strip())
    n_bootstrap_text = input("Enter number of bootstrap trials [200]: ").strip()
    n_bootstrap = int(n_bootstrap_text) if n_bootstrap_text else 200

    metrics = tangential_velocity_dispersion(file_name, pmin=pmin, n_bootstrap=n_bootstrap)
    output_file = write_metrics_to_file(metrics)

    print(f"Number of stars: {int(metrics['n_stars'])}")
    print(f"Median parallax (mas): {metrics['median_parallax_mas']:.6f}")
    print(f"Intrinsic 2D PM dispersion (mas/yr): {metrics['sigma_pm_2d_intrinsic_masyr']:.6f}")
    print(f"Intrinsic 2D tangential velocity dispersion (km/s): {metrics['sigma_2d_intrinsic_kms']:.6f}")
    print(f"Dispersion uncertainty: -{metrics['sigma_err_minus']:.6f} / +{metrics['sigma_err_plus']:.6f}")
    print(f"Results written to: {output_file}")
