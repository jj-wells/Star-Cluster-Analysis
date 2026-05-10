import json
from pathlib import Path

import numpy as np
from astropy.table import Table
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


PROJECT_DIR = Path(__file__).resolve().parents[1]
PYUPMASK_OUTPUT_DIR = PROJECT_DIR / "pyUPMASK-main" / "output"
METRIC_TRIALS_DIR = PROJECT_DIR / "metric_trials"
MST_DIR = PROJECT_DIR / "MST"


def load_cluster_table(file_name):

    data_file = PYUPMASK_OUTPUT_DIR / f"{file_name}.dat"
    return Table.read(str(data_file), format="ascii")


def select_members(
    table,
    pmin=0.95,
    prob_col="probs_final",
    ra_col="ra",
    dec_col="dec",
):

    members = table[np.asarray(table[prob_col], dtype=float) >= float(pmin)]  # keep only likely members

    ra = np.asarray(members[ra_col], dtype=float)
    dec = np.asarray(members[dec_col], dtype=float)
    mask = np.isfinite(ra) & np.isfinite(dec)  # drop rows that would break the projection
    members = members[mask]

    if len(members) < 3:
        raise ValueError("Not enough member stars after filtering.")

    return members


def load_projected_member_positions(
    file_name,
    pmin=0.95,
    ra_col="ra",
    dec_col="dec",
    prob_col="probs_final",
):

    table = load_cluster_table(file_name)
    members = select_members(
        table,
        pmin=pmin,
        prob_col=prob_col,
        ra_col=ra_col,
        dec_col=dec_col,
    )

    ra_deg = np.asarray(members[ra_col], dtype=float)
    dec_deg = np.asarray(members[dec_col], dtype=float)

    ra0_deg = float(np.mean(ra_deg))
    dec0_deg = float(np.mean(dec_deg))

    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    ra0_rad = np.radians(ra0_deg)
    dec0_rad = np.radians(dec0_deg)

    x = (ra_rad - ra0_rad) * np.cos(dec0_rad)  # local tangent-plane x
    y = dec_rad - dec0_rad  # local tangent-plane y
    positions = np.column_stack((x, y))

    metadata = {
        "file_name": file_name,
        "pmin": float(pmin),
        "projection_center_ra_deg": ra0_deg,
        "projection_center_dec_deg": dec0_deg,
    }

    return positions, metadata, members


def mean_pairwise_separation(positions):

    n_stars = len(positions)
    if n_stars < 2:
        raise ValueError("At least two stars are required.")

    distances = pdist(positions, metric="euclidean")  # all unique star-star separations
    return float(np.mean(distances))


def mst_mean_edge_length(positions):

    n_stars = len(positions)
    if n_stars < 2:
        raise ValueError("At least two stars are required.")

    distance_matrix = squareform(pdist(positions, metric="euclidean"))  # full pairwise distance grid
    mst = minimum_spanning_tree(distance_matrix)
    return float(np.mean(mst.data))


def mst_edge_segments(positions):

    n_stars = len(positions)
    if n_stars < 2:
        raise ValueError("At least two stars are required.")

    distance_matrix = squareform(pdist(positions, metric="euclidean"))
    mst = minimum_spanning_tree(distance_matrix).tocoo()  # easier to save as line segments

    return np.column_stack((mst.row, mst.col))


def cluster_radius(positions):

    centroid = np.mean(positions, axis=0)  # simple geometric centre
    radial_distances = np.sqrt(np.sum((positions - centroid) ** 2, axis=1))
    return float(np.max(radial_distances))


def normalized_mst_mean_edge_length(positions):

    n_stars = len(positions)
    radius = cluster_radius(positions)
    area = np.pi * radius**2
    mean_edge = mst_mean_edge_length(positions)
    normalization = np.sqrt(n_stars * area) / (n_stars - 1)  # standard Q-parameter scaling
    return float(mean_edge / normalization)


def normalized_mean_pairwise_separation(positions):

    radius = cluster_radius(positions)
    mean_separation = mean_pairwise_separation(positions)
    return float(mean_separation / radius)


def cluster_structure_metrics_from_positions(positions):

    mst_mean = mst_mean_edge_length(positions)
    pairwise_mean = mean_pairwise_separation(positions)
    radius = cluster_radius(positions)
    normalized_mst_mean = normalized_mst_mean_edge_length(positions)
    normalized_pairwise_mean = normalized_mean_pairwise_separation(positions)

    return {
        "mst_mean_edge_length": mst_mean,
        "mean_pairwise_separation": pairwise_mean,
        "cluster_radius": radius,
        "normalized_mst_mean_edge_length": normalized_mst_mean,
        "normalized_mean_pairwise_separation": normalized_pairwise_mean,
        "q_parameter": normalized_mst_mean / normalized_pairwise_mean,
        "n_stars": float(len(positions)),
    }


def bootstrap_q_parameter(
    positions,
    n_bootstrap=200,
    subsample_fraction=0.8,
    random_seed=42,
    max_q_stars=3000,
    progress_every=20,
):

    n_stars = len(positions)
    if n_stars < 3:
        raise ValueError("At least three stars are required for Q-parameter bootstrapping.")

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")

    if not 0.0 < subsample_fraction <= 1.0:
        raise ValueError("subsample_fraction must be between 0 and 1.")

    bootstrap_parent_size = n_stars
    if max_q_stars is not None:
        bootstrap_parent_size = min(n_stars, int(max_q_stars))

    subsample_size = max(3, int(subsample_fraction * bootstrap_parent_size))
    rng = np.random.default_rng(random_seed)
    q_bootstrap = np.empty(n_bootstrap, dtype=float)

    for index in range(n_bootstrap):
        if progress_every and (index == 0 or (index + 1) % progress_every == 0):
            print(f"Q bootstrap trial {index + 1}/{n_bootstrap}")
        sample_indices = rng.choice(n_stars, size=subsample_size, replace=False)
        sample_positions = positions[sample_indices]
        q_bootstrap[index] = cluster_structure_metrics_from_positions(sample_positions)["q_parameter"]

    q_ci16, q_ci84 = np.percentile(q_bootstrap, [16, 84])
    q_bootstrap_mean = float(np.mean(q_bootstrap))

    return {
        "q_bootstrap_method": "subsample_without_replacement",
        "q_bootstrap_n": float(n_bootstrap),
        "q_bootstrap_random_seed": float(random_seed),
        "q_bootstrap_parent_size": float(bootstrap_parent_size),
        "q_subsample_fraction": float(subsample_fraction),
        "q_subsample_size": float(subsample_size),
        "q_bootstrap_mean": q_bootstrap_mean,
        "q_bootstrap_std": float(np.std(q_bootstrap, ddof=1)) if n_bootstrap > 1 else 0.0,
        "q_ci16": float(q_ci16),
        "q_ci84": float(q_ci84),
        "q_err_minus": float(abs(q_bootstrap_mean - q_ci16)),
        "q_err_plus": float(abs(q_ci84 - q_bootstrap_mean)),
        "q_uncertainty_reference": "bootstrap_mean",
    }


def projected_catalog_q_metrics(
    file_name,
    pmin=0.95,
    ra_col="ra",
    dec_col="dec",
    prob_col="probs_final",
    n_bootstrap=200,
    subsample_fraction=0.8,
    random_seed=42,
    max_q_stars=3000,
    progress_every=20,
):

    positions, metadata, members = load_projected_member_positions(
        file_name=file_name,
        pmin=pmin,
        ra_col=ra_col,
        dec_col=dec_col,
        prob_col=prob_col,
    )

    results = cluster_structure_metrics_from_positions(positions)  # central Q value from all members
    results.update(
        bootstrap_q_parameter(
            positions=positions,
            n_bootstrap=n_bootstrap,
            subsample_fraction=subsample_fraction,
            random_seed=random_seed,
            max_q_stars=max_q_stars,
            progress_every=progress_every,
        )
    )
    results.update(metadata)
    results["n_members_available"] = float(len(positions))
    results["n_stars_used_for_q"] = float(len(positions))
    results["q_sample_method"] = "all_members"
    results["q_sample_random_seed"] = None
    results["max_q_stars"] = None if max_q_stars is None else float(max_q_stars)
    results["members"] = members
    results["positions"] = positions
    return results


def write_metrics_to_file(metrics, output_path=None):

    if output_path is None:
        output_dir = METRIC_TRIALS_DIR / metrics["file_name"]
        output_path = output_dir / f"{metrics['file_name']}_q_metrics_pmin_{str(metrics['pmin']).replace('.', 'p')}.json"

    serializable = {k: v for k, v in metrics.items() if k not in {"members", "positions"}}  # keep JSON lightweight
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return output_path


def write_mst_data(metrics, output_path=None):

    file_name = metrics["file_name"]
    pmin_label = str(metrics["pmin"]).replace(".", "p")

    if output_path is None:
        output_path = MST_DIR / file_name / f"{file_name}_mst_pmin_{pmin_label}.dat"

    positions = np.asarray(metrics["positions"], dtype=float)
    edges = mst_edge_segments(positions)

    node_count = len(positions)
    edge_count = len(edges)
    total_rows = node_count + edge_count

    kind = np.empty(total_rows, dtype=object)
    x1 = np.full(total_rows, np.nan, dtype=float)
    y1 = np.full(total_rows, np.nan, dtype=float)
    x2 = np.full(total_rows, np.nan, dtype=float)
    y2 = np.full(total_rows, np.nan, dtype=float)

    kind[:node_count] = "node"  # retained so the file still records the original vertices
    x1[:node_count] = positions[:, 0]
    y1[:node_count] = positions[:, 1]

    if edge_count:
        edge_start = node_count
        edge_end = node_count + edge_count
        kind[edge_start:edge_end] = "edge"  # each row stores one MST segment
        x1[edge_start:edge_end] = positions[edges[:, 0], 0]
        y1[edge_start:edge_end] = positions[edges[:, 0], 1]
        x2[edge_start:edge_end] = positions[edges[:, 1], 0]
        y2[edge_start:edge_end] = positions[edges[:, 1], 1]

    mst_table = Table(
        {
            "kind": kind,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mst_table.write(output_path, format="ascii.basic", overwrite=True)
    return output_path


if __name__ == "__main__":

    file_name = input("Enter file name: ").strip()
    pmin = float(input("Enter minimum membership probability: ").strip())
    n_bootstrap_text = input("Enter number of bootstrap trials [200]: ").strip()
    n_bootstrap = int(n_bootstrap_text) if n_bootstrap_text else 200
    max_q_stars_text = input("Enter maximum parent stars to use for each Q bootstrap [3000]: ").strip()
    max_q_stars = int(max_q_stars_text) if max_q_stars_text else 3000

    results = projected_catalog_q_metrics(
        file_name,
        pmin=pmin,
        n_bootstrap=n_bootstrap,
        max_q_stars=max_q_stars,
    )
    output_file = write_metrics_to_file(results)
    mst_output_file = write_mst_data(results)

    print(f"Members available: {int(results['n_members_available'])}")
    print(f"Stars used for Q-parameter: {int(results['n_stars'])}")
    print(f"Stars used per Q bootstrap trial: {int(results['q_subsample_size'])}")
    print(f"Mean MST edge length: {results['mst_mean_edge_length']:.6f}")
    print(f"Mean pairwise separation: {results['mean_pairwise_separation']:.6f}")
    print(f"Q parameter: {results['q_parameter']:.6f}")
    print(f"Q uncertainty: -{results['q_err_minus']:.6f} / +{results['q_err_plus']:.6f}")
    print(
        "Projection center (deg): "
        f"RA={results['projection_center_ra_deg']:.6f}, "
        f"Dec={results['projection_center_dec_deg']:.6f}"
    )
    print(f"Results written to: {output_file}")
    print(f"MST data written to: {mst_output_file}")
