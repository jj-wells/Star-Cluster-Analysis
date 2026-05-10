import json
from pathlib import Path

from functions.data_collection import get_coordinates, make_coordinates, get_loose_constraints, gaia_query, autoconstraints, write_input, run_pyupmask, has_loose_constraints
from functions.plotting import plot_CMD, plot_density, plot_stars, vel_heatmap, layout3d, plot_MST, plot_q_vs_vdisp, _latest_metric_file
from functions.q_para import projected_catalog_q_metrics, write_metrics_to_file as write_q_metrics_to_file, write_mst_data
from functions.v_disp import tangential_velocity_dispersion, write_metrics_to_file as write_v_disp_metrics_to_file
from scipy.stats import pearsonr
from scipy.stats import spearmanr


def q_v_disp_values(cluster_names):

    q_values = []
    v_disp_values = []

    for cluster_name in cluster_names:
        q_file = _latest_metric_file(cluster_name, "q_metrics")
        v_file = _latest_metric_file(cluster_name, "v_disp")
        q_values.append(json.loads(q_file.read_text(encoding="utf-8"))["q_parameter"])
        v_disp_values.append(json.loads(v_file.read_text(encoding="utf-8"))["sigma_2d_intrinsic_kms"])

    return q_values, v_disp_values


def correlation_stats(r_value, p_value, n_clusters):

    r_value = float(r_value)
    p_value = float(p_value)

    return {
        "r": r_value,
        "r_squared": r_value**2,
        "p_value": p_value,
        "N": n_clusters,
    }


def q_v_disp_correlations(cluster_names):

    q_values, v_disp_values = q_v_disp_values(cluster_names)

    pearson_r_value, pearson_p_value = pearsonr(q_values, v_disp_values)
    spearman_r_value, spearman_p_value = spearmanr(q_values, v_disp_values)

    return {
        "clusters": cluster_names,
        "pearson": correlation_stats(pearson_r_value, pearson_p_value, len(cluster_names)),
        "spearman": correlation_stats(spearman_r_value, spearman_p_value, len(cluster_names)),
    }


# r_app
# onc = 1.0
# ic348 = 0.4
# m67 = 0.5
# pleiades = 3.0
# m3 = 0.4
# m92 = 0.35
# ngc6530 = 0.3
# chamaeleon = 4.0
# taurus = 6.25
# ngc2808 = 0.27
# m4 = 0.43
# ngc188 = 0.25
# ngc752 = 0.5
# ic4651 = 0.2

# chamaeleon coords:
# RA = 180.4
# Dec = -76.7

# Taurus coords:
# RA = 68.5
# Dec = 27.0



intro = """
----------------------------------
*** Cluster - Code ***
----------------------------------

Information:
(1) 2 OPTIONS - Data collection / Analysis - **NOTE: Data collection will continue onto Analysis

(2) For Data Collection - The program will run a query to the GAIA database and then process it through the pyUPMASK clustering algorithm.

(3) For Analysis - The program will start by plotting a Colour Magnitude Diagram (CMD) and map an isochrone on top to verify the data. """
    
print(intro)

dataOrAnalyse = input("Enter Data (D) or Analyse (A) based on the context above [D/A]: ").strip().lower()


if dataOrAnalyse in ("d", "data"):

    ### Get coordinates using object name or manually
    Cluster_name = input("What cluster would you like to look at?: ")
    coord_option = input("Are you using the object name or do you have manual coordinates? [name/manual]:  ").strip().lower()
    apparent_radius = float(input("What is the apparent radius of the cluster?: "))*1.2     # 1 degree = 60 arcminutes
    file_name = input("Name your data file: ")

    if coord_option in ("manual", "m"):
        ra_deg = float(input("Enter the RA of the centre of your target in degrees: "))
        dec_deg = float(input("Enter the Dec of the centre of your target in degrees:"))
        coords = make_coordinates(ra_deg, dec_deg)
    else:
        coords = get_coordinates(Cluster_name)

    ### Wide query 
    loose_constraints = get_loose_constraints(Cluster_name)
    data = gaia_query(coords.ra.deg, coords.dec.deg, loose_constraints, apparent_radius)

    if has_loose_constraints(Cluster_name):
        better_data = data
    else:
        ### Narrows down constraints to allow for a more accurate Gaia query
        tight_constraints = autoconstraints(data)

        ### Second query with tighter constraints
        better_data = gaia_query(coords.ra.deg, coords.dec.deg, tight_constraints, apparent_radius)

    ### Writing data file to pyUPMASK input folder
    input_file = write_input(better_data, file_name)
    #input_file = write_input(data, file_name)


    ### Running pyUPMASK
    run_pyupmask(file_name)
    print("Data collection complete.")

if 'file_name' in locals() or 'file_name' in globals():
    pass
else:
    file_name = input("Name your data file for analysis: ")

print("Starting analysis...")

### Plotting
pmin = float(input("Enter your minimum probability for membership: "))

plot_CMD(file_name, pmin)

plot_density(file_name, pmin)

plot_stars(file_name, pmin)

vel_heatmap(file_name, pmin)

layout3d(file_name, pmin)
print("Plotting complete.")

### Structure and velocity metrics
n_bootstrap_text = input("Enter number of bootstrap trials for Q-parameter and velocity dispersion [200]: ").strip()
n_bootstrap = int(n_bootstrap_text) if n_bootstrap_text else 200
max_q_stars_text = input("Enter maximum parent stars to use for each Q bootstrap [3000]: ").strip()
max_q_stars = int(max_q_stars_text) if max_q_stars_text else 3000

print("Calculating Q-parameter and MST...")
q_metrics = projected_catalog_q_metrics(file_name, pmin=pmin, n_bootstrap=n_bootstrap, max_q_stars=max_q_stars)
q_output_file = write_q_metrics_to_file(q_metrics)

mst_output_file = write_mst_data(q_metrics)
mst_plot_file = plot_MST(file_name, pmin=pmin)
print(f"Q/MST complete. Results written to: {q_output_file}, {mst_output_file}, {mst_plot_file}")

print("Calculating tangential velocity dispersion...")
v_disp_metrics = tangential_velocity_dispersion(file_name, pmin=pmin, n_bootstrap=n_bootstrap)
v_disp_output_file = write_v_disp_metrics_to_file(v_disp_metrics)

print(f"Velocity dispersion complete. Results written to: {v_disp_output_file}")

add_to_metric_plot = input("Add this cluster to the Q-parameter vs velocity-dispersion plot? [y/n]: ").strip().lower()

if add_to_metric_plot in ("y", "yes"):
    q_v_disp_plot = plot_q_vs_vdisp(include_cluster=file_name, persist_cluster_list=True)
    print(f"Q-parameter vs velocity-dispersion plot written to: {q_v_disp_plot}")

correlation_clusters = ["ic348_5", "m3_1", "m67_1", "m92_2", "plei_13", "ic2391_1", "cha_1", "m4_1", "ngc188_1", "ic4651_1"]
full_sample_correlations = q_v_disp_correlations(correlation_clusters)
leave_one_out_correlations = []

for omitted_cluster in correlation_clusters:
    leave_one_out_clusters = [cluster_name for cluster_name in correlation_clusters if cluster_name != omitted_cluster]
    leave_one_out_result = q_v_disp_correlations(leave_one_out_clusters)
    leave_one_out_correlations.append({
        "omitted_cluster": omitted_cluster,
        "clusters": leave_one_out_result["clusters"],
        "pearson": leave_one_out_result["pearson"],
        "spearman": leave_one_out_result["spearman"],
    })

correlation_results = {
    "clusters": correlation_clusters,
    "full_sample": full_sample_correlations,
    "leave_one_out": leave_one_out_correlations,
}

pearson_result = full_sample_correlations["pearson"]
spearman_result = full_sample_correlations["spearman"]

print(f"Pearson correlation for full cluster sample, including cha_1: r = {pearson_result['r']:.6f}, R^2 = {pearson_result['r_squared']:.6f}, p = {pearson_result['p_value']:.6g}, N = {pearson_result['N']}")
print(f"Spearman correlation for full cluster sample, including cha_1: r = {spearman_result['r']:.6f}, R^2 = {spearman_result['r_squared']:.6f}, p = {spearman_result['p_value']:.6g}, N = {spearman_result['N']}")

correlations_output_file = Path(__file__).resolve().parent / "figures_metrics" / "q_vs_v_disp_correlations.json"
correlations_output_file.parent.mkdir(parents=True, exist_ok=True)
correlations_output_file.write_text(json.dumps(correlation_results, indent=2), encoding="utf-8")
print(f"Correlation tests complete. Results written to: {correlations_output_file}")
