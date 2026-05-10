from astroquery.utils.tap.core import TapPlus
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from pathlib import Path
import numpy as np
import subprocess
import requests
import sys
import os

PROJECT_DIR = Path(__file__).resolve().parents[1]
PYUPMASK_DIR = PROJECT_DIR / "pyUPMASK-main"

CLUSTER_LOOSE_CONSTRAINTS = {
    "pleiades": {"parallax": [5.0, 10.0], "pmra": [-40.0, 40.0], "pmdec": [-50.0, 20.0]},
    "ic348": {"parallax": [0.0, 2.0], "pmra": [-3.0, 4.0], "pmdec": [-5.0, 1.0]},
    "ic2391": {"parallax": [5.5, 8.0], "pmra": [-40.0, 40.0], "pmdec": [-50.0, 20.0]},
    "chamaeleon": {"parallax": [4.9, 5.5], "pmra": [-40.0, 40.0], "pmdec": [-50.0, 20.0]},
    "taurus": {"parallax": [6.0, 8.0], "pmra": [-40.0, 40.0], "pmdec": [-50.0, 20.0]},
    "ngc2808": {"parallax": [-0.2, 0.4], "pmra": [-40.0, 40.0], "pmdec": [-50.0, 20.0]},
    "m4": {"parallax": [0.0, 1.0], "pmra": [-20.0, -5.0], "pmdec": [-25.0, -10.0]},
    "ngc188": {"parallax": [0.3, 0.9], "pmra": [-40.0, 40.0], "pmdec": [-50.0, 20.0]},
    "ngc752": {"parallax": [1.5, 2.5], "pmra": [-40.0, 40.0], "pmdec": [-50.0, 20.0]},
    "ic4651": {"parallax": [0.5, 1.5], "pmra": [-40.0, 40.0], "pmdec": [-50.0, 20.0]}
}

CLUSTER_TIGHT_CONSTRAINTS = {}


# ---------------------------------------------------------

def has_loose_constraints(cluster_name):
    cluster_key = cluster_name.strip().lower()
    return cluster_key in CLUSTER_LOOSE_CONSTRAINTS

# ---------------------------------------------------------

def log_in():

    logged_in = input("Would you like to log into your Gaia Archive Account? [y/n]: ").strip().lower()

    if logged_in in ("y", "yes"):
        return True
    elif logged_in in ("no", "n"):
        return False
    else:
        print("Invalid input, type 'Yes' or 'No'.")
    

# ---------------------------------------------------------

def get_coordinates(cluster_name):
    coords = SkyCoord.from_name(cluster_name)
    return coords

# ---------------------------------------------------------

def make_coordinates(ra_deg, dec_deg):
    coords = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg", frame="icrs")
    return coords

# ---------------------------------------------------------

def get_loose_constraints(cluster_name):

    default_constraints = {"parallax": [-1.0, 25.0], "pmra": [-40.0, 40.0], "pmdec": [-50.0, 20.0]}
    return CLUSTER_LOOSE_CONSTRAINTS.get(cluster_name.strip().lower(), default_constraints)

# ---------------------------------------------------------

def get_tight_constraints(cluster_name, data):

    cluster_key = cluster_name.strip().lower()
    if cluster_key in CLUSTER_TIGHT_CONSTRAINTS:
        return CLUSTER_TIGHT_CONSTRAINTS[cluster_key]

    return autoconstraints(data)

# ---------------------------------------------------------

def gaia_query(ra, dec, constraints, radius):

    Gaia.TAP_SERVER = "https://gea.esac.esa.int/tap-server/tap"
    print(f"Using Gaia.TAP_SERVER = {Gaia.TAP_SERVER}")


    parallax_min = constraints["parallax"][0]
    parallax_max = constraints["parallax"][1]
    pmra_min = constraints["pmra"][0]
    pmra_max = constraints["pmra"][1]
    pmdec_min = constraints["pmdec"][0]
    pmdec_max = constraints["pmdec"][1]

    
    query_base = """
    SELECT
        source_id,
        ra, dec,
        parallax, parallax_error,
        pmra, pmra_error, pmdec, pmdec_error,
        ruwe,
        phot_g_mean_mag,
        phot_bp_mean_mag,
        phot_rp_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius})
    )
    AND parallax BETWEEN {parallax_min} AND {parallax_max}
    AND pmra BETWEEN {pmra_min} AND {pmra_max}
    AND pmdec BETWEEN {pmdec_min} AND {pmdec_max}
    AND ruwe < 1.4
    """

    query = query_base.format(ra=ra, dec=dec, radius=radius, parallax_min=parallax_min, parallax_max=parallax_max, pmra_min=pmra_min, pmra_max=pmra_max, pmdec_min=pmdec_min, pmdec_max=pmdec_max)

    

    use_login = log_in()
    missing_credentials = use_login and (not os.environ.get("GAIA_USER") or not os.environ.get("GAIA_PASS"))

    if missing_credentials:
        print("Missing Credentials, falling back to anonymous mode")

    if use_login and not missing_credentials:
        
        try:
            Gaia.login(
                user=os.environ["GAIA_USER"],
                password=os.environ["GAIA_PASS"]
            )
            print("Login successful.")

            try:
                print("Launching authorised ASYNC query...")
                job = Gaia.launch_job_async(query)
                results = job.get_results()
                print("Query successful.")
                return results
    
            except Exception as e:
                print("Query failed:", e)

        except Exception:
            pass

        
    if (not use_login) or missing_credentials:
        try:
            print("Launching anonymous ASYNC query...")
            job = Gaia.launch_job_async(query)
            results = job.get_results()
            return results
        
        except Exception as e:
            print("Query failed:", e)



#    job = Gaia.launch_job_async(query)
#    job.wait_for_job_end()
#    return job.get_results()

# -----------------------------------------------------------------

def autoconstraints(data, nsigma=1):

    if data is None:
        raise ValueError("Autoconstraints function received no data, Query Failed.")
    
    else:
    
        constraints = {}

        min_half_width = {"parallax": 0.2, "pmra": 0.5, "pmdec": 0.5}

        for col in ["parallax", "pmra", "pmdec"]:
            
            median = np.nanmedian(data[col])
            mad = np.nanmedian(np.abs(data[col] - median))
            sigma = 1.48 * mad
            half_width = max(nsigma * sigma, min_half_width[col])

            constraints[col] = [
                median - half_width,
                median + half_width
            ]

        constraints["parallax"][0] = max(constraints["parallax"][0], -1.0)
        
        return constraints

# ------------------------------------------------------------------

def write_input(data, file_name):

    bp = data['phot_bp_mean_mag']
    rp = data['phot_rp_mean_mag']
    data['bp_rp'] = bp - rp

    input_dir = PYUPMASK_DIR / "input"

    input_dir.mkdir(exist_ok=True)

    for file in input_dir.glob("*.dat"):
        file.unlink()

    input_file = input_dir / f"{file_name}.dat"

    data.write(
        input_file,
        format="ascii.basic",
        overwrite=True
    )

    return input_file

# ------------------------------------------------------------------

def run_pyupmask(input_filename: str):

    pyupmask_dir = PYUPMASK_DIR.resolve()
    script = pyupmask_dir / "pyUPMASK.py"

    if not script.exists():
        raise FileNotFoundError(f"Can't find {script}")

    cmd = [sys.executable, str(script), input_filename]

    completed = subprocess.run(
        cmd,
        cwd=str(pyupmask_dir),   
        check=True               
    )
    return completed
