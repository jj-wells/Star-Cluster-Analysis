# Cluster code

This code relies on the following packages being installed:
- python 3.12
- matplotlib
- numpy
- jupyterlab
- scipy
- pandas
- astropy
- astroquery
- requests
- scikit-learn
- numba

The recommended setup is to create a conda environment from the included file:

```bash
conda env create -f environment.yml
conda activate clusters
```

Alternatively, install the Python packages with:

```bash
pip install -r requirements.txt
```

It also relies on the pyUPMASK software that has been downloaded, this is a machine learning algorithm that builds off the original UPMASK package from R. The README file in the pyUPMASK folder explains the function as well as the params.ini file.

## Running

For now, due to the code being run in a separate anaconda environment, the code must be ran from the main.py file through the terminal, not just the regular run code button. 

There is some prerequisite information required by the Data Collection section of the code which is printed before the user is prompted for it. However, the program does give the option to the user to go straight to analysis if they had run the data collection earlier, for now this is rather strict in what is possible, it will just print the 5 set plots into their respective figure folders.

When the Data Collection section writes a new pyUPMASK input file, it automatically deletes any existing `.dat` files in `pyUPMASK-main/input/`. This is intentional so pyUPMASK only processes the current cluster file, but users should move or copy any input files they want to keep before running a new collection.

The bundled pyUPMASK code reads its `params.ini`, `input`, and `output` folders relative to the `pyUPMASK-main` folder. The main pipeline also runs pyUPMASK from that folder.

For Pleiades and IC348, the pipeline uses hard-coded Gaia query constraints and only runs one Gaia query. Other clusters use a broad first query, automatic constraints from that result, and then a second tighter query.

The CMD plotting stage expects a PARSEC isochrone file to exist in the `isochrones/` folder. If the named file is missing, the program will print `Upload a PARSEC isochrone for your cluster`.

The Q-parameter calculation uses SciPy distance and minimum-spanning-tree tools. The central Q value is calculated using all members above the chosen probability threshold. For dense clusters, the bootstrap uncertainty uses capped random subsamples, defaulting to a maximum parent sample of 3000 stars, so globular clusters remain tractable. The metric JSON records the full member count and the bootstrap sample sizes.
