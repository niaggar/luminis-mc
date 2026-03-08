# plots/loaders.py
from pathlib import Path
from luminis_mc import ResultsLoader 
import os

BASE_RESULTS = Path("/Users/niaggar/Documents/Thesis/Progress")   # or read from env var / config file


def load_run(run_dir: str | Path):
    """Return an open ResultsLoader for a single experiment run."""
    return ResultsLoader(BASE_RESULTS / run_dir)

def load_sweep(folder_path: str) -> dict:
    """
    Load all runs from a sweep directory into a dict of ResultsLoader instances.
    """
    sweep_path = BASE_RESULTS / folder_path
    # Validate the input path
    if not sweep_path:
        raise ValueError("The sweep path cannot be empty.")
    
    # Get all the folder names in sweep_path / runs
    runs_path = f"{sweep_path}/runs"
    try:
        run_folders = [f for f in os.listdir(runs_path) if os.path.isdir(os.path.join(runs_path, f))]
    except FileNotFoundError:
        raise FileNotFoundError(f"The specified path '{runs_path}' does not exist.")
    
    data = {}
    for run_folder in run_folders:
        run_path = os.path.join(runs_path, run_folder)
        try:
            loader = ResultsLoader(run_path)
            data[run_folder] = loader
        except Exception as e:
            print(f"Error loading data from {run_path}: {e}")

    return data
