import json
import logging
import os
from os.path import abspath, dirname, join
from pathlib import Path

import numpy as np
from hydra.core.hydra_config import HydraConfig


def load_result(
    filename="results.json", dirpath=None, multirun=None, numpy=True
):
    """Load the resulting data from a ``benchflow`` run or multirun.

    If a root path ``dirpath`` is not provided, will try to find the most
    recent run/multirun.

    Parameters
    ----------
    filename : str
        The name of the results file for each run. Defaults to "results.json".
    dirpath : str
        The root path of the ``hydra`` run or multirun. If not provided,
        attempts to find the most recent run/multirun in the default
        directory: ``benchflow/outputs`` or ``benchflow/multirun``,
        respectively.
    multirun : bool
        Whether to look for a multirun. Default ``False``.
    numpy : bool
        Whether to restore lists of metrics to ``numpy`` arrays when loading.
        Default ``True``.

    Returns
    -------
    results : dict
        A dictionary containing the metrics and additional info recorded during
        the run.
    """
    results = {}
    if multirun is None:
        multirun = os.path.isdir(os.path.join(dirpath, "0"))
    if multirun:
        filepaths = _get_filepath(filename, dirpath, multirun)
        for i, fp in enumerate(filepaths):
            dname = os.path.basename(os.path.dirname(fp))
            try:
                with open(fp, "r") as f:
                    res = json.load(f)
                    if (
                        numpy
                    ):  # Convert metrics from lists back to Numpy arrays
                        for key, val in res.get("metrics", {}).items():
                            for j, a in enumerate(val):
                                if type(a) == list:
                                    res["metrics"][key][j] = np.array(a)
                results[int(dname)] = res
            except FileNotFoundError as e:
                logging.warn(f"{e}")
                logging.warn(f"Skipping multirun {dname}.")
                results[int(dname)] = f"File not found: {e}"
    else:
        filepath = _get_filepath(filename, dirpath, multirun)
        with open(filepath, "r") as f:
            results = json.load(f)

            if numpy:  # Convert metrics from lists back to Numpy arrays
                for key, val in results.get("metrics", {}).items():
                    for j, a in enumerate(val):
                        if type(a) == list:
                            results["metrics"][key][j] = np.array(a)
    return results


def _get_filepath(filename, dirpath, multirun):
    """Get the absolute filepath(s) of the requested hydra run.

    If a root path ``dirpath`` is not provided, will try to find the most
    recent run/multirun.

    Parameters
    ----------
    filename : str
        The name of the results file for each run. Defaults to "results.json".
    dirpath : str
        The root path of the ``hydra`` run or multirun. If not provided,
        attempts to find the most recent run/multirun in the default
        directory: ``benchflow/outputs`` or ``benchflow/multirun``,
        respectively.
    multirun : bool
        Whether to look for a multirun.

    Returns
    -------
    filepath(s) : str or [str]
        The full path of the results file (if not a multirun) or a list of
        paths to the results files (if a multirun).
    """
    if dirpath is None:  # Get the latest results by default
        import benchflow

        dirpath = os.path.dirname(os.path.dirname(benchflow.__file__))
        if multirun:
            dirpath = os.path.join(dirpath, "multirun")
            latest_date = [
                d
                for d in sorted(os.listdir(dirpath))
                if os.path.isdir(os.path.join(dirpath, d))
            ][-1]
            dirpath = os.path.join(dirpath, latest_date)
            latest_time = [
                d
                for d in sorted(os.listdir(dirpath))
                if os.path.isdir(os.path.join(dirpath, d))
            ][-1]
            dirpath = os.path.join(dirpath, latest_time)
        else:
            dirpath = os.path.join(dirpath, "outputs")
            latest_date = [
                d
                for d in sorted(os.listdir(dirpath))
                if not os.path.isfile(os.path.join(dirpath, d))
            ][-1]
            dirpath = os.path.join(dirpath, latest_date)
            latest_time = [
                d
                for d in sorted(os.listdir(dirpath))
                if not os.path.isfile(os.path.join(dirpath, d))
            ][-1]
            dirpath = os.path.join(dirpath, latest_time)
    if multirun:
        subdirs = [
            d
            for d in sorted(os.listdir(dirpath))
            if os.path.isdir(os.path.join(dirpath, d)) and d != ".submitit"
        ]
        return [os.path.join(dirpath, d, filename) for d in subdirs]
    else:
        return os.path.join(dirpath, filename)


class ResultEncoder(json.JSONEncoder):
    """An encoder for converting to .json format.

    Handles ``numpy`` arrays by converting them to lists.
    See https://docs.python.org/3/library/json.html
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Exception):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def get_hydra_ouput_path(filename=None):
    """Construct the full file output path from user input or defaults.

    Parameters
    ----------
    filename : str, optional
        A filename to append to the output path (optional).

    Returns
    -------
    output_path : Path
    """
    try:  # If hydra was initialized from command line, use its defaults:
        hydra = HydraConfig.get()
        filepath = hydra.runtime.output_dir
    except ValueError:  # Otherwise use temp directory
        filepath = dirname(dirname(dirname(abspath(__file__))))
        filepath = join(filepath, "temp")
    if filename is not None:
        return Path(filepath).joinpath(filename)
    else:
        return Path(filepath)
