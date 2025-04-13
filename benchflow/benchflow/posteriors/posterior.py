import copy
import json
import logging
import traceback
from abc import abstractmethod
from os import makedirs
from os.path import abspath, dirname, join
from pathlib import Path
from typing import Optional

import dill
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from benchflow.utilities import ResultEncoder
from benchflow.utilities.cfg import cfg_to_metrics


class Posterior:
    """Abstract base class for ``benchflow`` posteriors.

    Attributes
    ----------
    history : [Posterior]
        A list of ``Posterior`` objects for each iteration of an algorithm.
    metrics : dict
        A dictionary containing the computed metrics. Keys are metrics names
        (e.g.  "c2st") and values are lists of computed metrics (one for each
        ``Posterior`` object in ``self.history``, and one for the final
        ``Posterior``)
    cfg : omegaconf.DictConfig
        The original ``hydra`` config describing the run.
    task : benchflow.task.Task
        The target ``Task`` for inference.
    """

    @abstractmethod
    def __init__(self, cfg, task, results: Optional[dict] = None):
        """Initialize a ``Posterior``."""
        self.history = []
        self.metrics = {}
        self.results = {}
        if results is not None:
            is_json_dumpable_dict(results)
            self.results = results
        self.cfg = cfg
        self.task = task

    def sample(self, n_samples):
        """Draw samples from the inferred posterior.

        Parameters
        ----------
        n_samples : int
            The number of samples to return.

        Returns
        -------
        samples : np.ndarray
            The posterior samples, shape ``(n_samples, D)`` where ``D`` is the
            task dimension.
        """
        raise NotImplementedError

    def get_samples(self, n_samples=10000):
        """Produce samples from the inferred posterior.

        Draws the samples if the ``Posterior`` provides a method
        ``sample(n_samples)`` to do so (e.g. in the case of a generative
        variational posterior). Otherwise, selects frozen samples if the
        ``Posterior`` provides an attribute ``posterior_samples : np.ndarray``
        (e.g. for MCMC methods). If neither the method nor attribute are
        present, ``benchflow`` will record the error and attempt to continue
        execution.

        Parameters
        ----------
        n_samples : int, optional
            The number of samples requested. Default 10,000.

        Returns
        -------
        samples : np.ndarray or Exception
            If samples were successfully collected, returns an array of shape
            ``(n, D)`` where ``D`` is the task dimension. ``n`` may be less
            than ``n_samples`` if the samples are drawn from stored posterior
            samples, in which case ``benchflow`` will log a warning and attempt
            to continue. If the samples were not successfully collected,
            returns the corresponding Exception.
        """
        try:
            if hasattr(self, "sample"):
                return self.sample(n_samples)
            elif hasattr(self, "posterior_samples"):
                if n_samples > self.posterior_samples.shape[0]:
                    logging.warn(
                        f"Requested {n_samples} posterior samples, but Posterior only has {self.posterior_samples.shape[0]}. Attempting to continue with fewer samples."
                    )
                return self.posterior_samples[:n_samples]
            else:
                raise AttributeError(
                    "Posterior has no existing samples and no posterior to"
                    + " sample from."
                )
        except Exception as e:
            logging.warn(
                traceback.format_exc() + "\nAttempting to continue...\n"
            )
            return e

    def save_results(
        self,
        filename="results",
        filepath=None,
        save_function_log=False,
        save_objects=False,
        save_all=True,
    ):
        """Save the resulting metrics / ``Posterior`` object(s) to disk.

        Parameters
        ----------
        filename : str, optional
            The filename (without extension) for saving results. Default
            "results".
        filepath : str, optional
            The root filepath for saving results. If ``None``, will default to
            the current ``hydra.runtime.output_dir`` if available, and
            ``benchflow/temp`` otherwise.
        save_function_log : str or bool, optional
            If "timings", then save only wall-clock timings of log-joint
            evaluations. Otherwise, save whole function log if ``True``, or
            don't save if ``False``. Default "timings".
        save_objects : bool, optional
            Whether to save the ``Posterior`` object(s) as ``.pkl`` binaries.
        save_all : bool, optional
            If ``save_objects`` is ``True``: whether to save ``Posterior``
            objects from all iterations (if ``True``), or only the final
            ``Posterior`` (if ``False``).
        """
        name = str(Path(filename).with_suffix(""))
        filename = name + ".json"
        fullpath = self._get_output_path(filepath, filename)
        results = self.results
        results["config"] = OmegaConf.to_container(self.cfg)
        results["metrics"] = self.metrics

        if save_function_log == "timings":
            results["timings"] = {
                "t0": self.task._log["t0"],
                "t_after": self.task._log["t_after"],
                "t_before": self.task._log["t_before"],
            }
        elif save_function_log:
            results["function_log"] = self.task._log

        logging.info(f"Saving results to {fullpath}")
        makedirs(dirname(fullpath), exist_ok=True)
        with open(fullpath, "w") as f:
            json.dump(results, f, indent=4, cls=ResultEncoder)
        if self.cfg.get("print_results"):
            print(f"Config: {results['config']}")
            print(f"Metrics: {results['metrics']}")
        if save_objects:
            self.save_objects(
                filename=name + ".pkl", filepath=filepath, save_all=save_all
            )

    def save_objects(
        self, filename="results.pkl", filepath=None, save_all=True
    ):
        """See documentation for ``Posterior.save_results(...)``."""
        fullpath = self._get_output_path(filepath, filename)
        if save_all:  # Include iteration history
            makedirs(dirname(fullpath), exist_ok=True)
            with open(fullpath, "wb") as f:
                dill.dump(self, f)
        else:  # Temporarily remove iteration history, then save
            hist = copy.deepcopy(self.history)
            self.history = []
            makedirs(dirname(fullpath), exist_ok=True)
            with open(fullpath, "wb") as f:
                dill.dump(self, f)
            self.history = hist  # Restore history

    def post_process(self):
        """Post-process intermediate results in ``self.history``, if needed."""
        pass

    def compute_metrics(self, num_iter=1):
        """Compute the requested metrics on the posterior(s)."""
        metrics = cfg_to_metrics(self.cfg)

        for name, metric in metrics.items():
            logging.info(f"Computing {name} metric...")
            record = []
            try:
                for prev_posterior in self.history[-num_iter:]:
                    record.append(metric(posterior=prev_posterior))
                record.append(metric(posterior=self))
                self.metrics[name] = record
                logging.info(f"{name} metric successfully computed.")
            except Exception:
                logging.error(f"Failed to compute {name} metric:")
                logging.error(
                    traceback.format_exc() + "\nAttempting to continue...\n"
                )

    def _get_output_path(self, filepath, filename):
        """Construct the full file output path from user input or defaults.

        Parameters
        ----------
        filepath : str
            The root filepath for saving results. If ``None``, will default to
            the current ``hydra.runtime.output_dir`` if available, and
            ``benchflow/temp`` otherwise.
        filename : str
            The filename (without extension) for saving results.
        """
        try:  # If hydra was initialized from command line, use its defaults:
            hydra = HydraConfig.get()
            if filepath is None:
                filepath = hydra.runtime.output_dir
        except ValueError:  # Otherwise use temp directory
            if filepath is None:
                filepath = dirname(dirname(dirname(abspath(__file__))))
                filepath = join(filepath, "temp")
        return join(filepath, filename)


def is_json_dumpable_dict(obj):
    """Check if an object can be dumped to JSON."""
    assert isinstance(obj, dict)
    try:
        json.dumps(obj)
    except TypeError:
        raise TypeError("Object is not JSON serializable.")
