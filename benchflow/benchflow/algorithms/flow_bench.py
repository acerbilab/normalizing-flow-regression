import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig

from benchflow.posteriors.flow_posterior import FlowRegressionPosterior
from benchflow.utilities.cfg import cfg_to_args, cfg_to_seed
from benchflow.utilities.dataset import (
    read_generated_initial_set,
)
from nfr.nfr import NFR

from .algorithm import Algorithm


class FlowRegression(Algorithm):
    """``benchmark`` algorithm for Normalizing Flow surrogate via regression."""

    def __init__(self, cfg):
        """Initialize an algorithm according to the ``hydra`` config."""
        # Get task and benchflow Posterior class object from config
        super().__init__(cfg)

        # Fix random seed (if specified):
        if cfg.get("seed") is not None:
            self.seed = cfg_to_seed(cfg)
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def run(self) -> FlowRegressionPosterior:
        """Run the inference and return a ``PyVBMCPosterior``.

        Returns
        -------
        posterior : benchflow.posteriors.FlowRegressionPosterior
            The ``Posterior`` object containing relevant information about the
            algorithm's execution and inference.
        """

        # Get any additional arguments / keyword arguments:
        args, kwargs = cfg_to_args(self.cfg, self.task)

        # Switch to noisy algorithm for noisy tasks, if not already specified:
        if kwargs is None:
            kwargs = {}
        if "options" not in kwargs:
            kwargs["options"] = {}
        options = kwargs.get("options")

        if self.cfg.task.get("noisy") and not options.get(
            "specify_target_noise"
        ):
            logging.info("Noisy task.")
            if kwargs.get("options"):
                kwargs["options"]["specify_target_noise"] = True
            else:
                kwargs["options"] = {"specify_target_noise": True}

        hydra_cfg = HydraConfig.get()
        if hydra_cfg.mode.name == "MULTIRUN":
            kwargs["options"]["experiment_folder"] = os.path.join(
                hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir
            )
            print(kwargs["options"]["experiment_folder"])
        else:
            kwargs["options"]["experiment_folder"] = hydra_cfg.run.dir

        D = self.task.D
        if self.cfg.algorithm.get("initial_set"):
            data = read_generated_initial_set(self.task, self.cfg)
            x = data["X"]
            log_likes = data["log_likes"]
            log_priors_orig = data["log_priors_orig"]
            S_orig = data["S_orig"]
            fun_evals_start = data["fun_evals"]

            kwargs["x0"] = x
            kwargs["options"]["log_likes"] = log_likes
            kwargs["options"]["log_priors_orig"] = log_priors_orig
            if S_orig is not None:
                kwargs["options"]["S_orig"] = S_orig
        else:
            raise NotImplementedError
        kwargs["bench_seed"] = self.seed

        # Initialize:
        nfr_object = NFR(*args, **kwargs)
        # Run inference:
        posterior_nfr = nfr_object.optimize()
        posterior = self.PosteriorClass(
            self.cfg,
            self.task,
            nfr_object=nfr_object,
            **posterior_nfr,
        )
        try:
            from benchflow.plotting.utils import corner_plot

            experiment_folder = Path(kwargs["options"]["experiment_folder"])
            samples = posterior.get_samples(10000)
            gt_samples = self.task.get_posterior_samples(10000)
            fig = corner_plot(
                gt_samples,
                samples,
                save_as=experiment_folder / "corner_plot.png",
            )
        except Exception as e:
            logging.error(e)
        # Construct Posterior object from results:
        return posterior
