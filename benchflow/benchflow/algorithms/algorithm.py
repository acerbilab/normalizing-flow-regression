import logging
from abc import abstractmethod

import numpy as np

from benchflow.posteriors import Posterior
from benchflow.utilities.cfg import (
    cfg_to_args,
    cfg_to_posterior_class,
    cfg_to_seed,
    cfg_to_task,
)


class Algorithm:
    """Abstract base class for inference algorithms"""

    def __init__(self, cfg):
        """Initialize an algorithm according to the ``hydra`` config."""
        # Get task and benchflow Posterior class object from config
        self.task = cfg_to_task(cfg)
        try:
            self.PosteriorClass = cfg_to_posterior_class(cfg)
        except AttributeError:
            logging.warning(
                "No corresponding posterior class for the algorithm found."
            )
        self.cfg = cfg

        # Fix random seed (if specified):
        if cfg.get("seed") is not None:
            np.random.seed(cfg_to_seed(cfg))

    @abstractmethod
    def run(self) -> Posterior:
        """Run the inference and return a corresponding ``Posterior``.

        Returns
        -------
        posterior : benchflow.posteriors.Posterior
            The ``Posterior`` object containing relevant information about the
            algorithm's execution and inference.
        """
        # Get any additional arguments / keyword argument:
        args, kwargs = cfg_to_args(self.cfg, self.task)
        # Construct the appropriate ``Posterior`` object:
        return self.PosteriorClass(self.core(*args, **kwargs))

    def core(self, *args, **kwargs):
        """Run the core algorithm."""
        pass
