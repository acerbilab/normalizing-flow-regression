import logging
import traceback
import warnings
from os.path import dirname, join
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from benchflow.distributions import Distribution
from benchflow.function_logger import ParameterTransformer, log_function


class Task:
    """Abstract base class for benchflow inference tasks/problems.

    Attributes
    ----------
    cfg : omegaconf.DictConfig
        The section of the Hydra config specifying the task information.
    D : int
        The dimension of the task.
    is_noisy : bool
        A flag indicating whether or not the task log-likelihood is noisy.
    name : str
        The general name of the task, e.g. "rosenbrock".
    id : str
        The unique name for the specific task configuration, e.g.
        "rosenbrock_default".
    sample_filename : str
        The name of the file containing reference posterior samples, located in
        ``benchflow/reference_samples/``.
    _log : Dict
        A dictionary containing information logged during function evaluations.
        See the ``benchflow.function_logger.log_function`` decorator for more
        details.
    """

    def __init__(self, cfg, D):
        """
        Parameters
        ----------
        cfg : omegaconf.DictConfig
            The section of the Hydra config specifying the task information.
        D : int
            The dimension of the task.
        """
        # Book-keeping:
        self.cfg = cfg
        # Get the name from the task config (which may be top-level),
        # otherwise lowercase this class name:
        if self.cfg.get("task"):
            self.name = self.cfg.task.get(
                "name", self.__class__.__name__.lower()
            )
        else:
            self.name = self.cfg.get("name", self.__class__.__name__.lower())
        self.id = self.cfg.get("id", self.name + "_default")
        self.sample_filename = f"{self.id}.csv"
        self.D = D  # Dimension
        if not hasattr(self, "is_noisy"):  # Default to non-noisy
            self.is_noisy = False

        # Set up logging:
        self._log = {}
        # Evaluated points, corresponding log-densities and noise estimates:
        self._log["theta"] = []
        self._log["log_y"] = []
        self._log["noise_est"] = []
        # wall-clock time before and after function evaluation:
        self._log["t0"] = timer()  # Initialization time
        self._log["t_before"] = []
        self._log["t_after"] = []
        self._log["fun_evals"] = 0  # Number of function evaluations

        self.mcmc_info = {  # Hints for MCMC reference sampling
            "multimodal": True,  # (Potentially) multimodal posterior
            "multiprocessing": True,  # Use multiprocessing, if possible
            "separate_tasks": True,  # Separate tasks for multiprocessing
        }

        self.task_info_dir = Path(__file__).parent / "task_info"
        self.transform_to_unconstrained_coordinates = getattr(
            self, "transform_to_unconstrained_coordinates", False
        )  # True means the task is transformed to take in unconstrained parameters as opposed to the original bounded parameters

    def log_prior(self, theta):
        """Compute the log density of the prior.

        Parameters
        ----------
        theta : np.ndarray
            The array of input point(s), of dimension ``(D,)`` or ``(n,D)``,
            where ``D`` is the problem dimension.

        Returns
        -------
            The log density of the prior at the input point(s), of dimension
            ``(n,1)``.
        """
        if not hasattr(self, "prior"):
            raise AttributeError("Task has no self.prior attribute.")
        if isinstance(self.prior, Distribution):
            return self.prior.logpdf(theta)
        elif callable(self.prior):
            return self.prior(theta)
        else:
            raise AttributeError("Task prior is not Distribution or callable.")

    def log_likelihood(self, theta):
        """Compute the log density of the likelihood.

        Parameters
        ----------
        theta : np.ndarray
            The array of input point(s), of dimension ``(D,)`` or ``(n,D)``,
            where ``D`` is the problem dimension.

        Returns
        -------
            The log density of the likelihood at the input point(s), of
            dimension ``(n,1)``.
        """
        return self.likelihood.logpdf(theta)

    @log_function
    def log_joint(self, theta):
        """Compute the log density of the joint.

        Parameters
        ----------
        theta : np.ndarray
            The array of input point(s), of dimension ``(D,)`` or ``(n,D)``,
            where ``D`` is the problem dimension.

        Returns
        -------
            The log density of the joint at the input point(s), of dimension
            ``(n,1)``.
        """
        log_likelihoods = self.log_likelihood(theta)
        log_priors = self.log_prior(theta)
        if not np.isscalar(log_priors) and not np.isscalar(log_likelihoods):
            assert log_likelihoods.shape == log_priors.shape, (
                "likelihood(s) and prior(s) have incompatible shapes: "
                f"{log_likelihoods.shape} and {log_priors.shape}."
            )
        return log_likelihoods + log_priors

    def x0(self, randomize=False, sz=1):
        """Select a starting point from within the tasks' plausible bounds.

        Parameters
        ----------
        randomize : bool, optional
            If ``False``, return the geometric mean of the plausible bounds,
            otherwise return an ``x0`` unformly at random from within the
            plausible bounds. Default ``False``.
        sz : int, optional
            The number of starting points to return. Default ``1``.

        Returns
        -------
        x0 : np.ndarray
            The array of starting points, of shape ``(sz, D)``, where ``D`` is
            the task dimension.
        """
        if randomize:
            return np.random.uniform(self.plb, self.pub, size=(sz, self.D))
        else:
            return np.tile((self.plb + self.pub) / 2, (sz, 1))

    def _get_prior_samples(self, n_samples=10000):
        """Get samples from the task's prior distribution. The samples might
        be outside of the bounds."""
        try:
            prior_samples = self.prior.sample(n_samples)
        except Exception:
            raise NotImplementedError(
                "sampling from the prior is not properly implemented."
            )  # noqa: E501
        return prior_samples

    def get_prior_samples(self, n_samples=10000):
        """Get samples from the task's prior distribution. The samples are
        ensured to be within the bounds.

        Parameters
        ----------
        n_samples : int, optional
            The number of posterior samples to return, by default 10000

        Returns
        -------
        prior_samples: np.ndarray
            Samples from the prior distribution, of shape ``(n_samples, D)``,
        """
        prior_samples = self._get_prior_samples(n_samples)
        assert prior_samples.ndim == 2
        # Prior samples should be within the bounds
        inds = np.all(
            (prior_samples >= self.lb) & (prior_samples <= self.ub), 1
        )
        prior_samples = prior_samples[inds]
        i_iter = 0
        while prior_samples.shape[0] < n_samples:
            new_samples = self._get_prior_samples(
                n_samples - prior_samples.shape[0]
            )
            inds = np.all(
                (new_samples >= self.lb) & (new_samples <= self.ub), 1
            )
            prior_samples = np.vstack((prior_samples, new_samples[inds]))
            i_iter += 1
            if i_iter > 10:
                raise RuntimeError(
                    "Could not sample enough points from the prior within the "
                    "bounds."
                )
        return prior_samples

    def get_posterior_samples(self, n_samples=10000):
        """Get samples from the task's posterior distribution.

        Attempts to sample directly from the task's posterior ``Distribution``
        object, if found. Otherwise attempts to get MCMC reference samples from
        the corresponding file in ``benchflow/reference_samples``.

        Parameters
        ----------
        n_samples : int, optional
            The number of posterior samples to return. Default ``10000``.

        Returns
        -------
        samples : np.ndarray or Exception
            If the samples were sucessfully gathered, returns them as an array
            of shape ``(n_samples, D)``, where ``D`` is the task dimension.
            Otherwise, returns an Exception describing the failure.
        """
        try:
            if hasattr(self, "posterior"):
                # Prefer sampling from analytical posterior
                return self.posterior.rvs(n_samples)
            elif hasattr(self, "get_reference_samples"):
                return self.get_reference_samples(n_samples)
            else:
                raise AttributeError(
                    "Task has no reference samples and no posterior to"
                    + " sample from."
                )
        except Exception as e:
            logging.warning(
                traceback.format_exc() + "\nAttempting to continue...\n"
            )
            return e

    def get_reference_samples(self, n_samples):
        """Get MCMC posterior samples from file.

        Parameters
        ----------
        n_samples : int
            The number of samples to return from the file (selected from the
            beginning of the file).

        Returns
        -------
        samples : np.ndarray
            The posterior samples, of shape ``(n_samples, D)`` where ``D`` is
            the task dimension.
        """
        root_path = dirname(dirname(__file__))
        file_path = join(root_path, "reference_samples", self.sample_filename)
        full_samples = np.genfromtxt(file_path, delimiter=",")
        n_reference = full_samples.shape[0]
        if n_samples > n_reference:
            warnings.warn(
                f"Task {self.id}: More posterior samples requested "
                + f"({n_samples}) than available from reference samples "
                + f"({n_reference}). Continuing with fewer samples."
            )
        else:
            thin = n_reference // n_samples
            full_samples = full_samples[::thin, :]
        return full_samples[:n_samples, :]

    def get_marginals(self):
        """Get a marginal density function, and the HPD marginal bounds.

        Tries first to return an analytical pdf from
        ``self.posterior.marginal_pdf``, otherwise interpolates a callable
        from ``self.marginal_pdf`` (an array of marginal densities).

        Returns
        -------
        marginal_pdf : callable
            A function ``marginal_pdf(x, d)`` which takes as input
            the position ``x : float`` and dimension ``d : int`` at which to
            evaulate the marginal pdf.
        marginal_bounds : np.ndarray
            An array of shape ``(2, D)``, where ``D`` is the task dimension.
            The pair ``marginal_bounds[:, d]`` represents the lower and upper
            bounds containing most (~99.99%) of the marginal posterior mass
            along dimension ``d``.
        """
        try:  # Analytical marginals
            return self.posterior.marginal_pdf, self.posterior.marginal_bounds
        except AttributeError:  # Pre-computed marginal pdf on grid
            y = self.posterior_marginal_pdf
            Nx = y.shape[0]
            x = np.zeros((Nx, self.D))
            for d in range(self.D):
                x[:, d] = np.linspace(
                    self.posterior_marginal_bounds[0, d],
                    self.posterior_marginal_bounds[1, d],
                    Nx,
                )
            return (
                lambda xx, d: interp1d(
                    x[:, d], y[:, d], fill_value=0, bounds_error=False
                )(xx),
                self.posterior_marginal_bounds,
            )

    def map(self, n_particles=25):
        """Get the MAP estimate of the task.

        Tries first to find any analytically defined mode, otherwise attempts
        to find the mode by optimization.

        Parameters
        ----------
        n_particles : int, optional
            The number of randomized initialization points to be used when
            seeking the mode through optimization. Default ``25``.

        Returns
        -------
        map : np.ndarray
            The MAP estimate, of shape ``(1, D)`` where ``D`` is the task
            dimension.
        """
        try:
            try:
                return self.posterior.mode  # Mode of analytical posterior
            except AttributeError:
                return self.posterior_mode  # Analytical posterior mode
        except AttributeError:  # Otherwise optimize to find mode(s)
            modes = np.zeros((n_particles, self.D))
            lpdf = np.zeros((n_particles, 1))
            for n in range(n_particles):  # Start at randomized x0's
                x0 = self.x0(randomize=True)
                bounds = np.vstack((self.lb, self.ub)).T
                opt = minimize(
                    lambda x: -self.log_joint(x),
                    x0,
                    method="BFGS",
                    bounds=bounds,
                )
                modes[n, :] = opt.x
                lpdf[n] = -opt.fun
            # Return the best of the local modes
            idx = np.argmax(lpdf, axis=0)
            return np.atleast_2d(modes[idx, :])

    def _transform_task_to_unconstrained(self):
        """Transform task to take unconstrained parameters as inputs."""
        if self.transform_to_unconstrained_coordinates:
            logging.warning(
                "Task already transformed to unconstrained coords."
            )
            return
        self.transform_to_unconstrained_coordinates = True
        self.transform = ParameterTransformer(
            self.D, self.lb, self.ub, self.plb, self.pub
        )
        self.lb = self.transform(self.lb)
        self.ub = self.transform(self.ub)
        self.plb = self.transform(self.plb)
        self.pub = self.transform(self.pub)
        return


def make_noisy(cls):
    """Derive and return a noisy-likelihood subclass from a given Task class.

    The ``hydra`` config may specify a callable noise function
    ``cfg.task.options.noise_function(theta)`` which should return
    ``(noise : float, est_sd : float)`` where ``noise`` is to be added to the
    task's log-likelihood at each point ``theta``, and ``est_sd`` is the
    estimated standard deviation of the noise. If no noise function is
    provided, Gaussian noise with a standard deviation of
    ``cfg.task.options.noise_sd`` (defaults to 1.0). The original ``Task``'s
    log-likelihood and log-joint functions are modified to return the
    appropriate densities with added noise, and the noise estimate. Additional
    methods ``_deterministic_log_likelihood`` and``_deterministic_log_joint``
    are added so that the original non-noisy densities can still be accessed.


    Parameters
    ----------
    cls : benchflow.tasks.Task
        The base task class from which to construct a noisy version.

    Returns
    -------
    NoisyTask : benchflow.tasks.Task
        The derived noisy class.
    """

    class NoisyTask(cls):
        def __init__(self, *args, **kwargs):
            self.noise_function = kwargs.pop("noise_function", None)
            # Default to constant normal noise, if not given noise function:
            if self.noise_function is None:
                self.noise_sd = kwargs.pop("noise_sd", 1.0)
                self.noise_function = lambda t: (
                    np.random.normal(
                        scale=self.noise_sd,
                        size=(np.atleast_2d(t).shape[0], 1),
                    ),
                    self.noise_sd,
                )
            # Finish initializing:
            self.is_noisy = True
            super().__init__(*args, **kwargs)
            self.id = "noisy_" + self.id

        # Save handles to the deterministic log-likelihood and log-joint,
        # e.g. for testing:
        def _deterministic_log_likelihood(self, theta):
            return super().log_likelihood(theta)

        def _deterministic_log_joint(self, theta):
            log_likelihoods = self._deterministic_log_likelihood(theta)
            log_priors = self.log_prior(theta)
            if not np.isscalar(log_priors) and not np.isscalar(
                log_likelihoods
            ):
                assert log_likelihoods.shape == log_priors.shape, (
                    "likelihood(s) and prior(s) have incompatible shapes: "
                    f"{log_likelihoods.shape} and {log_priors.shape}."
                )
            return log_likelihoods + log_priors

        # Override the log-likelihood and log-joint methods to add noise:
        def log_likelihood(self, theta):
            noise, noise_est = self.noise_function(theta)
            return super().log_likelihood(theta) + noise, noise_est

        def log_prior(self, theta):
            return super().log_prior(theta)

        @log_function
        def log_joint(self, theta):
            lls, noise_est = self.log_likelihood(theta)
            return lls + self.log_prior(theta), noise_est

    return NoisyTask
