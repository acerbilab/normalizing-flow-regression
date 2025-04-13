import numpy as np
import scipy.stats as scs
from scipy.integrate import dblquad, quad

from .task import Task


class MultiBanana(Task):
    """Synthetic task with multivariate Rosenbrock-Gaussian log likelihood.

    Attributes
    ----------
    a : float, optional
        A parameter controlling the shape of the log likelihood. Default 1.
    b : float, optional
        A parameter controlling the shape of the log likelihood. Default 100.
    scale : float, optional
        A parameter controlling the output scale of the log density.
    likelihood : callable
        The log density of the likelihood.
    prior : benchflow.distributions.Distribution or callable, optional
        The prior Distribution, or a callable which returns the log density
        of the prior. Defaults to standard Normal.
    lb : np.ndarray
        The absolute lower bounds of the task. Shape ``(1,D)``.
    ub : np.ndarray
        The absolute upper bounds of the task. Shape ``(1,D)``.
    plb : np.ndarray
        The plausibe lower bounds of the task. Shape ``(1,D)``.
    pub : np.ndarray
        The plausible upper bounds of the task. Shape ``(1,D)``.
    """

    def __init__(self, cfg, D=10, num_bananas=1):
        """
        Parameters
        ----------
        cfg : omegaconf.DictConfig
            The section of the Hydra config specifying the task information.
        prior : benchflow.distributions.Distribution or callable, optional
            The prior Distribution, or a callable which returns the log density
            of the prior. Defaults to standard Normal.
        a : float
            A shape parameter for the Rosenbrock function.
        b : float
            A shape parameter for the Rosenbrock function.
        scale : float
            An output scale parameter for the Rosenbrock function.
        """
        # Initialize general attributes:
        super().__init__(cfg=cfg, D=D)
        assert D >= 4 and D % 2 == 0
        self.num_bananas = num_bananas

        self.prior_mu = np.zeros((1, D))
        self.prior_std = 3 * np.ones((1, D))
        self.lpriorfun = lambda x: np.sum(
            scs.norm.logpdf(x, self.prior_mu, self.prior_std), 1
        )

        self.lb = np.full((1, D), -np.inf)  # Lower bounds
        self.ub = np.full((1, D), np.inf)  # Upper bounds
        self.plb = np.full(
            (1, D), self.prior_mu - self.prior_std
        )  # Plausible LB
        self.pub = np.full(
            (1, D), self.prior_mu + self.prior_std
        )  # Plausible UB
        # # Integrate to find normalizing constant:

        def target_1(x, y):
            return np.exp(
                -((x**2 - y) ** 2 + (x - 1) ** 2 / 100)
                + np.sum(
                    scs.norm.logpdf(
                        np.array([x, y]),
                        self.prior_mu[0, :2],
                        self.prior_std[0, :2],
                    )
                )
            )

        def target_2(x):
            return np.exp(
                -(x**2) / 2
                + scs.norm.logpdf(
                    x, self.prior_mu[0, -1], self.prior_std[0, -1]
                )
            )

        if self.num_bananas == 2 and D == 6:
            self.posterior_log_Z = -6.822309022102013
        else:
            v1 = np.log(dblquad(target_1, -np.inf, np.inf, -np.inf, np.inf)[0])
            v2 = np.log(quad(target_2, -np.inf, np.inf)[0])
            self.posterior_log_Z = v1 * self.num_bananas + v2 * (
                self.D - self.num_bananas * 2
            )

        print("True log marginal likelihood: ", self.posterior_log_Z)

    def log_likelihood(self, theta):
        """Compute the log density of the likelihood.

        Parameters
        ----------
        theta : np.ndarray
            The array of input point(s), of dimension ``(D,)`` or ``(n,D)``, where
            ``D`` is the problem dimension.

        Returns
        -------
            The log density of the likelihood at the input point(s), of
            dimension ``(n,1)``.
        """
        theta = np.atleast_2d(theta)
        # assert theta.shape == (1, self.D)

        ll = 0
        for i in range(self.num_bananas):
            xs = theta[:, 2 * i]
            ys = theta[:, 2 * i + 1]
            ll -= (xs**2 - ys) ** 2 + (xs - 1) ** 2 / 100
        zs = theta[:, 2 * self.num_bananas :]
        ll -= np.sum(zs**2, 1) / 2
        return np.atleast_2d(ll)

    def log_prior(self, theta):
        theta = np.atleast_2d(theta)
        return np.atleast_2d(self.lpriorfun(theta)).T

    def _get_prior_samples(self, n_samples=1):
        return np.random.multivariate_normal(
            self.prior_mu.ravel(), np.diagflat(self.prior_std**2), n_samples
        )
