from abc import abstractmethod

import numpy as np
import scipy.stats as sps
from scipy.optimize import fminbound, minimize

from benchflow.utilities import logsumexp


class Distribution:
    """Abstract base class for bnechflow probability distributions."""

    @abstractmethod
    def __init__(self):
        pass

    def logpdf(self, x):
        """Compute the log-pdf of the distribution.

        Parameters
        ----------
        x : np.ndarray
            The array of input point(s), of dimension ``(D,)`` or ``(n,D)``, where
            ``D`` is the distribution dimension.

        Returns
        -------
        logpdf : np.ndarray
            The log-density of the prior at the input point(s), of dimension
            ``(n,1)``.
        """
        x = np.atleast_2d(x)
        n, d = x.shape
        return self.distribution.logpdf(x).reshape((n, 1))

    def pdf(self, x):
        """Compute the pdf of the distribution.

        Parameters
        ----------
        x : np.ndarray
            The array of input point(s), of dimension ``(D,)`` or ``(n,D)``, where
            ``D`` is the distribution dimension.

        Returns
        -------
        pdf : np.ndarray
            The density of the prior at the input point(s), of dimension
            ``(n,1)``.
        """
        return np.exp(self.logpdf(x))

    def rvs(self, size=1):
        """Sample random variables from the distribution.

        Parameters
        ----------
        size : int or (int, int), optional
            The number (if int) or shape (if tuple) of random variables draw.

        Returns
        -------
        rvs : np.ndarray
            The random variables, of shape ``(size, D)`` if ``size`` is scalar,
            otherwise of shape ``size``.
        """
        if np.isscalar(size):
            return self.distribution.rvs(size=size).reshape((size, self.D))
        else:
            return self.distribution.rvs(size=size)


class GaussianMixture(Distribution):
    """``benchflow`` mixture of multivariate normal distributions.

    Attributes
    ----------
    D : int
        The distribution's dimension.
    weights : np.ndarray
        The mixture weights, shape ``(M,)`` where ``M`` is the number of
        components.
    log_weights : np.ndarray
        The log mixture weights, shape ``(M,)``.
    mu : np.ndarray
        The component means, shape ``(M, D)``.
    cov : np.ndarray
        The component covariance matrices, shape ``(M, D, D)``.
    mean : np.ndarray
        The mean of the full mixture, shape ``(1, D)``.
    total_cov : np.ndarray
        The covariance of the full mixture, shape ``(D, D)``.
    mode : np.ndarray
        The mode of the full mixture, shape ``(1, D)``.
    marginal_bounds : np.ndarray
        The bounds containing most of the distribution's mass. Shape
        ``(2, D)``, such that the interval between ``[0, d]`` and ``[1, d]``
        contains ~99.9998% of the marginal mass along dimension ``d``.
    marginal_densities : np.ndarray
        An array of marginal densities along each dimension, pre-computed on an
        evenly-spaced grid of ``Nx`` points (default ``10000``) between each
        pair of marginal bounds. Shape ``(D, Nx)``.
    """

    def __init__(self, mu, cov, weights=None, log_weights=None):
        """Initialize a Gaussian mixture distribution.

        Parameters
        ----------
        D : int
            The distribution's dimension.
        weights : np.ndarray, optional
            The mixture weights, shape ``(M,)`` where ``M`` is the number of
            components. At least one of ``weights`` or ``log_weights`` must be
            provided.
        log_weights : np.ndarray, optional
            The log mixture weights, shape ``(M,)``. At least one of
            ``weights`` or ``log_weights`` must be provided.
        """
        if weights is None and log_weights is None:
            raise ValueError("Must supply either weights or log_weights.")
        if log_weights is None:
            log_weights = np.log(weights)
        if weights is None:
            weights = np.exp(log_weights)
        self.weights = weights
        self.log_weights = log_weights
        self.mu = mu
        self.cov = cov
        (
            self.mean,
            self.total_cov,
            self.mode,
            self.marginal_bounds,
            self.marginal_densities,
        ) = self._gaussian_mixture_stats()

    def logpdf(self, x):
        """Compute the log-pdf of the distribution.

        Parameters
        ----------
        x : np.ndarray
            The array of input point(s), of dimension ``(D,)`` or ``(n,D)``, where
            ``D`` is the distribution dimension.

        Returns
        -------
        logpdf : np.ndarray
            The log-density of the prior at the input point(s), of dimension
            ``(n,1)``.
        """
        x = np.atleast_2d(x)
        M, D = self.mu.shape
        n, d = x.shape
        if d != D:
            raise ValueError(
                f"x has length {d} along dim 1, but dimension is {D}"
            )
        log_densities = np.zeros((n, M))
        for m in range(M):
            log_densities[:, m] = self.log_weights[
                m
            ] + sps.multivariate_normal.logpdf(
                x, self.mu[m, :], self.cov[m, :, :]
            )

        return logsumexp(log_densities, axis=1, keepdims=True)

    def rvs(self, size=1):
        """Sample random variables from the distribution.

        Parameters
        ----------
        size : int or (int, int), optional
            The number (if int) or shape (if tuple) of random variables draw.

        Returns
        -------
        rvs : np.ndarray
            The random variables, of shape ``(size, D)`` if ``size`` is scalar,
            otherwise of shape ``size``.

        Raises
        ------
        ValueError
            If the input ``size`` is a tuple whose second dimension is not
            ``D``.
        """
        M, D = self.mu.shape
        if not np.isscalar(size):
            size, D_ = np.atleast_2d(size).shape
            if D_ != D:
                raise ValueError(
                    "size {(size,D_)} does not match the broadcast shape of "
                    + "(_,{D})."
                )
        samples = np.zeros((size, D))
        for n in range(size):
            k = np.random.choice(
                M,
                p=self.weights.reshape(
                    M,
                ),
            )
            samples[n, :] = sps.multivariate_normal.rvs(
                mean=self.mu[k, :], cov=self.cov[k, :, :]
            )
        return samples

    def _gaussian_mixture_stats(self, Tol=1e-6, Nx=10000):
        """Compute statistics of the Gaussian mixture.

        Finds the mixture's mean and covariance analytically, finds the mode
        and marginal bounds by optimization, and pre-computes the marginal pdf
        on a grid between each pair marginal bounds.

        Parameters
        ----------
        Tol : float
            The tolerance of the marginal bounds: searches for ``lb`` and
            ``ub`` such that ``marginal_cdf(lb) == Tol`` and
            ``marginal_cdf(ub) == 1 - Tol``.
        Nx : int
            The number of evenly-spaced grid points, between each marginal
            ``lb`` and ``ub``, on which to evaluate the marginal pdf.

        Returns
        -------
        mean : np.ndarray
            The mean of the full mixture, shape ``(1, D)``.
        total_cov : np.ndarray
            The covariance of the full mixture, shape ``(D, D)``.
        mode : np.ndarray
            The mode of the full mixture, shape ``(1, D)``.
        marginal_bounds : np.ndarray
            The bounds containing most of the distribution's mass. Shape
            ``(2, D)``, such that the interval between ``[0, d]`` and
            ``[1, d]`` contains ``(1 - 2 * Tol)`` of the marginal mass along
            dimension ``d``.
        marginal_densities : np.ndarray
            An array of marginal densities along each dimension, pre-computed
            on an evenly-spaced grid of ``Nx`` points (default ``10000``)
            between each pair of marginal bounds. Shape ``(D, Nx)``.
        """
        M, D = self.mu.shape

        def target(x):
            return -self.pdf(x).squeeze()

        x_min = np.zeros((M, D))
        f_min = np.zeros((M, 1))
        for m in range(M):
            opt = minimize(target, self.mu[m, :], method="BFGS")
            x_min[m, :] = opt.x
            f_min[m] = opt.fun
        idx = np.argmin(f_min)

        mode = x_min[idx, :]
        mean = np.exp(logsumexp(self.log_weights + np.log(self.mu), axis=0))
        total_cov = np.zeros((D, D))
        for m in range(M):
            total_cov += self.weights[m] * (
                self.cov[m, :, :]
                + np.outer(self.mu[m, :] - mean, self.mu[m, :] - mean)
            )

        # Compute bounds for marginal total variation
        LB = np.zeros((D,))
        UB = np.zeros((D,))
        marginal_bounds = np.zeros((2, D))
        marginal_densities = np.zeros((D, Nx))

        for d in range(D):
            mu = self.mu[:, d]
            sigma = self.cov[:, d, d]

            LB[d] = np.amin(mu - 6 * sigma)
            UB[d] = np.amax(mu + 6 * sigma)
            cdf = lambda x: self.marginal_cdf(x, d)

            # Find lower/upper bounds
            x_lb = fminbound(lambda x: np.abs(cdf(x) - Tol), LB[d], UB[d])
            x_ub = fminbound(
                lambda x: np.abs(cdf(x) - (1 - Tol)), LB[d], UB[d]
            )
            if not np.isscalar(x_lb):
                assert x_lb.shape == (1,)
                x_lb = x_lb[0]
            if not np.isscalar(x_ub):
                assert x_ub.shape == (1,)
                x_ub = x_ub[0]
            marginal_bounds[:, d] = np.array([x_lb, x_ub])

            # Evaluate marginal pdf
            x_range = np.linspace(x_lb, x_ub, Nx)
            marginal_densities[d, :] = self.marginal_pdf(x_range, d)

        return mean, total_cov, mode, marginal_bounds, marginal_densities

    def marginal_pdf(self, x, d):
        """Compute the marginal pdf of the distribution.

        Returns the pdf of the marginal along dimension ``d``, at the point
        ``x``.

        Parameters
        ----------
        x : float
            The point at which to evaluate the marginal pdf.
        d : int
            The dimension :math: `d \\in [0, D)` of marginalization.
        """
        x = np.atleast_1d(x)
        n = x.shape[0]
        y = np.zeros((n,))
        M = self.weights.size

        mu = self.mu[:, d]
        sigma = np.sqrt(self.cov[:, d, d])
        for m in range(M):
            y += self.weights[m, :] * sps.norm.pdf(x, mu[m], sigma[m])

        return y

    def marginal_cdf(self, x, d):
        """Compute the marginal cdf of the distribution.

        Returns the cdf of the marginal along dimension ``d``, at the point
        ``x``.

        Parameters
        ----------
        x : float
            The point at which to evaluate the marginal cdf.
        d : int
            The dimension :math: `d \\in [0, D)` of marginalization.
        """
        x = np.atleast_1d(x)
        n = x.shape[0]
        y = np.zeros((n,))
        M = self.weights.size

        mu = self.mu[:, d]
        sigma = np.sqrt(self.cov[:, d, d])

        for m in range(M):
            y += self.weights[m, :] * sps.norm.cdf(x, mu[m], sigma[m])

        return y


class IndependentStudentT(Distribution):
    """``benchflow`` independent multivariate Student's t-distribution.

    Each marginal distribution is a Student's t-distribution with specified
    location, scale, and degrees of freedom. Largely a wrapper around the
    corresponding ``scipy.stats`` distribution, to ensure input/output shapes
    are consistent.

    Attributes
    ----------
    D : int
        The distribution's dimension.
    df : np.ndarray
        The degrees of freedom of each marginal distribution, shape ``(D,)``.
    loc : np.ndarray
        The location parameters of each marginal distribution, shape ``(D,)``.
    scale : np.ndarray
        The scale parameters of each marginal distribution, shape ``(D,)``.
    distribution : scipy.stats distribution
        The underlying ``scipy.stats`` distribution.
    """

    def __init__(self, df, loc=None, scale=None):
        """Initialize an independent multivariate Student's t-distribution.

        Parameters
        ----------
        df : np.ndarray
            The degrees of freedom of each marginal distribution, shape
            ``(D,)``.
        loc : np.ndarray
            The location parameters of each marginal distribution, shape
            ``(D,)``.
        scale : np.ndarray
            The scale parameters of each marginal distribution, shape ``(D,)``.
        """
        self.D = df.size
        if loc is None:
            loc = np.zeros(df.shape)
        if scale is None:
            scale = np.ones(df.shape)
        self.df = df
        self.loc = loc
        self.scale = scale
        self.distribution = sps.t(df=df, loc=loc, scale=scale)

    def logpdf(self, x):
        """Compute the log-pdf of the distribution.

        Parameters
        ----------
        x : np.ndarray
            The array of input point(s), of dimension ``(D,)`` or ``(n,D)``, where
            ``D`` is the distribution dimension.

        Returns
        -------
        logpdf : np.ndarray
            The log-density of the prior at the input point(s), of dimension
            ``(n,1)``.
        """
        xx = np.atleast_2d(x)
        return np.sum(self.distribution.logpdf(xx), axis=1, keepdims=True)


class Normal(Distribution):
    """``benchflow`` (multivariate) normal distribution.

    Essentially a wrapper around the corresponding ``scipy.stats``
    distribution, to ensure that the input/output shapes are consistent.

    Attributes
    ----------
    D : int
        The distribution's dimension.
    mean : np.ndarray
        The distribution's mean, shape ``(1, D)``.
    mode : np.ndarray
        The distribution's mode (same as ``mean``), shape ``(1, D)``.
    cov : np.ndarray
        The covariance matrix, shape ``(D, D)``.
    is_diag : bool
        A flag indicating if the covariance matrix is (close to) diagonal.
    distribution : scipy.stats distribution
        The underlying ``scipy`` distribution.
    """

    def __init__(self, D, mean=None, sigma=1.0, cov=None):
        """Initialize a multivariate normal distribution.

        Parameters
        ----------
        D : int
            The distribution's dimension.
        mean : np.ndarray, optional
            The distribution mean, of shape ``(1, D)``. Defaults to zero-mean.
        sigma : float or np.ndarray, optional
            The distribution standard deviation. If ``sigma`` is scalar, it is
            interpreted as a scaling factor of a standard normal distribution.
            If ``sigma`` is one-dimensional or of shape ``(1, D)`` or
            ``(D, 1)``, it is interpreted as the square-root diagonal of an
            indep. covariance matrix. If ``sigma`` has shape ``(D, D)`` it is
            interpreted as the elementwise square root of the covariance
            matrix. Default is 1.0, leading to a standard normal distribution.
        cov : np.ndarray, optional
            The distribution's covariance matrix. Takes precedence over
            ``sigma`` if both are provided. Default ``None``.
        """
        self.D = D
        if mean is None:
            mean = np.zeros((D,))
        if cov is None:
            if np.isscalar(sigma):
                cov = np.eye(D) * sigma**2
            elif sigma.ndim == 1:
                cov = np.diag(sigma.reshape((D,)) ** 2)
            elif sigma.shape == ((1, D)) or sigma.shape == ((D, 1)):
                cov = np.diag(sigma.reshape((D,)) ** 2)
            elif sigma.shape == ((D, D)):
                cov = sigma**2
            else:
                raise ValueError(
                    f"`sigma` is the wrong shape for {D} dimensional distribution."
                )

        mean = mean.reshape((1, D))
        self.mean = mean
        self.mode = mean
        self.cov = cov
        self.is_diag = np.allclose(cov, np.diag(np.diagonal(cov)))
        self.distribution = sps.multivariate_normal(
            self.mean.reshape((D,)), self.cov
        )

        self.marginal_bounds = np.concatenate(
            [
                self.mean - 3 * np.diag(self.cov),
                self.mean + 3 * np.diag(self.cov),
            ]
        )

    def marginal_pdf(self, x, d):
        """Compute the marginal pdf of the distribution.

        Returns the pdf of the marginal along dimension ``d``, at the point
        ``x``.

        Parameters
        ----------
        x : float
            The point at which to evaluate the marginal pdf.
        d : int
            The dimension :math: `d \\in [0, D)` of marginalization.
        """
        return sps.norm.pdf(x, self.mean[0, d], np.sqrt(self.cov[d, d]))

    def marginal_cdf(self, x, d):
        """Compute the marginal cdf of the distribution.

        Returns the cdf of the marginal along dimension ``d``, at the point
        ``x``.

        Parameters
        ----------
        x : float
            The point at which to evaluate the marginal cdf.
        d : int
            The dimension :math: `d \\in [0, D)` of marginalization.
        """
        return sps.norm.cdf(x, self.mean[0, d], np.sqrt(self.cov[d, d]))


class Uniform(Distribution):
    """``benchflow`` box-uniform distribution.

    Attributes
    ----------
    D : int
        The distribution's dimension.
    lb : np.ndarray
        The lower bounds of the box along each dimension, shape ``(1, D)``.
    ub : np.ndarray
        The upper bounds of the box along each dimension, shape ``(1, D)``.
    log_volume : float
        The logarithm of the volume of the bounding box.
    """

    def __init__(self, D, lb, ub):
        """Initialize a box-uniform distribution.

        Parameters
        ----------
        D : int
            The distribution's dimension.
        lb : np.ndarray
            The lower bounds of the box along each dimension, shape ``(1, D)``.
        ub : np.ndarray
            The upper bounds of the box along each dimension, shape ``(1, D)``.

        Raises
        ------
        ValueError
            If the shapes of ``lb`` and ``ub`` do not match, or if ``lb`` is
            not strictly less than ``ub`` (elementwise).
        """
        if np.shape(lb) != np.shape(ub):
            raise ValueError(
                f"lb of shape {np.shape(lb)} does not match ub of shape "
                + f"{np.shape(ub)}."
            )
        if not np.all(lb < ub):
            raise ValueError(f"lb={lb} is not strictly less than ub={ub}.")
        self.D = D
        self.lb = lb
        self.ub = ub
        self.log_volume = np.log(np.prod(ub - lb))

    def logpdf(self, x):
        """Compute the log-pdf of the distribution.

        Parameters
        ----------
        x : np.ndarray
            The array of input point(s), of dimension ``(D,)`` or ``(n,D)``, where
            ``D`` is the distribution dimension.

        Returns
        -------
        logpdf : np.ndarray
            The log-density of the prior at the input point(s), of dimension
            ``(n,1)``.
        """
        xx = np.atleast_1d(x)
        if xx.ndim == 1:
            if xx.size != self.D:
                raise ValueError(
                    f"x has incorrect dimensions {xx.shape} for distribution dimension {self.D}"
                )
            return np.full((1, 1), -self.log_volume)
        else:
            n, d = x.shape
            if d != self.D:
                raise ValueError(
                    f"x has incorrect dimensions {x.shape} for distribution dimension {self.D}"
                )

            return np.full((n, 1), -self.log_volume)


class Flat(Distribution):
    """``benchflow`` improper flat distribution.

    Returns a constant density of 1.0, regardless of input.

    Attributes
    ----------
    D : int
        The distribution's dimension.
    """

    def __init__(self, D):
        """Initialize an improper flat distribution.

        Parameters
        ----------
        D : int
            The distribution's dimension.
        """
        self.D = D

    def logpdf(self, x):
        """Compute the log-pdf of the distribution.

        Parameters
        ----------
        x : np.ndarray
            The array of input point(s), of dimension ``(D,)`` or ``(n,D)``, where
            ``D`` is the distribution dimension.

        Returns
        -------
        logpdf : np.ndarray
            The log-density of the prior at the input point(s), of dimension
            ``(n,1)``.
        """
        xx = np.atleast_2d(x)
        n, d = xx.shape
        if d != self.D:
            raise ValueError(
                f"x has incorrect dimensions {x.shape} for distribution dimension {self.D}"
            )
        return np.zeros((n, 1))


class SplineTrapezoidal(Distribution):
    r"""Multivariate spline-trapezoidal prior.

    A prior distribution represented by a density with external bounds ``a``
    and ``b`` and internal points ``u`` and ``v``. Each marginal distribution
    has a spline-trapezoidalal density which is uniform between ``u[i]`` and
    ``v[i]`` and falls of as a cubic spline to zero ``a[i]`` and ``b[i]``, such
    that the pdf is continuous and its derivatives at ``a[i]``, ``u[i]``,
    ``v[i]``, and ``b[i]`` are zero (so the derivatives are also continuous)::

                 ______________________
                |      __________      |
                |     / |      | \     |
        p(X(i)) |    |  |      |  |    |
                |    |  |      |  |    |
                |___/___|______|___\___|
                  a[i] u[i]  v[i] b[i]
                          X(i)

    The overall density is a product of these marginals.

    Attributes
    ----------
    D : int
        The dimension of the prior distribution.
    a : np.ndarray
        The lower bound(s), shape `(1, D)`.
    u : np.ndarray
        The lower pivot(s), shape `(1, D)`.
    v : np.ndarray
        The upper pivot(s), shape `(1, D)`.
    b : np.ndarray
        The upper bound(s), shape `(1, D)`.
    """

    def __init__(self, a, u, v, b, D=None):
        """Initialize a multivariate trapezoidal prior.

        Parameters
        ----------
        a : np.ndarray | float
            The lower bound(s), shape `(D,)` where `D` is the dimension
            (parameters of type ``float`` will be tiled to this shape).
        u : np.ndarray | float
            The lower pivot(s), shape `(D,)` where `D` is the dimension
            (parameters of type ``float`` will be tiled to this shape).
        v : np.ndarray | float
            The upper pivot(s), shape `(D,)` where `D` is the dimension
            (parameters of type ``float`` will be tiled to this shape).
        b : np.ndarray | float
            The upper bound(s), shape `(D,)` where `D` is the dimension
            (parameters of type ``float`` will be tiled to this shape).
        D : int, optional
            The distribution dimension. If given, will convert scalar `a`, `u`,
            `v`, and `b` to this dimension.

        Raises
        ------
        ValueError
            If the order ``a[i] < u[i] < v[i] < b[i]`` is not respected, for any `i`.
        """
        self.a, self.u, self.v, self.b = tile_inputs(
            a, u, v, b, size=D, squeeze=True
        )
        if np.any(
            (self.a >= self.u) | (self.u >= self.v) | (self.v >= self.b)
        ):
            raise ValueError(
                "Bounds and pivots should respect the order a < u < v < b."
            )
        self.D = self.a.size

    def logpdf(self, x):
        """Compute the log-pdf of the multivariate trapezoidal prior.

        Parameters
        ----------
        x : np.ndarray
            The array of input point(s), of dimension `(D,)` or `(n, D)`, where
            `D` is the distribution dimension.

        Returns
        -------
        log_pdf : np.ndarray
            The log-density of the prior at the input point(s), of dimension
            `(n, 1)`.
        """
        x = np.atleast_2d(x)
        n, D = x.shape
        log_pdf = np.full_like(x, -np.inf)
        # a b c d
        # a u v b
        # norm_factor = u - v + 0.5 * (b - v + u - a)
        log_norm_factor = np.log(0.5 * (self.v - self.u + self.b - self.a))

        # ignore log(0) warnings here
        old_settings = np.seterr(divide="ignore")
        for d in range(D):
            # Left tail
            mask = (x[:, d] >= self.a[d]) & (x[:, d] < self.u[d])
            z = (x[mask, d] - self.a[d]) / (self.u[d] - self.a[d])
            log_pdf[mask, d] = (
                np.log(-2 * z**3 + 3 * z**2) - log_norm_factor[d]
            )

            # Plateau
            mask = (x[:, d] >= self.u[d]) & (x[:, d] < self.v[d])
            log_pdf[mask, d] = -log_norm_factor[d]

            # Right tail
            mask = (x[:, d] >= self.v[d]) & (x[:, d] < self.b[d])
            z = 1 - (x[mask, d] - self.v[d]) / (self.b[d] - self.v[d])
            log_pdf[mask, d] = (
                np.log(-2 * z**3 + 3 * z**2) - log_norm_factor[d]
            )
        np.seterr(**old_settings)

        return np.sum(log_pdf, axis=1, keepdims=True)


def tile_inputs(*args, size=None, squeeze=False):
    """Tile scalar inputs to have the same dimension as array inputs.

    If all inputs are given as scalars, returned arrays will have shape `size`
    if `size` is a tuple, or shape `(size,)` if `size` is an integer.

    Parameters
    ----------
    *args : [Union[float, np.ndarray]]
        The inputs to tile.
    size : Union[int, tuple], optional
        The desired size/shape of the output, default `(1,)`.
    squeeze : bool
        If `True`, then drop 1-d axes from inputs. Default `False`.

    Raises
    ------
    ValueError
        If the non-scalar arguments do not have the same shape, or if they do
        not agree with `size`.
    """
    if type(size) == int:
        size = (size,)
    shape = None

    # Check that all non-scalar inputs have the same shape
    args = list(args)
    for i, arg in enumerate(args):
        if not (np.isscalar(arg)):
            if squeeze:
                arg = args[i] = np.atleast_1d(np.squeeze(np.array(arg)))
            else:
                arg = args[i] = np.array(arg)
            if shape is None:
                shape = arg.shape
            elif arg.shape != shape:
                raise ValueError(
                    f"All inputs should have the same shape, but found inputs with shapes {shape} and {arg.shape}."
                )

    if size is None:
        if shape is None:
            # Default to shape (1,)
            size = (1,)
        else:
            # Or use inferred shape
            size = shape

    for i, arg in enumerate(args):
        if np.isscalar(arg):
            args[i] = np.full(size, arg)
        else:
            args[i] = args[i].reshape(size)

    return args
