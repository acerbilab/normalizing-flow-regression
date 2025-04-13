from typing import Optional, Union

import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import interp1d

from .stats import kde1d


def mtv(
    X1: Optional[Union[np.ndarray, callable]] = None,
    X2: Optional[Union[np.ndarray, callable]] = None,
    posterior=None,
    *args,
    **kwargs,
) -> Union[np.ndarray, Exception]:
    """
    Marginal total variation distances between two sets of posterior samples.

    Compute the total variation distance between posterior samples X1 and
    posterior samples X2, separately for each dimension (hence
    "marginal" total variation distance, MTV).

    Parameters
    ----------
    X1 : np.ndarray or callable, optional
        A ``N1``-by-``D`` matrix of samples, typically N1 = 1e5.
        Alternatively, may be a callable ``X1(x, d)`` which returns the marginal
        pdf along dimension ``d`` at point(s) ``x``.
    X2 : np.ndarray or callable, optional
        Another ``N2``-by-``D`` matrix of samples, typically N2 = 1e5.
        Alternatively, may be a callable ``X2(x, d)`` which returns the marginal
        pdf along dimension ``d`` at point(s) ``x``.
    posterior: benchflow.posteriors.Posterior, optional
        The posterior object from a benchflow run. Used to obtain samples if
        ``X1`` or ``X2`` are ``None``.

    Returns
    -------
    mtv: np.ndarray
        A ``D``-element vector whose elements are the total variation distance
        between the marginal distributions of ``vp`` and ``vp1`` or ``samples``,
        for each coordinate dimension.

    Raises
    ------
    ValueError
        Raised if neither ``vp2`` nor ``samples`` are specified.

    Notes
    -----
    The total variation distance between two densities `p1` and `p2` is:

    .. math:: TV(p1, p2) = \\frac{1}{2} \\int | p1(x) - p2(x) | dx.

    """
    # If samples are not provided, fetch them from the posterior object:
    if all(a is None for a in [X1, X2, posterior]):
        raise ValueError("No samples/callable or posterior provided.")
    if posterior is not None:
        try:  # Get analytical marginals, if possible
            X1, bounds_1 = posterior.get_marginals()
        except AttributeError:  # Otherwise use samples
            X1 = posterior.get_samples()
            if isinstance(X1, Exception):
                return X1  # Record errors, if any
        try:  # Get analytical marginals, if possible
            X2, bounds_2 = posterior.task.get_marginals()
        except AttributeError:  # Otherwise use samples
            X2 = posterior.task.get_posterior_samples()
            if isinstance(X2, Exception):
                return X2  # Record errors, if any
        D = posterior.task.D
    else:
        D = X1.shape[1]

    nkde = 2**13
    mtv = np.zeros((D,))

    # Compute marginal total variation
    for d in range(D):
        if not callable(X1):
            yy1, x1mesh, _ = kde1d(X1[:, d], nkde)
            # Ensure normalization
            yy1 = yy1 / simpson(yy1, x1mesh)

            def f1(x):
                return interp1d(
                    x1mesh,
                    yy1,
                    kind="cubic",
                    fill_value=np.array([0]),
                    bounds_error=False,
                )(x)

        else:

            def f1(x):
                return X1(x, d).ravel()  # Analytical marginal

            x1mesh = bounds_1[:, d]  # Marginal bounds

        if not callable(X2):
            yy2, x2mesh, _ = kde1d(X2[:, d], nkde)
            # Ensure normalization
            yy2 = yy2 / simpson(yy2, x2mesh)

            def f2(x):
                return interp1d(
                    x2mesh,
                    yy2,
                    kind="cubic",
                    fill_value=np.array([0]),
                    bounds_error=False,
                )(x)

        else:

            def f2(x):
                return X2(x, d).ravel()  # Analytical marginal

            x2mesh = bounds_2[:, d]  # Marginal bounds

        def f(x):
            return np.abs(f1(x) - f2(x))

        lb = min(x1mesh[0], x2mesh[0])
        ub = max(x1mesh[-1], x2mesh[-1])
        if not np.isinf(lb) and not np.isinf(ub):
            # Grid integration (faster)
            grid = np.linspace(lb, ub, int(1e6))
            y_tot = f(grid)
            mtv[d] = 0.5 * simpson(y_tot, grid)
        else:
            # QUADPACK integration (slower)
            mtv[d] = 0.5 * quad(f, lb, ub)[0]
    mtv = np.maximum(0, mtv)  # Ensure non-negative
    mtv = np.minimum(1, mtv)  # Ensure bounded by 1
    return mtv


def mmtv(
    X1: Optional[Union[np.ndarray, callable]] = None,
    X2: Optional[Union[np.ndarray, callable]] = None,
    posterior=None,
    *args,
    **kwargs,
) -> Union[float, Exception]:
    """
    Mean marginal total variation dist. between two set of posterior samples.
    """
    result = mtv(X1, X2, posterior)
    if isinstance(result, Exception):
        return result
    else:
        return result.mean()
