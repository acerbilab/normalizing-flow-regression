from typing import Optional

import numpy as np

from .stats import kldiv_mvn


def gskl(
    X1: Optional[np.ndarray] = None,
    X2: Optional[np.ndarray] = None,
    posterior=None,
    *args,
    **kwargs,
):
    """ "Gaussianized" symmetric Kullback-Leibler divergence (gsKL) between two
    sets of samples.

    gsKL is the symmetric KL divergence between two multivariate normal
    distributions with the same moments as samples. The symmetric KL divergence
    is the average of forward and reverse KL divergence.

    Parameters
    ----------
    X1 : np.ndarray, optional
        A ``N1``-by-``D`` matrix of samples.
    X2 : np.ndarray, optional
        Another ``N2``-by-``D`` matrix of samples.
    posterior: benchflow.posteriors.Posterior, optional
        The posterior object from a benchflow run. Used to obtain samples if
        ``X1`` or ``X2`` are ``None``.

    Returns
    -------
    kl: float
        gsKL of the two sets of samples.

    Notes
    -----
    Since the KL divergence is not symmetric, the method returns the average of
    forward and the reverse KL divergence, that is KL(``vp1`` || ``vp2``) and
    KL(``vp2`` || ``vp1``).

    """
    # If samples are not provided, fetch them from the posterior object:
    if all(a is None for a in [X1, X2, posterior]):
        raise ValueError("No samples or posterior provided.")
    if posterior is not None:
        X1 = posterior.get_samples()
        X2 = posterior.task.get_posterior_samples()
        if isinstance(X1, Exception):  # Record errors, if any
            return X1
        if isinstance(X2, Exception):
            return X2

    q1mu = np.mean(X1, axis=0)
    q1sigma = np.cov(X1.T)
    q2mu = np.mean(X2, axis=0)
    q2sigma = np.cov(X2.T)

    kls = kldiv_mvn(q1mu, q1sigma, q2mu, q2sigma)

    # Correct for numerical errors
    kls[kls < 0] = 0
    return kls.mean()
