import logging
import traceback

import numpy as np


def lml_error(posterior, *args, **kwargs):
    """Calculate the absolute error of the estimated lml.

    The ``Posterior`` object should provide a method ``get_lml_estimate()``
    which returns the estimated log marginal likelihood (LML), and the ``Task``
    should provide an attribute ``posterior_log_Z`` which is the reference LML.
    If either one is not found, ``benchflow`` will record the error and attempt
    to continue execution.

    Parameters
    ----------
    posterior : benchflow.posteriors.Posterior
        The inferred ``Posterior`` produced by an algorithm.

    Returns
    -------
    lml_error : float or Exception
        The absolute error between the inferred and reference LML, or the
        exception encountered in trying to compute it.
    """
    try:
        lml_post = posterior.get_lml_estimate()
    except Exception as e:
        logging.warn(traceback.format_exc() + "\nAttempting to continue...\n")
        return e
    if hasattr(posterior.task, "posterior_log_Z"):
        lml_true = posterior.task.posterior_log_Z
        return np.abs(lml_post - lml_true)
    else:
        return (
            "Task has no attribute posterior_log_Z (ground truth log"
            + " normalizing constant)."
        )


def lml(posterior, *args, **kwargs):
    """Get the  ``Posterior``'s estimate of the log marginal likelihood.

    The ``Posterior`` object should provide a method ``get_lml_estimate()``
    which returns the estimated log marginal likelihood (LML). If it has no
    such method ``benchflow`` will record the error and attempt to continue
    execution.

    Parameters
    ----------
    posterior : benchflow.posteriors.Posterior
        The inferred ``Posterior`` produced by an algorithm.

    Returns
    -------
    lml : float or Exception
        The estimated log marginal likelihood, or the Exception encountered in
        trying to access it.
    """
    try:
        return posterior.get_lml_estimate()
    except Exception as e:
        logging.warn(traceback.format_exc() + "\nAttempting to continue...\n")
        return e


def lml_sd(posterior, *args, **kwargs):
    """Get the standard deviation of ``Posterior``'s estimate of the LML.

    The ``Posterior`` object should provide a method ``get_lml_sd()``
    which returns the estimated standard deviation of the  log marginal
    likelihood (LML). If it has no such method ``benchflow`` will record the
    error and attempt to continue execution.

    Parameters
    ----------
    posterior : benchflow.posteriors.Posterior
        The inferred ``Posterior`` produced by an algorithm.

    Returns
    -------
    lml_sd : float or Exception
        The estimated standard deviation of the log marginal likelihood, or the
        Exception encountered in trying to access it.
    """
    try:
        return posterior.get_lml_sd()
    except Exception as e:
        logging.warn(traceback.format_exc() + "\nAttempting to continue...\n")
        return e
