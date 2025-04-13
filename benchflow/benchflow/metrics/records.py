import logging
import traceback


def fun_evals(posterior, *args, **kwargs):
    """Record the number of function evaluations at each iteration."""
    try:
        return posterior.fun_evals()
    except Exception as e:
        logging.warn(traceback.format_exc() + "\nAttempting to continue...\n")
        return e


def idx_best(posterior, *args, **kwargs):
    """Record the index of the best previous VP at each iteration."""
    try:
        return posterior.idx_best()
    except Exception as e:
        logging.warn(traceback.format_exc() + "\nAttempting to continue...\n")
        return e


def wall_time(posterior, *args, **kwargs):
    """Record the wall-clock time at each iteration."""
    try:
        return posterior.wall_time()
    except Exception as e:
        logging.warn(traceback.format_exc() + "\nAttempting to continue...\n")
        return e
