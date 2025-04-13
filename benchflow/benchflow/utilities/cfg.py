import hashlib
import logging
from importlib import import_module
from inspect import getmembers, isfunction

from omegaconf import DictConfig, ListConfig, OmegaConf


def cfg_to_algorithm(cfg):
    """Find the Algorithm corresponding to a hydra config.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        The main hydra configuration object.

    Returns
    -------
    algorithm : benchflow.algorithms.Algorithm
        An instance of the algorithm specified by the config.
    """
    import benchflow.algorithms

    Algorithm = getattr(benchflow.algorithms, cfg.algorithm["class"])
    return Algorithm(cfg)


def cfg_to_posterior_class(cfg):
    """Find the Posterior subclass corresponding to a hydra config.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        The main hydra configuration object.

    Returns
    -------
    PosteriorClass : benchflow.posteriors.Posterior
        A subclass (of class Posterior) corresponding to the appropriate
        algorithm.
    """
    import benchflow.posteriors

    classname = cfg.algorithm["class"]
    PosteriorClass = getattr(benchflow.posteriors, classname + "Posterior")
    return PosteriorClass


def cfg_to_namespace(cfg):
    """Generate a namespace for variables defined in a hydra config.

    Restricts ``eval`` statements to a concrete namespace, for safety. Allows
    users to import packages and define variables within the configuration of
    each algorithm and task.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        A sub-branch of a hydra config, containing a list ``imports`` of import
        statements and a dict ``vars`` of shorthand variables used in the
        configuration.

    Returns
    -------
    namespace : dict
        A dictionary containing the local modules and variables defined by the
        config.
    """
    namespace = {}

    for imp in cfg.get("imports", []):
        imp = [s.strip() for s in imp.split("as")]
        if len(imp) == 2:  # import-as
            namespace[imp[1]] = import_module(imp[0])
        else:
            namespace[imp[0]] = import_module(imp[0])
    namespace["benchflow"] = import_module("benchflow")

    for key, val in cfg.get("vars", {}).items():
        namespace[key] = eval(str(val), namespace)

    return namespace


def cfg_to_args(cfg, task=None):
    """Evaluate arguments and keywords arguments from a config.

    Restricts ``eval`` statements to a concrete namespace, for safety. Allows
    users to import packages and define variables within the configuration of
    each algorithm and task.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        A sub-branch of a hydra config, containing a list ``args`` of arguments
        and a dict ``kwargs`` of keyword arguments.

    Returns
    -------
    args : list
        A list containing the evaluated arguments.
    kwargs : dict
        A  dictionary containing the evaluated keyword arguments.
    """
    namespace = cfg_to_namespace(cfg["algorithm"])
    if task is not None:
        namespace["task"] = task

    args = []
    for arg in cfg.algorithm.get("args", []):
        args.append(eval(arg, namespace))

    kwargs = {}
    for key, val in cfg.algorithm.get("kwargs", {}).items():
        if isinstance(val, (dict, DictConfig)):
            kwargs[key] = dict(val)
        else:
            try:
                kwargs[key] = eval(str(val), namespace)
            except NameError:
                kwargs[key] = str(val)

    return args, kwargs


def cfg_to_task(cfg):
    """Find the Task corresponding to a hydra config.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        The main hydra configuration object.

    Returns
    -------
    task : benchflow.tasks.Task
        An instance of the task specified by the config.
    """
    import benchflow.tasks

    task_cfg = cfg.get("task", cfg)
    # (Assume cfg is top-level if key 'task' does not exist.)
    namespace = cfg_to_namespace(task_cfg)

    task_module = import_module("benchflow.tasks." + task_cfg["file_name"])
    task_classname = task_cfg.get("classname", task_cfg["file_name"].capitalize())
    task = getattr(task_module, task_classname)

    kwargs = {}
    for key, val in task_cfg.get("options", {}).items():
        try:
            kwargs[key] = eval(str(val), namespace)
        except NameError:
            kwargs[key] = val

    if task_cfg.get("simulate_noise"):
        task = benchflow.tasks.make_noisy(task)
    task = task(cfg=cfg, **kwargs)

    return task


def cfg_to_metrics(cfg):
    """Find the metrics corresponding to a hydra config.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        The main hydra configuration object.

    Returns
    -------
    metrics : dict
        A dictionary of ``name: callable`` pairs for each metric requested in
        the config.
    """
    import benchflow.metrics

    metrics = {}
    metric_list = cfg.get("metrics", [])
    if metric_list == "all":  # Get all public callables from module
        for (name, callabl) in getmembers(benchflow.metrics, isfunction):
            metrics[name] = callabl
    else:
        if type(metric_list) == str:  # Convert single metric to list
            metric_list = [metric_list]
        if type(metric_list) == ListConfig:  # Convert from omegaconf
            metric_list = OmegaConf.to_object(metric_list)
        if type(metric_list) != list:
            raise ValueError(
                "cfg.metrics must be a list, 'all' or the "
                + "string name of a single metric."
            )
        for name in metric_list:
            try:
                metric = getattr(benchflow.metrics, name)
                metrics[name] = metric
            except AttributeError:
                logging.warn(
                    f"Metric not {name} not found, will not be computed."
                )

    return metrics


def cfg_to_seed(cfg):
    """Generate a random seed as specified by the hydra config.

    If the top-level field ``unique_seed`` is ``True``, each configuration
    will be hashed with an initial seed (default 0) to produce a unique random
    seed. Useful if, for example, each algorithm should produce independent
    results, even given the 'same' seed.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        The main hydra configuration object.

    Returns
    -------
    seed : int
        The generated seed.
    """
    if cfg.get("unique_seed"):  # Hash the configuration itself with the seed:
        init_seed = cfg.get("seed", 0)
        return combine_seeds(init_seed, cfg)
    else:
        return cfg.get("seed")


def combine_seeds(s, t, n_digits=8):
    """Hash the strings of two objects to create a unique seed.

    Parameters
    ----------
    s : any (must support str(s))
        The first object.
    t : any (must support str(s))
        The second object.

    Returns
    -------
    seed : int
        The generated seed.
    """
    string = str(s) + "," + str(t)
    bits = bytes(string, "utf-8")
    return int(hashlib.sha256(bits).hexdigest()[0:n_digits], 16)
