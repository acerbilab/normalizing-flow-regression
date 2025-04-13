import json
import logging
import pickle
from pathlib import Path

import numpy as np
from omegaconf.dictconfig import DictConfig


def read_generated_initial_set(task, cfg: DictConfig):
    logging.info("Use provided initial set.")
    D = task.D
    initial_set = cfg.algorithm.initial_set
    base_dir = Path(__file__).parents[2]
    if isinstance(initial_set, DictConfig):
        # Initial sets are provided by specifying directory
        file_dir = base_dir / Path(initial_set.get("dir", "./initial_points"))
        if cfg.algorithm.get("budget"):
            budget = cfg.algorithm.get("budget")
            budget = eval(str(budget))
            file_dir = base_dir / Path(
                initial_set.get(
                    "json_dir",
                    f"./initial_points/{cfg.task.name}/{cfg.algorithm.init_source}",
                )
            )
            with open(file_dir / f"budget:{budget}.json", "r") as f:
                dataset = json.load(f)

            data, opt_results = combine_data_from_files(
                dataset["initial_sets"][f"{initial_set.id}"], base_dir=base_dir
            )
        else:
            include = initial_set.include
            if isinstance(include, str) and include[0] == "@":
                include = eval(include[1:])
            else:
                assert isinstance(include, list)
            exclude = initial_set.get("exclude")
            if exclude is None:
                exclude = []
            elif isinstance(exclude, int):
                exclude = [exclude]
            exclude = set(exclude)
            include = [p for p in include if p not in exclude]
            file_paths = []
            for p in include:
                file_path = file_dir / f"{p}/initial_train_set.pkl"
                if file_path.exists():
                    file_paths.append(file_path)
                else:
                    raise ValueError(f"{file_path} doesn't exist.")

            data, opt_results = combine_data_from_files(file_paths)
    else:
        # Initial set is a path to a file
        data_path = base_dir / cfg.algorithm.initial_set
        data = read_from_file(data_path)
    method = data["method"]
    if method == "Slice Sampling":
        raise NotImplementedError()
    x = data["X"]
    y = data["y"]
    log_likes = data["log_likes"]
    log_priors_orig = data["log_priors"]
    assert x.shape[1] == task.D
    assert np.allclose(log_likes + log_priors_orig, y)
    S_orig = data.get("S")
    # fun_evals_start = data["fun_evals"]  # it's typically larger than np.size(y), use fun_evals_start = np.size(y) for now
    threshold = nstd_threshold(
        cfg.algorithm.get("init_keep_factor", np.inf), D
    )
    if threshold <= 0:
        threshold = np.inf

    if task.is_noisy:
        # A conservative adjustment of threshold
        threshold += 2 * 1.96 * task.noise_sd
    logging.info(f"threshold: {threshold}")

    ymax = y.max()
    mask = ymax - y <= threshold
    if cfg.algorithm.get("half"):
        logging.info("cutting")
        mask = mask & (x[:, 0] < 0)
    x = x[mask]
    log_likes = log_likes[mask]
    log_priors_orig = log_priors_orig[mask]
    y = y[mask]
    if S_orig is not None:
        S_orig = S_orig[mask]
    logging.info(f"{np.sum(mask)}/{np.size(mask)} points are kept.")
    assert np.isfinite(x).all() and np.isfinite(y).all()
    if cfg.task.get("noisy", False):
        assert (
            S_orig is not None
        ), "S_orig need to be parsed. Typically it's cfg.task.options.noise_sd."
    if cfg.algorithm.get("cross_validate"):
        logging.info("Cross validation run.")
        # Keep top 20% points
        N_hpd = int(np.size(y) * 0.2)
        # Drop 20% points randomly from the rest
        N_drop = int(np.size(y) * 0.2)
        inds = np.argsort(y.flatten())[::-1][N_hpd:]
        inds = np.random.choice(inds, size=N_drop, replace=False)
        x = np.delete(x, inds, axis=0)
        y = np.delete(y, inds, axis=0)
        if S_orig is not None:
            S_orig = np.delete(S_orig, inds, axis=0)
    else:
        logging.info("Normal run on untrimmed data.")
    return x, y, S_orig


def save_to_file(data, file_path):
    """Save data to a file."""
    if file_path is None:
        return
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Data saved to {file_path}")


def read_from_file(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def combine_data_from_files(file_paths, base_dir=None):
    assert len(file_paths) > 0, "No file found."
    logging.info(f"Combining data from {len(file_paths)} files.")
    path_str = "\n".join(map(str, file_paths))
    logging.info(f"File paths: \n{path_str}")
    data = {}
    keys_to_concat = [
        "X",
        "y",
        "S",
        "log_likes",
        "log_priors",
        "log_likes_exact",
    ]
    optimization_results = []
    for file_path in file_paths:
        if isinstance(file_path, list):
            start = file_path[1]
            end = file_path[2]
            file_path = file_path[0]
            if base_dir is not None:
                file_path = base_dir / file_path
        else:
            start = 0
            end = -1
        data_cur = read_from_file(file_path)
        opt_cur = data_cur.get("optimization_results")
        if opt_cur is not None:
            optimization_results.append(opt_cur)
        for k, v in data_cur.items():
            if k in keys_to_concat:
                if k not in data:
                    data[k] = []
                data[k].append(v[start:end])
            elif k in ["fun_evals"]:
                if k not in data:
                    data[k] = 0
                if end == -1:
                    data[k] += v
                else:
                    data[k] += end - start
            else:
                if k not in data:
                    data[k] = []
                data[k].append(v)
    for k, v in data.items():
        if k in keys_to_concat:
            data[k] = np.concatenate(v, axis=0)
    method = set(data["method"])
    assert (
        len(method) == 1
    ), "Use initial sets with different methods are not supported yet."
    data["method"] = method.pop()
    if len(optimization_results) > 0:
        X_opts = []
        y_opts = []
        for opt_result in optimization_results:
            for result in opt_result:
                X_opts.append(result["x_opt"])
                y_opts.append(result["f_opt"])
        X_opts = np.stack(X_opts)
        y_opts = np.array(y_opts)
        # inds = np.argsort(y_opts)[::-1]
        # X_opts = X_opts[inds]
        # y_opts = y_opts[inds]
        return data, {"X_opts": X_opts, "y_opts": y_opts}
    return data, None


def nstd_threshold(n1, d):
    from scipy.special import erfc
    from scipy.stats import chi2

    """Ref: https://arxiv.org/abs/2211.02045"""
    return chi2.isf(erfc(n1 / np.sqrt(2)), d) / 2
