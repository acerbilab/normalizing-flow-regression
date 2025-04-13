import random
import time
from copy import deepcopy
from pathlib import Path

import cma
import numpy as np
import scipy.stats as scs
from tqdm.auto import tqdm

from benchflow.algorithms import Algorithm
from benchflow.function_logger import (
    BudgetExhaustedException,
    FunctionLogger,
    IdentityTransformer,
    ParameterTransformer,
)
from benchflow.utilities.cfg import cfg_to_seed, cfg_to_task
from benchflow.utilities.dataset import read_from_file, save_to_file


class GenerateInitialSet(Algorithm):
    """For generating initial train set."""

    def __init__(self, cfg):
        assert not cfg["task"].get(
            "transform_to_unconstrained_coordinates", False
        ), (
            "This algorithm computes and stores log priors in the original"
            "space so don't transform to unconstrained coordinates inside task"
            "object. The bound constraints will be handled properly here."
        )

        self.task = cfg_to_task(cfg)
        self.cfg = cfg
        if not cfg.algorithm["map_optimization"].get("original_space"):
            assert (
                cfg.algorithm.get("transform_to_unconstrained_coordinates")
                is True
            ), "The task should be transformed to unconstrained coordinates if one want to find MAP in unconstrained space."
        if cfg.algorithm.get("transform_to_unconstrained_coordinates"):
            self.parameter_transformer = ParameterTransformer(
                self.task.D,
                self.task.lb,
                self.task.ub,
                self.task.plb,
                self.task.pub,
            )
        else:
            self.parameter_transformer = IdentityTransformer()

        self.D = self.task.D
        self.lb = self.parameter_transformer(self.task.lb)
        self.ub = self.parameter_transformer(self.task.ub)
        self.plb = self.parameter_transformer(self.task.plb)
        self.pub = self.parameter_transformer(self.task.pub)

        if self.task.is_noisy:
            if cfg.algorithm.get("debugging"):
                # Record exact log likelihood
                import copy

                task_exact = copy.deepcopy(self.task)
                task_exact.is_noisy = False

                def log_density(theta):
                    log_likelihood_exact = super(
                        type(task_exact), task_exact
                    ).log_likelihood(theta)
                    noise, noise_est = self.task.noise_function(theta)
                    log_likelihood = log_likelihood_exact + noise
                    log_prior = self.task.log_prior(theta)
                    return (
                        log_likelihood,
                        log_prior,
                        noise_est,
                        log_likelihood_exact,
                    )

            else:

                def log_density(theta):
                    log_likelihood, noise_est = self.task.log_likelihood(theta)
                    log_prior = self.task.log_prior(theta)
                    return log_likelihood, log_prior, noise_est

            uncertainty_handling_level = 2
        else:

            def log_density(theta):
                log_likelihood = self.task.log_likelihood(theta)
                log_prior = self.task.log_prior(theta)
                return log_likelihood, log_prior

            uncertainty_handling_level = 0

        self.function_logger = FunctionLogger(
            fun=log_density,
            D=self.task.D,
            noise_flag=self.task.is_noisy,
            uncertainty_handling_level=uncertainty_handling_level,
            parameter_transformer=self.parameter_transformer,
        )

        assert cfg.algorithm["method"] in [
            "Slice Sampling",
            "CMA-ES",
            "PyBADS",
            "Mixed Samples",
        ], (
            "Unknown method: "
            + cfg.algorithm["method"]
            + ". Supported methods: "
            + ", ".join(
                ["Slice Sampling", "CMA-ES", "PyBADS", "Mixed Samples"]
            )
        )
        self.optimizers = ["CMA-ES", "PyBADS"]
        if cfg.algorithm["method"] in self.optimizers:
            if cfg.algorithm["map_optimization"].get("original_space"):
                print("Find MAP in the original constrained space")
                self.function_logger.return_original_space = True
            else:
                print("Find MAP in the transformed unconstrained space")
                self.function_logger.return_original_space = False

            self.N = cfg.algorithm["map_optimization"]["N_fun_evals"]
            self.function_logger.budget = self.N
            assert self.N >= 1
            # if self.N >= 100000:
            #     raise ValueError(
            #         "N_fun_evals is too large, which results in long waiting time."
            #     )

        if cfg.algorithm.get("data_save_path"):
            self.save_path = Path(cfg.algorithm["data_save_path"])
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.save_path = None

        # Fix random seed (if specified):
        if cfg.get("seed") is not None:
            self.seed = cfg_to_seed(cfg)
            random.seed(self.seed)
            np.random.seed(self.seed)

    def run(self):
        """Generate initial train set according to the configurations."""
        cfg = self.cfg["algorithm"]
        method = cfg["method"]
        self.start_time = time.time()
        print("Generating initial set...")
        if method == "Slice Sampling":
            data = self._slice_sampling()
        elif method == "Mixed Samples":
            data = self._generate_mixed_samples()
        elif method in self.optimizers:
            data = self._map_optimization(method)
        else:
            raise ValueError(f"Unknown method: {method}")
        data["runtime_total"] = time.time() - self.start_time
        print(f"Total runtime: {data['runtime_total']}")
        save_to_file(data, self.save_path)
        return data

    def _map_optimization(self, method="PyBADS"):
        tol_fun = self.cfg["algorithm"]["map_optimization"]["tol_fun"]
        # Uniform random samples in the plausible box
        # (in transformed space)
        # if self.N < 10:
        #     rand_count = self.N
        # else:
        #     rand_count = int(0.1 * self.N)
        rand_count = min(20 * self.D, self.N // 2)

        random_Xs = (
            np.random.rand(rand_count, self.D) * (self.pub - self.plb)
            + self.plb
        )
        try:
            prior_samples_orig = self.task.get_prior_samples(rand_count)
            prior_samples = self.parameter_transformer(prior_samples_orig)
            random_Xs = np.concatenate([random_Xs, prior_samples], axis=0)
            print("Prior samples are used.")
        except (NotImplementedError, RuntimeError):
            print("Prior samples are not available.")
            if np.all(np.isfinite(self.task.lb)) and np.all(
                np.isfinite(self.task.ub)
            ):
                samples_bound = (
                    np.random.rand(rand_count, self.D)
                    * (self.task.ub - self.task.lb)
                    + self.task.lb
                )
                samples_bound = self.parameter_transformer(samples_bound)
                random_Xs = np.concatenate([random_Xs, samples_bound], axis=0)
                print("Uniform samples in the bound constraints are used.")

        # Evaluate function and cache values
        for x in random_Xs:
            self.function_logger(x)

        X_start = self.function_logger.X[self.function_logger.X_flag, :]
        y_start = self.function_logger.y[self.function_logger.X_flag]

        inds = np.argsort(y_start, axis=None)[::-1]

        # Do MAP optimizations multiple times until the budget is exhausted
        # and an exception will be raised.
        i = 0
        converged = False  # whether the first optimization run is converged
        optimization_results = []
        try:
            while True:
                assert (
                    i < rand_count
                ), "not enough starting points to use for optimization"
                ind = inds[i]
                x0 = X_start[ind]
                i += 1
                result = {}
                if method == "CMA-ES":

                    def fun(x):
                        return -self.function_logger(x)[0]

                    # It's advised to scale the inputs
                    fun_scaled = cma.ScaleCoordinates(
                        fun,
                        multipliers=self.pub.squeeze() - self.plb.squeeze(),
                    )
                    lb = fun_scaled.inverse(self.lb.squeeze())
                    ub = fun_scaled.inverse(self.ub.squeeze())
                    x0 = fun_scaled.inverse(x0)
                    sigma0 = 0.25

                    tol_x = self.cfg["algorithm"]["map_optimization"].get(
                        "tol_x", 0.01
                    )

                    cma_options = {
                        # "verbose": -9,
                        "tolx": tol_x,
                        "tolfun": tol_fun,
                        "maxfevals": self.N,  # simply use self.N here
                        "bounds": [lb, ub],
                        "seed": np.nan,
                    }
                    if self.function_logger.noise_flag:
                        noise_handler = cma.NoiseHandler(self.D)
                    else:
                        noise_handler = None

                    res = cma.fmin(
                        fun_scaled,
                        x0,
                        sigma0,
                        options=cma_options,
                        noise_handler=noise_handler,
                    )
                    x_opt, f_opt = res[0], -res[1]
                    result["termination"] = dict(res[-3])
                elif method == "PyBADS":
                    from pybads.bads import BADS

                    bads_options = {"tol_fun": tol_fun}
                    if self.function_logger.noise_flag:
                        bads_options["uncertainty_handling"] = True
                        bads_options["specify_target_noise"] = True

                        def f_objective(x):
                            res = self.function_logger(x)
                            return -res[0], res[1]

                    else:

                        def f_objective(x):
                            return -self.function_logger(x)[0]

                    bads = BADS(
                        f_objective,
                        np.atleast_2d(x0),
                        self.lb,
                        self.ub,
                        self.plb,
                        self.pub,
                        options=bads_options,
                    )
                    res = bads.optimize()
                    x_opt, f_opt = res["x"], -res["fval"]
                    result["termination"] = res["message"]
                # elif method == "test":
                # DON'T WORK WELL
                # from noisyopt import minimizeCompass, minimizeSPSA
                # result = minimizeCompass(lambda x: -self.function_logger(np.array(x))[0], x0=x0, deltatol=tol_fun, paired=False, disp=True)
                # result = minimizeSPSA(lambda x: -self.function_logger(np.array(x))[0], x0=x0, paired=False, disp=True)
                # from skopt import gp_minimize
                # result = gp_minimize(
                #     lambda x: -self.function_logger(np.array(x))[0],
                #     [(-100., 100.) for i in range(self.D)],
                #     n_calls=self.N,
                #     verbose=True,
                #     x0 = x0.tolist(),
                #     noise=self.task.noise_sd,
                # )
                # x_opt, f_opt = result.x, -result.fun
                else:
                    raise NotImplementedError(f"{method} is not supported.")
                converged = True
                x_opt = self.function_logger.parameter_transformer.inverse(
                    x_opt
                )
                result.update({"x_opt": x_opt, "f_opt": f_opt})
                if self.task.is_noisy and self.cfg.algorithm.get("debugging"):
                    import copy

                    task_exact = copy.deepcopy(self.task)
                    task_exact.is_noisy = False
                    f_opt_exact = super(
                        type(task_exact), task_exact
                    ).log_likelihood(x_opt) + super(
                        type(task_exact), task_exact
                    ).log_prior(x_opt)
                    result["f_opt_exact"] = f_opt_exact.item()
                optimization_results.append(result)
                if self.cfg.algorithm["map_optimization"].get(
                    "stop_after_first"
                ):
                    print("The MAP optimization is finished, break.")
                    break
        except BudgetExhaustedException:
            print("Function evaluation budget is exhausted.")

        print(f"MAP optimization ran for {i} times.")

        # Get data
        data = _retrieve_data_from_function_logger(
            self.function_logger, self.cfg.algorithm.get("debugging")
        )
        data.update(
            {
                "method": method,
                "cfg": self.cfg,
                "fun_evals": self.function_logger.func_count,
                "optimization_results": optimization_results,
            }
        )
        for result in optimization_results:
            print(result["f_opt"])
        print(
            "Number of function evaluations: ", self.function_logger.func_count
        )
        data["converged"] = True
        data["num_MAP_runs"] = i
        if not converged:
            data["converged"] = False
            data["runtime_total"] = time.time() - self.start_time
            save_to_file(data, self.save_path)
            if self.cfg.algorithm["map_optimization"].get(
                "raise_exception_if_not_converged", True
            ):
                raise Exception(
                    f"MAP optimization failed since first optimization run is not converged yet when budget is exhausted. Try to increase the budget or decrease `tol_fun`. The data is still saved to {self.save_path}."
                )
        return data

    def _slice_sampling(self):
        try:
            from gpyreg.slice_sample import SliceSampler
        except ImportError:
            raise ImportError("Please install gpyreg to use slice sampling.")

        assert (
            not self.task.is_noisy
        ), "Slice sampling only works for non-noisy tasks."
        cfg = self.cfg["algorithm"]
        N_chains = cfg["slice_sampling"]["N_chains"]
        N_samples_per_chain = cfg["slice_sampling"].get("N_samples_per_chain")

        data = {
            "method": "Slice Sampling",
            "cfg": self.cfg,
            "fun_evals": [],
            "sampling_results": [],
            "X_evals": [],
            "y_evals": [],
            "log_likes_evals": [],
            "log_priors_evals": [],
        }
        for i in range(N_chains):
            x0 = self.task.x0(randomize=True, sz=1).flatten()
            widths = np.squeeze(self.pub - self.plb)

            function_logger = deepcopy(self.function_logger)
            slicer = SliceSampler(
                lambda x: function_logger(x)[0],
                x0,
                widths,
                np.squeeze(self.lb),
                np.squeeze(self.ub),
            )

            sampling_result = slicer.sample(N_samples_per_chain, burn=0)
            data["sampling_results"].append(sampling_result)

            data["fun_evals"].append(function_logger.func_count)
            tmp_data = _retrieve_data_from_function_logger(function_logger)
            data["X_evals"].append(tmp_data["X"])
            data["y_evals"].append(tmp_data["y"])
            data["log_likes_evals"].append(tmp_data["log_likes"])
            data["log_priors_evals"].append(tmp_data["log_priors"])

        return data

    def _generate_mixed_samples(self):
        cfg = self.cfg["algorithm"]["mixed_samples"]
        assert (
            sum(cfg["fraction"].values()) == 1
        ), "The sum of fractions should be 1."
        N = cfg["N_samples"]
        assert N > 0, "The number of samples should be positive."

        N_per_source = {}  # number of samples per source
        N_left = N
        for key, fraction in cfg["fraction"].items():
            assert (
                fraction >= 0 and fraction <= 1
            ), "The fraction should be in [0, 1]."
            n_points = int(N * fraction)
            if n_points >= 0:
                N_per_source[key] = n_points
                N_left -= n_points
        if N_left > 0:
            N_per_source[list(N_per_source.keys())[0]] += N_left

        mixed_samples = []
        posterior_samples = None
        for key, n_points in N_per_source.items():
            if n_points == 0:
                continue
            if key == "prior":
                prior_samples = self.task.get_prior_samples(n_points)
                mixed_samples.append(prior_samples)
                print(f"{n_points} prior samples are used.")
            elif key == "plausible":
                points = np.random.uniform(
                    self.task.plb, self.task.pub, size=(n_points, self.task.D)
                )
                mixed_samples.append(points)
                print(
                    f"{n_points} uniform samples in the plausible box are used."
                )
            elif key == "bound":
                if np.all(np.isfinite(self.task.lb)) and np.all(
                    np.isfinite(self.task.ub)
                ):
                    lb = self.task.lb
                    ub = self.task.ub
                    print(
                        f"Bounds are finite. {n_points} uniform samples in the bounds are used."
                    )
                else:
                    lb_candidate = self.task.plb - 10 * (
                        self.task.pub - self.task.plb
                    )
                    ub_candidate = self.task.pub + 10 * (
                        self.task.pub - self.task.plb
                    )
                    inds = ~np.isfinite(self.task.lb)
                    lb = self.task.lb.copy()
                    lb[inds] = lb_candidate[inds]
                    inds = ~np.isfinite(self.task.ub)
                    ub = self.task.ub.copy()
                    ub[inds] = ub_candidate[inds]
                    assert np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))
                    print(
                        f"Some bound constraints are infinite, so the expanded plausible ranges are used for the infinite axes. {n_points} samples are used."
                    )
                points = np.random.uniform(
                    lb, ub, size=(n_points, self.task.D)
                )
                mixed_samples.append(points)
                print(f"{n_points} uniform samples in the bounds are used.")
            elif key == "posterior":
                posterior_samples = self.task.get_posterior_samples(n_points)
                mixed_samples.append(posterior_samples)
                print(f"{n_points} posterior samples are used.")
            elif key == "broad_posterior":
                assert (
                    posterior_samples is not None
                ), "broad_posterior requires posterior samples."
                settings = cfg["broaden_posterior_settings"]
                # Sampling is done in the unconstrained space
                posterior_samples_uncons = self.parameter_transformer(
                    posterior_samples
                )

                # Compute the mean and covariance of the posterior samples
                mean = np.mean(posterior_samples_uncons, axis=0)
                cov_mat = np.cov(posterior_samples_uncons, rowvar=False)

                if settings.get("diagonal"):
                    cov_mat = np.diag(np.diag(cov_mat))
                if settings.get("isotropic"):
                    assert settings.get(
                        "diagonal", True
                    ), "`diagonal` can't be false when `isotropic=True`"
                    cov_mat = np.max(np.diag(cov_mat)) * np.eyes(self.task.D)
                if settings["method"] == "t-distribution":
                    broaden_posterior = scs.multivariate_t(
                        mean, cov_mat, df=settings["t-distribution"]["df"]
                    )
                    points = broaden_posterior.rvs(n_points)
                else:
                    raise NotImplementedError()
                # Transform to original constrained space
                points = self.parameter_transformer.inverse(points)
                mixed_samples.append(points)
                print(f"{n_points} broadened posterior samples are used.")
            else:
                raise ValueError()
        mixed_samples = [np.atleast_2d(samples) for samples in mixed_samples]
        mixed_samples = np.concatenate(mixed_samples)

        # Transform to unconstrained space since the function logger expects
        # unconstrained parameters.
        mixed_samples_uncons = self.parameter_transformer(mixed_samples)
        # Evaluate function and cache values
        for i, x in enumerate(tqdm(mixed_samples_uncons)):
            self.function_logger(x)

        data = _retrieve_data_from_function_logger(self.function_logger)
        data.update(
            {
                "method": "Mixed Samples",
                "cfg": self.cfg,
                "fun_evals": self.function_logger.func_count,
            }
        )
        return data


def _retrieve_data_from_function_logger(function_logger, debugging=False):
    log_likes = function_logger.log_likes[function_logger.X_flag].flatten()
    log_priors = function_logger.log_priors_orig[
        function_logger.X_flag
    ].flatten()
    y = function_logger.y_orig[function_logger.X_flag].flatten()
    X = function_logger.X_orig[function_logger.X_flag]
    assert np.allclose(log_likes + log_priors, y)
    data = {"X": X, "y": y, "log_likes": log_likes, "log_priors": log_priors}
    if function_logger.noise_flag:
        S = function_logger.S_orig[function_logger.X_flag]
        data["S"] = S
        if debugging:
            data["log_likes_exact"] = function_logger.log_likes_exact[
                function_logger.X_flag
            ].flatten()
    return data


def get_hpd(X: np.ndarray, y: np.ndarray, hpd_frac: float = 0.8):
    """
    Get high-posterior density dataset.

    Parameters
    ==========
    X : ndarray, shape (N, D)
        The training points.
    y : ndarray, shape (N, 1)
        The training targets.
    hpd_frac : float
        The portion of the training set to consider, by default 0.8.

    Returns
    =======
    hpd_X : ndarray
        High-posterior density training points.
    hpd_y : ndarray
        High-posterior density training targets.
    hpd_range : ndarray, shape (D,)
        The range of values of hpd_X in each dimension.
    indices : ndarray
        The indices of the points returned with respect to the original data
        being passed to the function.
    """

    N, D = X.shape

    # Subsample high posterior density dataset.
    # Sort by descending order, not ascending.
    order = np.argsort(y, axis=None)[::-1]
    hpd_N = max(round(hpd_frac * N), 1)
    indices = order[0:hpd_N]
    hpd_X = X[indices]
    hpd_y = y[indices]

    if hpd_N > 0:
        hpd_range = np.max(hpd_X, axis=0) - np.min(hpd_X, axis=0)
    else:
        hpd_range = np.full((D), np.NaN)

    return hpd_X, hpd_y, hpd_range, indices
