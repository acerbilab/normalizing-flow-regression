import copy
import gc
import logging
import os
import pathlib
import sys
import time
from pathlib import Path

import corner
import dill
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from nflows.distributions.normal import StandardNormal
from nfr.training_wrappers import FlowEnsembles

from .flow_helpers import create_transforms
from .function_logger import FunctionLogger
from .iteration_history import IterationHistory
from .nfr_helpers import (
    _get_training_data,
    get_initial_samples,
    warp_input,
)
from .options import Options
from .parameter_transformer import ParameterTransformer
from .utils import (
    get_torch_model_size,
    plot_with_extra_data,
    wandb_log,
)

torch.set_default_dtype(torch.float64)


class NFR:
    def __init__(
        self,
        x0: np.ndarray = None,
        lower_bounds: np.ndarray = None,
        upper_bounds: np.ndarray = None,
        plausible_lower_bounds: np.ndarray = None,
        plausible_upper_bounds: np.ndarray = None,
        options: dict = None,
        task_name: str = None,
        **kwargs,
    ):
        self.task_name = task_name
        self.bench_seed = kwargs.get("bench_seed")

        # Initialize variables and algorithm structures
        if x0 is None:
            if (
                plausible_lower_bounds is None
                or plausible_upper_bounds is None
            ):
                raise ValueError(
                    """UnknownDims If no starting point is
                 provided, PLB and PUB need to be specified."""
                )
            else:
                x0 = np.full((plausible_lower_bounds.shape), np.NaN)

        if x0.ndim == 1:
            logging.warning("Reshaping x0 to row vector.")
            x0 = x0.reshape((1, -1))

        self.D = x0.shape[1]
        # load basic and advanced options and validate the names
        file_path = os.path.dirname(os.path.realpath(__file__))
        basic_path = file_path + "/option_configs/basic_options.ini"
        self.options = Options(
            basic_path,
            evaluation_parameters={"D": self.D},
            user_options=options,
        )

        advanced_path = file_path + "/option_configs/advanced_options.ini"
        self.options.load_options_file(
            advanced_path,
            evaluation_parameters={"D": self.D},
        )
        self.options.validate_option_names([basic_path, advanced_path])

        # Check consistency
        if self.options["init_design"] == "provided":
            provided_sample_count = x0.shape[0]
            assert provided_sample_count >= 10, (
                "Provided initial points are supposed to be at least 10."
            )
        else:
            raise NotImplementedError

        pathlib.Path(self.options["experiment_folder"]).mkdir(
            parents=True, exist_ok=True
        )
        print(
            f"Outputs will be written to folder: {self.options['experiment_folder']}"
        )

        # Create an initial logger for initialization messages:
        self.logger = self._init_logger("_init")

        # Empty LB and UB are Infs
        if lower_bounds is None:
            lower_bounds = np.ones((1, self.D)) * -np.inf

        if upper_bounds is None:
            upper_bounds = np.ones((1, self.D)) * np.inf

        # Check/fix boundaries and starting points
        (
            self.x0,
            self.lower_bounds,
            self.upper_bounds,
            self.plausible_lower_bounds,
            self.plausible_upper_bounds,
        ) = self._bounds_check(
            x0,
            lower_bounds,
            upper_bounds,
            plausible_lower_bounds,
            plausible_upper_bounds,
        )

        # starting point
        if not np.all(np.isfinite(self.x0)):
            # print('Initial starting point is invalid or not provided.
            # Starting from center of plausible region.\n');
            self.x0 = 0.5 * (
                self.plausible_lower_bounds + self.plausible_upper_bounds
            )

        # Initialize transformation to unbounded parameters
        if self.options.get("turn_off_unbounded_transform", False):
            self.parameter_transformer = ParameterTransformer(
                self.D,
            )  # Identity transform
        else:
            self.parameter_transformer = ParameterTransformer(
                self.D,
                self.lower_bounds,
                self.upper_bounds,
                self.plausible_lower_bounds,
                self.plausible_upper_bounds,
                transform_type=self.options["bounded_transform"],
            )

        # Optimization starts from iteration 0
        self.iteration = -1
        # Whether the optimization has finished
        self.is_finished = False

        self.optim_state = self._init_optim_state()

        self.function_logger = FunctionLogger(
            fun=None,  # not needed
            D=self.D,
            noise_flag=self.optim_state.get("uncertainty_handling_level") > 0,
            uncertainty_handling_level=self.optim_state.get(
                "uncertainty_handling_level"
            ),
            cache_size=self.options.get("cache_size"),
            parameter_transformer=self.parameter_transformer,
        )

        # Create surrogate model
        if torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device("cuda")
        else:
            self.cuda = False
            self.device = torch.device("cpu")
        self.logger.debug(f"Using device {self.device}")
        transforms_options = dict(self.options)
        transforms_options["D"] = self.D
        if transforms_options["activation_fn"] == "relu":
            transforms_options["activation_fn"] = F.relu
        elif transforms_options["activation_fn"] == "celu":
            transforms_options["activation_fn"] = F.celu
        else:
            raise NotImplementedError

        assert self.options["n_ensembles"] == 1
        n_ensembles = self.options["n_ensembles"]
        transforms = [
            create_transforms(transforms_options) for i in range(n_ensembles)
        ]
        base_dists = [
            StandardNormal(shape=[self.D]) for i in range(n_ensembles)
        ]
        self.surrogate = FlowEnsembles(
            base_dists,
            transforms,
            options=self.options,
            D=self.D,
        )
        task_params_bounded = (
            np.isfinite(self.lower_bounds).any()
            or np.isfinite(self.upper_bounds).any()
        )
        if (
            self.options.get("turn_off_unbounded_transform")
            and task_params_bounded
        ):
            self.surrogate.bounds_info = {
                "need_rejection_sampling": True,
                "lb": self.lower_bounds,
                "ub": self.upper_bounds,
            }

        self.logger.debug(
            "Net size: {0[0]:.2f} MB, {0[1]} params, {0[2]} buffers.".format(
                get_torch_model_size(self.surrogate)
            )
        )

        self.iteration_history = IterationHistory(
            [
                "surrogate",
                "function_logger",
                "optim_state",
                "random_state",
                "function_logger.beta",
            ]
        )
        self.random_state = np.random.get_state()
        self.gp_model = None
        self.fig_dir = os.path.join(
            self.options["experiment_folder"],
            "figure",
        )
        pathlib.Path(self.fig_dir).mkdir(parents=True, exist_ok=True)

        # Get and record initial observations
        self.function_logger, self.optim_state = get_initial_samples(
            self.optim_state,
            self.function_logger,
            self.logger,
            self.options,
        )

        x0_trans = self.parameter_transformer(self.x0)
        self.logger.info(
            f"x0 range (transformed): "
            f"min={x0_trans.min(0)}, max={x0_trans.max(0)}"
        )
        self.logger.info(
            f"x0 range (original): {self.x0.min(0)}, {self.x0.max(0)}"
        )

    def _set_up_p0_and_input_space(self, debug=False):
        """The behavior depends on "init_design" option. Currently only
        support init_design=provided.

        init_design=provided: It uses user provided initial set to estimate a
        mulitivariate p0 and update parameter_transformer such that p0 is
        transformed to a standard normal (whitening transform).
        """

        if self.options.get("init_design") in ["provided"]:
            X_train, Y_train, S_train, S_train_raw = _get_training_data(
                self.function_logger,
                self.options,
                add_noise=self.options.get("init_p0_add_noise", 0.0),
                torch_device=self.device,
                return_std=True,
            )[:4]
            self.logger.debug("Density estimation is used for p0.")
            weighted_mean, weighted_cov = self._estimate_mvn(
                X_train, Y_train, S_train_raw
            )

            self.p0_info = "multivariate_normal"
            self.p0 = torch.distributions.MultivariateNormal(
                weighted_mean.detach(), weighted_cov.detach()
            )  # Defined in unconstrained space
            self.p0_parameter_transformer = copy.deepcopy(
                self.parameter_transformer
            )

            def log_p0_fun(x_orig):
                """Compute log probability in the original constrained space."""
                x = self.p0_parameter_transformer(x_orig)
                log_abs_det = (
                    self.p0_parameter_transformer.log_abs_det_jacobian(x)
                ).squeeze()
                x_torch = torch.from_numpy(x).to(self.device)
                log_p = (
                    self.p0.log_prob(x_torch).detach().cpu().numpy().squeeze()
                )
                log_p -= log_abs_det
                log_p = np.atleast_1d(log_p)
                assert log_p.ndim == 1
                return log_p[:, None]

            self.log_p0_fun = log_p0_fun
            p0_lnZ = 0

            p0_cov = weighted_cov.detach().cpu().numpy()
            p0_mean = weighted_mean.detach().cpu().numpy()
        else:
            raise NotImplementedError

        # Initialize lnZ with p0_lnZ
        if self.options.get("annealed_target"):
            self.surrogate.set_lnZ(p0_lnZ)

        # Set log_p0_fun in function logger
        self.function_logger.log_p0_fun = self.log_p0_fun

        # Whether p0 need to be fitted by the flow in the first iteration
        if self.p0_info == "multivariate_normal":
            # the input space is warped such that p0 is a standard normal (base distribution of the flow)
            # TODO: This implementation is unnecessarily complex (lazily taking warping function from pyvbmc). A simpler approach would be to set the base distribution of the flow directly.
            if self.options.get("warp_rotoscaling"):
                self.logger.debug("Warping input space with roto-scaling")
                (
                    self.parameter_transformer,
                    self.optim_state,
                    self.function_logger,
                ) = warp_input(
                    p0_mean,
                    p0_cov,
                    self.optim_state,
                    self.function_logger,
                    self.parameter_transformer,
                    self.options,
                )
            self.optim_state["fit_initial_flow"] = (
                True  # Still let flow fit p0 in the first iteration
            )
        else:
            self.logger.warning(
                "p0 is not a multivariate normal. The priors for the flow may not be appropriate."
            )
            self.optim_state["fit_initial_flow"] = True

        if self.options.get("plot"):
            samples = self.p0.sample(torch.Size([10000]))
            samples = samples.detach().cpu().numpy()
            fig = corner.corner(samples[:, :])
            D = samples.shape[1]
            X = X_train.detach().cpu().numpy()
            axes = np.array(fig.axes).reshape((D, D))

            for r in range(1, D):
                for c in range(D - 1):
                    if r > c:
                        axes[r, c].scatter(
                            X[:, c],
                            X[:, r],
                            s=2,
                            c="red",
                            marker="x",
                            # alpha=0.5
                        )
            fig.savefig(
                os.path.join(
                    self.fig_dir,
                    "p0_samples(p0 space).png",
                )
            )
        if debug:
            return X_train, Y_train, S_train, S_train_raw

    def _estimate_mvn(self, x_train, y_train, S_train_raw):
        y_train = y_train.flatten()
        S_train_raw = S_train_raw.flatten()
        if self.options.get("init_p0_threshold", False):
            # Use a threshold to select the top proportion of the data
            delta_thresh = self.options["init_p0_threshold_value"]
            y_lcb_max = (y_train - 1.96 * S_train_raw).max()
            lb = y_lcb_max - delta_thresh
            inds = y_train >= lb
            Np = inds.sum()
            self.logger.debug(
                f"Points with log density larger than {y_lcb_max} - {delta_thresh} = {lb} is used for estimating p0."
            )
        else:
            top_proportion = self.options.get("p0_top_proportion", 0.5)
            self.logger.debug(
                f"A multivariate normal calculated with top {top_proportion * 100}% in terms of log density is used as p0."
            )
            Np = max(int(x_train.shape[0] * top_proportion), 1)
            vals, inds = torch.topk(y_train, Np, largest=True, sorted=True)
        self.logger.debug(f"Number of points used for p0 estimation: {Np}")
        x_train = x_train[inds]
        y_train = y_train[inds]

        if self.options.get("init_p0_reweight", False):
            unnormalized_weights = np.log(Np + 1 / 2) - torch.log(
                torch.linspace(1, Np, Np)
            )
            weights = unnormalized_weights / torch.sum(unnormalized_weights)
        else:
            weights = 1 / torch.ones(Np)
        weights = weights.to(self.device)
        weighted_mean = torch.mean(weights * x_train.T, 1)
        weighted_cov = torch.cov(x_train.T, aweights=weights, correction=0)
        weighted_cov.diagonal().copy_(
            torch.clip(weighted_cov.diag(), min=1e-3)
        )
        if self.options.get("init_p0_diag", True):
            # Use only the diagonal from covariance matrix
            weighted_cov = weighted_cov.diag().diag()
        try:
            torch.linalg.cholesky(weighted_cov)
        except torch._C._LinAlgError:
            weighted_cov = weighted_cov.diag().diag()
        return weighted_mean, weighted_cov

    def optimize(self, steps=None):
        """If steps is not None, then optimize for the given steps of iterations from current state."""
        # Initialize main logger with potentially new options:
        self.logger = self._init_logger()

        _ = self._init_logger("_debug")

        if self.iteration == -1:
            if self.optim_state["uncertainty_handling_level"] > 0:
                self.logger.info(
                    "Beginning optimization assuming NOISY "
                    "observations of the log-likelihood."
                )
            else:
                self.logger.info(
                    "Beginning optimization assuming EXACT "
                    "observations of the log-likelihood."
                )

            self._set_up_p0_and_input_space()

        if steps is not None:
            assert steps >= 1
        step = 0

        # Avoid slow corner plotting by only plotting first few dimensions
        if self.D > 6:
            self.subspace = 5
        else:
            self.subspace = self.D

        while not self.is_finished:
            self.iteration += 1
            step += 1
            iteration_values = {}
            self.logger.debug(f"Iteration {self.iteration}")
            self.optim_state["iter"] = self.iteration
            self.optim_state["timing"] = {}
            self.surrogate.parameter_transformer = self.parameter_transformer

            # Set inverse temperature beta, the observations will be updated inside function_logger once beta changes
            if self.options.get("annealed_target"):
                if self.options.get("target_annealing_schedule") == "linear":
                    beta = np.maximum(
                        self.iteration / (self.options["num_warmup"] - 1), 0.0
                    )
                elif (
                    self.options.get("target_annealing_schedule")
                    == "quadratic"
                ):
                    beta = (
                        np.maximum(
                            self.iteration / (self.options["num_warmup"] - 1),
                            0.0,
                        )
                        ** 2
                    )
                else:
                    raise NotImplementedError
                beta = np.minimum(beta, 1.0)
            else:
                beta = 1.0
            self.function_logger.beta = beta
            wandb_log({"beta": beta})
            # Get the observations
            (
                x_train,
                y_train,
                s2_train,
                s2_train_raw,
                t_train,
            ) = _get_training_data(self.function_logger, self.options)
            X_train = torch.from_numpy(x_train).to(self.device)
            Y_train = torch.from_numpy(y_train).to(self.device)
            S_train = torch.from_numpy(np.sqrt(s2_train)).to(self.device)
            S_train_raw = torch.from_numpy(np.sqrt(s2_train_raw)).to(
                self.device
            )
            N_train = X_train.shape[0]

            ## Start flow training
            self.logger.debug("Start training the flow..")
            tic = time.time()
            if self.iteration == 0 and self.optim_state["fit_initial_flow"]:
                # Fit the initial flow with density estimation or variational inference
                self.logger.debug("Initial flow fitting")
                if self.options.get("fit_initial_flow_method") == "base_dist":
                    self.logger.debug("Set the flow parameters close to zero.")
                    for param in self.surrogate.parameters():
                        param.data *= 0.001
                else:
                    raise NotImplementedError

                if self.options.get("plot"):
                    flow_samples = self.surrogate.sample(10000)
                    flow_samples_orig = self.parameter_transformer.inverse(
                        flow_samples
                    )
                    fig = corner.corner(flow_samples_orig[:, : self.subspace])
                    fig.savefig(
                        os.path.join(
                            self.fig_dir,
                            "initial_flow_fit.png",
                        )
                    )

            if self.iteration == 0:
                debug_plot_iteration_0(self, X_train)

            ## Fit the flow via regression
            # Fit lnZ -> Fit flow parameters + lnZ
            self.logger.debug("Flow regression")
            self.logger.debug(f"{X_train.shape[0]} training points.")

            if self.iteration == 0 and self.options.get("annealed_target"):
                self.logger.debug(
                    "lnZ optimization is disabled for iteration 0 since the ground truth value is zero."
                )
                self.surrogate.optimize_lnZ_flag = False
            else:
                self.surrogate.optimize_lnZ_flag = self.options.get(
                    "optimize_lnZ", False
                )

            self.surrogate.fit(
                X_train,
                Y_train,
                S_train,
                S_train_raw,
                self.options["flow_optimizer"],
            )
            self.optim_state["timing"]["flow_training"] = time.time() - tic
            self.logger.debug(
                f"Flow training: {self.optim_state['timing']['flow_training']:.2f} seconds"
            )

            with torch.no_grad():
                assert isinstance(self.surrogate, FlowEnsembles)
                loss = []
                self.surrogate.update_variables(
                    X_train, Y_train, S_train, S_train_raw
                )
                for i in range(self.surrogate.n_ensembles):
                    loss.append(
                        self.surrogate.compute_loss(
                            X_train,
                            Y_train,
                            S_train,
                            S_train_raw,
                            i,
                            logger=self.logger,
                        )
                    )

            self.logger.debug(
                f"estimated lnZ = {self.surrogate.lnZ.detach().cpu().squeeze().numpy()} , function_logger.beta = {self.function_logger.beta:.3f}"
            )
            self.optim_state["lnZ"] = self.surrogate.lnZ.data

            # Check termination conditions
            (
                self.is_finished,
                termination_message,
                success_flag,
            ) = self._check_termination_conditions()

            ## Iteration plot
            self.save_surrogate_samples()
            tic = time.time()
            if self.options.get("plot"):
                if self.is_finished:
                    # Plot all dimensions
                    self.subspace = self.D
                self.plot_iteration()
            self.optim_state["timing"]["plot"] = time.time() - tic
            self.logger.debug(
                f"Plot: {self.optim_state['timing']['plot']:.2f} seconds"
            )

            self.random_state = np.random.get_state()
            iteration_values.update(
                {
                    "optim_state": self.optim_state,
                    "random_state": self.random_state,
                    "function_logger.beta": self.function_logger.beta,
                }
            )
            # Record all useful stats
            self.iteration_history.record_iteration(
                iteration_values,
                self.iteration,
            )

            plt.close("all")
            gc.collect()
            torch.cuda.empty_cache()

            if steps is not None:
                self.is_finished = False
                if step >= steps:
                    break

        lnZ_estimated = torch.mean(self.surrogate.lnZ)
        lnZ_estimated = lnZ_estimated.cpu().detach().numpy()
        self.surrogate.parameter_transformer = self.parameter_transformer
        # try:
        #     self.save_surrogate_samples(
        #         Path(self.options["experiment_folder"])
        #         / "posterior_samples/samples_final.csv",
        #         10000,
        #         save_density=True,
        #     )
        # except Exception as e:
        #     self.logger.info(str(e))
        final_result_dict = self.surrogate.get_info()
        return {
            "final_surrogate": self.surrogate,
            "final_lml": lnZ_estimated,
            "final_result_dict": final_result_dict,
        }

    def _check_termination_conditions(self):
        """
        Private method to determine the status of termination conditions.
        """
        is_finished_flag = False
        termination_message = ""
        success_flag = False
        output_dict = dict()
        iteration = self.optim_state.get("iter")
        if iteration + 1 >= self.options.get("nfr_iter"):
            is_finished_flag = True
            termination_message = (
                "Inference terminated: reached maximum number "
                + "of iterations options.maxiter."
            )

        return (
            is_finished_flag,
            termination_message,
            success_flag,
        )

    def _bounds_check(
        self,
        x0: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        plausible_lower_bounds: np.ndarray = None,
        plausible_upper_bounds: np.ndarray = None,
    ):
        """
        Private function to do the initial check of the bounds.
        """

        N0, D = x0.shape

        if plausible_lower_bounds is None or plausible_upper_bounds is None:
            if N0 > 1:
                # TODO: estimate via top x% of points
                self.logger.warning(
                    "PLB and/or PUB not specified. Estimating"
                    "plausible bounds from starting set X0..."
                )
                width = x0.max(0) - x0.min(0)
                if plausible_lower_bounds is None:
                    plausible_lower_bounds = x0.min(0) - width / N0
                    plausible_lower_bounds = np.maximum(
                        plausible_lower_bounds, lower_bounds
                    )
                if plausible_upper_bounds is None:
                    plausible_upper_bounds = x0.max(0) + width / N0
                    plausible_upper_bounds = np.minimum(
                        plausible_upper_bounds, upper_bounds
                    )

                idx = plausible_lower_bounds == plausible_upper_bounds
                if np.any(idx):
                    plausible_lower_bounds[idx] = lower_bounds[idx]
                    plausible_upper_bounds[idx] = upper_bounds[idx]
                    self.logger.warning(
                        "pbInitFailed: Some plausible bounds could not be "
                        "determined from starting set. Using hard upper/lower"
                        " bounds for those instead."
                    )
            else:
                self.logger.warning(
                    "pbUnspecified: Plausible lower/upper bounds PLB and"
                    "/or PUB not specified and X0 is not a valid starting set. "
                    "Using hard upper/lower bounds instead."
                )
                if plausible_lower_bounds is None:
                    plausible_lower_bounds = np.copy(lower_bounds)
                if plausible_upper_bounds is None:
                    plausible_upper_bounds = np.copy(upper_bounds)

        # Try to reshape bounds to row vectors
        lower_bounds = np.atleast_1d(lower_bounds)
        upper_bounds = np.atleast_1d(upper_bounds)
        plausible_lower_bounds = np.atleast_1d(plausible_lower_bounds)
        plausible_upper_bounds = np.atleast_1d(plausible_upper_bounds)
        try:
            if lower_bounds.shape != (1, D):
                logging.warning("Reshaping lower bounds to (1, %d).", D)
                lower_bounds = lower_bounds.reshape((1, D))
            if upper_bounds.shape != (1, D):
                logging.warning("Reshaping upper bounds to (1, %d).", D)
                upper_bounds = upper_bounds.reshape((1, D))
            if plausible_lower_bounds.shape != (1, D):
                logging.warning(
                    "Reshaping plausible lower bounds to (1, %d).", D
                )
                plausible_lower_bounds = plausible_lower_bounds.reshape((1, D))
            if plausible_upper_bounds.shape != (1, D):
                logging.warning(
                    "Reshaping plausible upper bounds to (1, %d).", D
                )
                plausible_upper_bounds = plausible_upper_bounds.reshape((1, D))
        except ValueError as exc:
            raise ValueError(
                "Bounds must match problem dimension D=%d.", D
            ) from exc

        # check that plausible bounds are finite
        if np.any(np.invert(np.isfinite(plausible_lower_bounds))) or np.any(
            np.invert(np.isfinite(plausible_upper_bounds))
        ):
            raise ValueError(
                "Plausible interval bounds PLB and PUB need to be finite."
            )

        # Test that all vectors are real-valued
        if (
            np.any(np.invert(np.isreal(x0)))
            or np.any(np.invert(np.isreal(lower_bounds)))
            or np.any(np.invert(np.isreal(upper_bounds)))
            or np.any(np.invert(np.isreal(plausible_lower_bounds)))
            or np.any(np.invert(np.isreal(plausible_upper_bounds)))
        ):
            raise ValueError(
                """All input vectors (x0, lower_bounds, upper_bounds,
                 plausible_lower_bounds, plausible_upper_bounds), if specified,
                 need to be real valued."""
            )

        # Cast all vectors to floats
        # (integer_vars are represented as floats but handled separately).
        if np.issubdtype(x0.dtype, np.integer):
            logging.warning("Casting initial points to floating point.")
            x0 = x0.astype(np.float64)
        if np.issubdtype(lower_bounds.dtype, np.integer):
            logging.warning("Casting lower bounds to floating point.")
            lower_bounds = lower_bounds.astype(np.float64)
        if np.issubdtype(upper_bounds.dtype, np.integer):
            logging.warning("Casting upper bounds to floating point.")
            upper_bounds = upper_bounds.astype(np.float64)
        if np.issubdtype(plausible_lower_bounds.dtype, np.integer):
            logging.warning(
                "Casting plausible lower bounds to floating point."
            )
            plausible_lower_bounds = plausible_lower_bounds.astype(np.float64)
        if np.issubdtype(plausible_upper_bounds.dtype, np.integer):
            logging.warning(
                "Casting plausible upper bounds to floating point."
            )
            plausible_upper_bounds = plausible_upper_bounds.astype(np.float64)

        # Fixed variables (all bounds equal) are not supported
        fixidx = (
            (lower_bounds == upper_bounds)
            & (upper_bounds == plausible_lower_bounds)
            & (plausible_lower_bounds == plausible_upper_bounds)
        )
        if np.any(fixidx):
            raise ValueError(
                """FixedVariables: fixed variables are not supported. Lower
                 and upper bounds should be different."""
            )

        # Test that plausible bounds are different
        if np.any(plausible_lower_bounds == plausible_upper_bounds):
            raise ValueError(
                """MatchingPB:For all variables,
            plausible lower and upper bounds need to be distinct."""
            )

        # Check that all X0 are inside the bounds
        if np.any(x0 < lower_bounds) or np.any(x0 > upper_bounds):
            raise ValueError(
                """InitialPointsNotInsideBounds: The starting
            points X0 are not inside the provided hard bounds LB and UB."""
            )

        # % Compute "effective" bounds (slightly inside provided hard bounds)
        bounds_range = upper_bounds - lower_bounds
        bounds_range[np.isinf(bounds_range)] = 1e3
        scale_factor = 1e-3
        realmin = sys.float_info.min
        LB_eff = lower_bounds + scale_factor * bounds_range
        LB_eff[np.abs(lower_bounds) <= realmin] = (
            scale_factor * bounds_range[np.abs(lower_bounds) <= realmin]
        )
        UB_eff = upper_bounds - scale_factor * bounds_range
        UB_eff[np.abs(upper_bounds) <= realmin] = (
            -scale_factor * bounds_range[np.abs(upper_bounds) <= realmin]
        )
        # Infinities stay the same
        LB_eff[np.isinf(lower_bounds)] = lower_bounds[np.isinf(lower_bounds)]
        UB_eff[np.isinf(upper_bounds)] = upper_bounds[np.isinf(upper_bounds)]

        if np.any(LB_eff >= UB_eff):
            raise ValueError(
                """StrictBoundsTooClose: Hard bounds LB and UB
                are numerically too close. Make them more separate."""
            )

        # Fix when provided X0 are almost on the bounds -- move them inside
        if np.any(x0 < LB_eff) or np.any(x0 > UB_eff):
            self.logger.warning(
                "InitialPointsTooClosePB: The starting points X0 are on "
                f"or numerically too close to the hard bounds LB (n = {np.sum(x0 < LB_eff)}) and UB (n = {np.sum(x0 > UB_eff)}). "
                # "Moving the initial points more inside..."
            )
            # # Remove points that are outside the bounds
            # out_of_bounds = (x0 < LB_eff) | (x0 > UB_eff)
            # x0 = x0[~np.any(out_of_bounds, axis=1)]

            # Move points inside the bounds
            # x0 = np.maximum((np.minimum(x0, UB_eff)), LB_eff)

        # Test order of bounds (permissive)
        ordidx = (
            (lower_bounds <= plausible_lower_bounds)
            & (plausible_lower_bounds < plausible_upper_bounds)
            & (plausible_upper_bounds <= upper_bounds)
        )
        if np.any(np.invert(ordidx)):
            raise ValueError(
                """StrictBounds: For each variable, hard and
            plausible bounds should respect the ordering LB < PLB < PUB < UB."""
            )

        # Test that plausible bounds are reasonably separated from hard bounds
        if np.any(LB_eff > plausible_lower_bounds) or np.any(
            plausible_upper_bounds > UB_eff
        ):
            self.logger.warning(
                "TooCloseBounds: For each variable, hard "
                "and plausible bounds should not be too close. "
                "Moving plausible bounds."
            )
            plausible_lower_bounds = np.maximum(plausible_lower_bounds, LB_eff)
            plausible_upper_bounds = np.minimum(plausible_upper_bounds, UB_eff)

        # Test order of bounds
        ordidx = (
            (lower_bounds < plausible_lower_bounds)
            & (plausible_lower_bounds < plausible_upper_bounds)
            & (plausible_upper_bounds < upper_bounds)
        )
        if np.any(np.invert(ordidx)):
            raise ValueError(
                """StrictBounds: For each variable, hard and
            plausible bounds should respect the ordering LB < PLB < PUB < UB."""
            )

        # Check that variables are either bounded or unbounded
        # (not half-bounded)
        if np.any(
            (np.isfinite(lower_bounds) & np.isinf(upper_bounds))
            | (np.isinf(lower_bounds) & np.isfinite(upper_bounds))
        ):
            raise ValueError(
                """HalfBounds: Each variable needs to be unbounded or
            bounded. Variables bounded only below/above are not supported."""
            )

        return (
            x0,
            lower_bounds,
            upper_bounds,
            plausible_lower_bounds,
            plausible_upper_bounds,
        )

    def _init_optim_state(self):
        """
        A private function to init the optim_state dict that contains
        information about VBMC variables.
        """
        # Record starting points (original coordinates)
        log_likes = np.array(self.options.get("log_likes")).ravel()
        log_priors_orig = np.array(self.options.get("log_priors_orig")).ravel()

        sigma = np.full([self.x0.shape[0]], np.nan)
        if self.options.get("specify_target_noise"):
            sigma = np.array(self.options.get("S_orig")).ravel()
            if np.size(sigma) == 1:
                sigma = np.repeat(sigma, len(log_likes))

        if len(log_likes) == 0:
            log_likes = np.full([self.x0.shape[0]], np.nan)
            log_priors_orig = np.full([self.x0.shape[0]], np.nan)
            sigma = np.full([self.x0.shape[0]], np.nan)

        if len(self.x0) != len(log_likes) or len(log_likes) != len(
            log_priors_orig
        ):
            raise ValueError(
                """vbmc:MismatchedStartingInputs The number of
            points in X0 and of their function values as specified in
            self.options.log_likes, self.options.log_prior_orig are
            not the same."""
            )

        x0_trans = self.parameter_transformer(self.x0)
        # Remove points with inf values, meaning they are too close to the bounds
        inds_valid = ~np.isinf(x0_trans).any(axis=1)
        self.logger.warning(
            f"Removing {np.sum(~inds_valid)}/{self.x0.shape[0]} points that are inf in transformed space (too close to the bounds): {self.x0[~inds_valid]}"
        )
        self.x0 = self.x0[inds_valid]
        log_likes = log_likes[inds_valid]
        log_priors_orig = log_priors_orig[inds_valid]
        sigma = sigma[inds_valid]

        inds = np.random.permutation(len(self.x0))  # shuffle the indices
        optim_state = {}
        optim_state["cache"] = {}
        optim_state["cache"]["x_orig"] = self.x0[inds]
        optim_state["cache"]["log_likes"] = log_likes[inds]
        optim_state["cache"]["log_priors_orig"] = log_priors_orig[inds]

        if self.options.get("specify_target_noise"):
            optim_state["cache"]["S_orig"] = sigma[inds]
            assert np.all(
                ~np.isnan(
                    optim_state["cache"]["S_orig"][
                        ~np.isnan(optim_state["cache"]["log_likes"])
                    ]
                )
            ), (
                "S_orig need to be provided for all points in X0 with non-nan log likelihoods."
            )

        # Does the starting cache contain function values?
        optim_state["cache_active"] = np.any(
            np.isfinite(optim_state.get("cache").get("log_likes"))
        )

        optim_state["lb_orig"] = self.lower_bounds.copy()
        optim_state["ub_orig"] = self.upper_bounds.copy()
        optim_state["plb_orig"] = self.plausible_lower_bounds.copy()
        optim_state["pub_orig"] = self.plausible_upper_bounds.copy()

        # Transform variables (Transform of lower bounds and upper bounds can
        # create warning but we are aware of this and output is correct)
        # with np.errstate(divide="ignore"):
        optim_state["lb_tran"] = self.parameter_transformer(self.lower_bounds)
        optim_state["ub_tran"] = self.parameter_transformer(self.upper_bounds)
        optim_state["plb_tran"] = self.parameter_transformer(
            self.plausible_lower_bounds
        )
        optim_state["pub_tran"] = self.parameter_transformer(
            self.plausible_upper_bounds
        )

        # Before first iteration
        # Iterations are from 0 onwards in optimize so we should have -1
        # here.
        optim_state["iter"] = -1

        # Set uncertainty handling level
        # (0: none; 1: unknown noise level; 2: user-provided noise)
        if self.options.get("specify_target_noise"):
            optim_state["uncertainty_handling_level"] = 2
        elif len(self.options.get("uncertainty_handling")) > 0:
            optim_state["uncertainty_handling_level"] = 1
        else:
            optim_state["uncertainty_handling_level"] = 0

        optim_state["D"] = self.D
        optim_state["task_name"] = self.task_name
        optim_state["bench_seed"] = self.bench_seed
        return optim_state

    def _init_logger(self, substring=""):
        """
        Private method to initialize the logging object.

        Parameters
        ----------
        substring : str
            A substring to append to the logger name (used to create separate
            logging objects for initialization and optimization, in case
            options change in between). Default "" (empty string).

        Returns
        -------
        logger : logging.Logger
            The main logging interface.
        """
        # set up logger
        name = "NFR" + substring
        logger = logging.getLogger(name)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger.setLevel(logging.INFO)
        if self.options.get("display") == "off":
            logger.setLevel(logging.WARN)
        elif self.options.get("display") == "iter":
            logger.setLevel(logging.INFO)
        elif self.options.get("display") == "full":
            logger.setLevel(logging.DEBUG)
        # Add a special logger for sending messages only to the default stream:
        logger.stream_only = logging.getLogger("stream_only")

        # Options and special handling for writing to a file:

        # If logging for the first time, get write mode from user options
        # (default "a" for append)
        if substring == "_init":
            log_file_mode = self.options.get("log_file_mode", "a")
        # On subsequent writes, switch to append mode:
        else:
            log_file_mode = "a"

        if logger.hasHandlers():
            logger.handlers.clear()

        if self.options.get("log_file_name") and self.options.get(
            "log_file_level"
        ):
            file_handler = logging.FileHandler(
                filename=os.path.join(
                    self.options["experiment_folder"],
                    self.options["log_file_name"],
                ),
                mode=log_file_mode,
            )

            # Set file logger level according to string or logging level:
            log_file_level = self.options.get("log_file_level", logging.INFO)
            if log_file_level == "off":
                file_handler.setLevel(logging.WARN)
            elif log_file_level == "iter":
                file_handler.setLevel(logging.INFO)
            elif log_file_level == "full":
                file_handler.setLevel(logging.DEBUG)
            elif log_file_level in [0, 10, 20, 30, 40, 50]:
                file_handler.setLevel(log_file_level)
            else:
                raise ValueError(
                    "Log file logging level is not a recognized"
                    + "string or logging level."
                )

            # Add a filter to ignore messages sent to logger.stream_only:
            def log_file_filter(record):
                return record.name != "NFR.stream_only"

            file_handler.addFilter(log_file_filter)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)
        logger.propagate = False
        return logger

    def save(self, filepath):
        try:
            filepath_saving = filepath + "-saving"
            with open(filepath_saving, "wb") as f:
                dill.dump(self, f)
            os.rename(filepath_saving, filepath)
        except:
            pass

    def save_surrogate_samples(
        self, filepath=None, n_samples=None, save_density=False
    ):
        if n_samples is None:
            n_samples = self.options.get("save_surrogate_samples")
        if filepath is None:
            filepath = (
                Path(self.options["experiment_folder"])
                / f"posterior_samples/samples_{self.iteration}.csv"
            )
        if n_samples:
            samples_trans = self.surrogate.sample(n_samples)
            samples = self.parameter_transformer.inverse(samples_trans)
            self.logger.info(f"Saving posterior samples to {filepath!s}")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                np.savetxt(f, samples)

            if save_density:
                filepath = filepath.parent / (filepath.stem + "_density.csv")
                self.logger.info(f"Saving posterior samples to {filepath!s}")
                log_abs_dets = self.parameter_transformer.log_abs_det_jacobian(
                    samples_trans
                )
                logp_flow_trans = (
                    self.surrogate.log_prob(samples_trans)
                    .detach()
                    .cpu()
                    .numpy()
                )
                logp_flow = (
                    logp_flow_trans - log_abs_dets
                )  # flow density in the original bounded space
                with open(filepath, "w") as f:
                    np.savetxt(f, logp_flow)

    def plot_iteration(self):
        # Display results at every iteration
        with torch.no_grad():
            fig_id = 0
            X_train_np, Y_train_np = _get_training_data(
                self.function_logger, self.options
            )[:2]
            extra_data = None
            highlight_inds = np.argsort(-Y_train_np.flatten())[:1]
            fig = plot_with_extra_data(
                self.surrogate,
                title="flow",
                X=X_train_np,
                # plot_data=True if self.iteration == 0 else False,
                plot_data=True,
                figure_size=(2 * self.subspace, 2 * self.subspace),
                highlight_data=highlight_inds,
                extra_data=extra_data,
                plot_style={
                    "extra_data": {
                        "s": 20,
                        "color": "red",
                        "marker": "x",
                    }
                },
                parameter_transformer=self.parameter_transformer,
                subspace=self.subspace,
            )
            fig.savefig(
                os.path.join(
                    self.fig_dir,
                    f"Iter_{self.iteration}_Fig_{fig_id}",
                )
            )

            fig = plot_with_extra_data(
                self.surrogate,
                title="flow (unconstrained)",
                X=X_train_np,
                # plot_data=True if self.iteration == 0 else False,
                plot_data=True,
                figure_size=(2 * self.subspace, 2 * self.subspace),
                highlight_data=highlight_inds,
                extra_data=extra_data,
                plot_style={
                    "extra_data": {
                        "s": 20,
                        "color": "red",
                        "marker": "x",
                    }
                },
                subspace=self.subspace,
            )
            fig.savefig(
                os.path.join(
                    self.fig_dir,
                    f"Iter_{self.iteration}_Fig_{fig_id}_unconstrained",
                )
            )


def debug_plot_iteration_0(self, X_train):
    if self.options.get("plot"):
        with torch.no_grad():
            X_train_np = X_train.cpu().numpy()
            fig = plot_with_extra_data(
                self.surrogate,
                title="flow",
                X=X_train_np,
                plot_data=True if self.iteration == 0 else False,
                figure_size=(2 * self.subspace, 2 * self.subspace),
                extra_data=None,
                plot_style={
                    "extra_data": {
                        "s": 20,
                        "color": "red",
                        "marker": "x",
                    }
                },
                parameter_transformer=self.parameter_transformer,
                subspace=self.subspace,
            )
            fig.savefig(
                os.path.join(
                    self.fig_dir,
                    f"Iter_{self.iteration}_Fig_0_pre-training",
                )
            )

            fig = plot_with_extra_data(
                self.surrogate,
                title="flow (unconstrained)",
                X=X_train_np,
                plot_data=True if self.iteration == 0 else False,
                figure_size=(2 * self.subspace, 2 * self.subspace),
                extra_data=None,
                plot_style={
                    "extra_data": {
                        "s": 20,
                        "color": "red",
                        "marker": "x",
                    }
                },
                subspace=self.subspace,
            )
            fig.savefig(
                os.path.join(
                    self.fig_dir,
                    f"Iter_{self.iteration}_Fig_0_unconstrained_pre-training",
                )
            )
