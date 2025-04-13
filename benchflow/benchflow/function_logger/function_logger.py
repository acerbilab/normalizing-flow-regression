import copy
from functools import wraps
from timeit import default_timer as timer
from typing import Callable

import numpy as np

from .parameter_transformer import ParameterTransformer
from .utils import Timer


class BudgetExhaustedException(Exception):
    pass


class FunctionLogger:
    """
    Class that evaluates a function and caches its values.

    Parameters
    ----------
    fun : callable
        The function to be logged.
        `fun` must take a vector input and return (log likelihood, log prior)
         and optionally, the (estimated) SD of the returned value (if the
        function fun is stochastic).
    D : int
        The number of dimensions that the function takes as input.
    noise_flag : bool
        Whether the function fun is stochastic or not.
    uncertainty_handling_level : {0, 1, 2}
        The uncertainty handling level which can be one of
        (0: none; 1: unknown noise level; 2: user-provided noise).
    cache_size : int, optional
        The initial size of caching table (default 500).
    parameter_transformer : ParameterTransformer, optional
        A ParameterTransformer is required to transform the parameters
        between constrained and unconstrained space, by default None.
    """

    def __init__(
        self,
        fun,
        D: int,
        noise_flag: bool,
        uncertainty_handling_level: int,
        cache_size: int = 500,
        parameter_transformer: ParameterTransformer = None,
    ):
        self.fun = fun
        self._log_p0_fun = None  # default None, p0 will be set as the prior
        self.D: int = D
        self.noise_flag: bool = noise_flag
        self.uncertainty_handling_level: int = uncertainty_handling_level
        self.transform_parameters = parameter_transformer is not None
        self.parameter_transformer = parameter_transformer

        self.func_count: int = 0
        self.cache_count: int = 0
        self.X_orig = np.full([cache_size, self.D], np.nan)
        self.y_orig = np.full([cache_size, 1], np.nan)
        # store log likelihood and lop prior seperately
        self.log_likes = np.full([cache_size, 1], np.nan)
        self.log_priors_orig = np.full([cache_size, 1], np.nan)
        self.log_priors = np.full(
            [cache_size, 1], np.nan
        )  # in unconstrained space
        self.log_p0s_orig = np.full([cache_size, 1], np.nan)
        self.log_p0s = np.full(
            [cache_size, 1], np.nan
        )  # in unconstrained space
        self._beta = 1

        self.X = np.full([cache_size, self.D], np.nan)
        self.y = np.full([cache_size, 1], np.nan)
        self.nevals = np.full([cache_size, 1], 0)
        self.ymax_ind = None
        self.ymax = np.nan

        if self.noise_flag:
            # for getting noise of temperred target
            self.S = np.full([cache_size, 1], np.nan)
            # for storing original noise
            self.S_orig = np.full([cache_size, 1], np.nan)
            # for storing exact log likelihood (debugging only)
            self.log_likes_exact = np.full([cache_size, 1], np.nan)

        self.Xn: int = -1  # Last filled entry

        # Use 1D array since this is a boolean mask.
        self.X_flag = np.full((cache_size,), False, dtype=bool)
        self.fun_evaltime = np.full([cache_size, 1], np.nan)
        self.total_fun_evaltime = 0

        self.budget = np.inf  # Budget number of function evaluations
        self.return_original_space = False  # Return values in original space

    def check_budget(self):
        if np.sum(self.nevals) >= self.budget:
            raise BudgetExhaustedException("Budget is exhausted!")

    @property
    def log_p0_fun(self):
        # tempered_post = p0**(1-beta) * (prior*likelihood)**beta
        # p0 is defined in original space as self.fun
        return self._log_p0_fun

    @log_p0_fun.setter
    def log_p0_fun(self, value: Callable):
        self._log_p0_fun = value
        # Need to update the values if log_p0_fun is set
        X_orig = self.X_orig[self.X_flag]
        X = self.X[self.X_flag]
        log_p0_vals = self.log_p0_fun(X_orig)
        log_p0_vals = log_p0_vals.reshape(self.log_p0s_orig[self.X_flag].shape)
        self.log_p0s_orig[self.X_flag] = log_p0_vals
        log_abs_det = self.parameter_transformer.log_abs_det_jacobian(X)
        log_abs_det = log_abs_det.reshape(self.log_p0s_orig[self.X_flag].shape)
        self.log_p0s[self.X_flag] = (
            self.log_p0s_orig[self.X_flag] + log_abs_det
        )
        self._update_values()

    def _update_values(self):
        self.y_orig = (
            self.log_p0s_orig * (1 - self.beta)
            + (self.log_likes + self.log_priors_orig) * self.beta
        )
        self.y = (
            self.log_p0s * (1 - self.beta)
            + (self.log_likes + self.log_priors) * self.beta
        )

        if np.any(self.X_flag):
            self.ymax_ind = np.nanargmax(self.y[self.X_flag])
            self.ymax = self.y[self.X_flag][self.ymax_ind]
        else:
            self.ymax_ind = None
            self.ymax = np.nan
        if self.noise_flag:
            self.S = self.beta * self.S_orig

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value
        print(
            f"Update log joint values according to target temperature {value}."
        )
        self._update_values()

    def __call__(self, x: np.ndarray):
        """
        Evaluates the function FUN at x and caches values.

        Parameters
        ----------
        x : np.ndarray
            The point at which the function will be evaluated. The shape of x
            should be (1, D) or (D,).

        Returns
        -------
        fval : float
            The result of the evaluation.
        SD : float
            The (estimated) SD of the returned value.
        idx : int
            The index of the last updated entry.

        Raises
        ------
        ValueError
            Raise if the function value is not a finite real-valued scalar.
        ValueError
            Raise if the (estimated) SD (second function output)
            is not a finite, positive real-valued scalar.
        """
        self.check_budget()
        timer = Timer()
        if x.ndim > 1:
            x = x.squeeze()
        if x.ndim == 0:
            x = np.atleast_1d(x)
        assert x.size == x.shape[0]
        # Convert back to original space
        if self.transform_parameters:
            x_orig = self.parameter_transformer.inverse(
                np.reshape(x, (1, x.shape[0]))
            )[0]
        else:
            x_orig = x

        try:
            timer.start_timer("funtime")
            log_like_val_exact = None
            if self.noise_flag and self.uncertainty_handling_level == 2:
                # fval_orig, fsd = self.fun(x_orig)
                res = self.fun(x_orig)
                if len(res) == 3:
                    log_like_val, log_prior_val_orig, fsd = res
                else:
                    (
                        log_like_val,
                        log_prior_val_orig,
                        fsd,
                        log_like_val_exact,
                    ) = res  # log_like_val_exact is available for debugging
            else:
                # fval_orig = self.fun(x_orig)
                log_like_val, log_prior_val_orig = self.fun(x_orig)
                if self.noise_flag:
                    fsd = 1
                else:
                    fsd = None
            if self.log_p0_fun is None:
                log_p0_val_orig = log_prior_val_orig
            else:
                log_p0_val_orig = self.log_p0_fun(x_orig)

            fval_orig = (1 - self.beta) * log_p0_val_orig + self.beta * (
                log_like_val + log_prior_val_orig
            )

            if isinstance(fval_orig, np.ndarray):
                # fval_orig can only be an array with size 1
                fval_orig = fval_orig.item()
            timer.stop_timer("funtime")

        except Exception as err:
            err.args += (
                "FunctionLogger:FuncError "
                + "Error in executing the logged function"
                + "with input: "
                + str(x_orig),
            )
            raise

        # if fval is an array with only one element, extract that element
        if not np.isscalar(fval_orig) and np.size(fval_orig) == 1:
            fval_orig = np.array(fval_orig).flat[0]

        # Check function value
        if np.any(
            not np.isscalar(fval_orig)
            or not np.isfinite(fval_orig)
            or not np.isreal(fval_orig)
        ):
            error_message = """FunctionLogger:InvalidFuncValue:
            The returned function value must be a finite real-valued scalar
            (returned value {})"""
            raise ValueError(error_message.format(str(fval_orig)))

        # Check returned function SD
        if self.noise_flag and (
            not np.isscalar(fsd)
            or not np.isfinite(fsd)
            or not np.isreal(fsd)
            or fsd <= 0.0
        ):
            error_message = """FunctionLogger:InvalidNoiseValue
                The returned estimated SD (second function output)
                must be a finite, positive real-valued scalar
                (returned SD: {})"""
            raise ValueError(error_message.format(str(fsd)))

        # record timer stats
        funtime = timer.get_duration("funtime")

        self.func_count += 1
        fval, idx = self._record(
            x_orig,
            x,
            log_like_val,
            log_prior_val_orig,
            log_p0_val_orig,
            fval_orig,
            fsd,
            funtime,
            log_like_val_exact,
        )

        # optimstate.N = self.Xn
        # optimstate.Neff = np.sum(self.nevals[self.X_flag])
        # optimState.totalfunevaltime = optimState.totalfunevaltime + t;

        if self.return_original_space:
            return fval_orig, fsd, idx
        return fval, fsd, idx

    def add(
        self,
        x: np.ndarray,
        log_like_val: float,
        log_prior_val_orig: float,
        log_p0_val_orig: float,
        fsd: float = None,
        fun_evaltime=np.nan,
    ):
        """
        Add an previously evaluated function sample to the function cache.

        Parameters
        ----------
        x : np.ndarray
            The point at which the function has been evaluated. The shape of x
            should be (1, D) or (D,).
        log_like_val: float
            The log likelihood value.
        log_prior_val_orig: float
            The log prior value in the original space.
        log_p0_val_orig: float
            The log p0 value in the original space.
        fsd : float, optional
            The (estimated) SD of the returned value (if heteroskedastic noise
            handling is on) of the evaluation of the function, by default None.
        fun_evaltime : float
            The duration of the time it took to evaluate the function,
            by default np.nan.

        Returns
        -------
        fval : float
            The result of the evaluation.
        SD : float
            The (estimated) SD of the returned value.
        idx : int
            The index of the last updated entry.

        Raises
        ------
        ValueError
            Raise if the function value is not a finite real-valued scalar.
        ValueError
            Raise if the (estimated) SD (second function output)
            is not a finite, positive real-valued scalar.
        """
        if x.ndim > 1:
            x = x.squeeze()
        if x.ndim == 0:
            x = np.atleast_1d(x)
        assert x.size == x.shape[0]
        # Convert back to original space
        if self.transform_parameters:
            x_orig = self.parameter_transformer.inverse(
                np.reshape(x, (1, x.shape[0]))
            )[0]
        else:
            x_orig = x

        if self.noise_flag:
            if fsd is None:
                fsd = 1
        else:
            fsd = None

        fval_orig = (1 - self.beta) * log_p0_val_orig + self.beta * (
            log_like_val + log_prior_val_orig
        )
        # Check function value
        if (
            not np.isscalar(fval_orig)
            or not np.isfinite(fval_orig)
            or not np.isreal(fval_orig)
        ):
            error_message = """FunctionLogger:InvalidFuncValue:
            The returned function value must be a finite real-valued scalar
            (returned value {})"""
            raise ValueError(error_message.format(str(fval_orig)))

        # Check returned function SD
        if self.noise_flag and (
            not np.isscalar(fsd)
            or not np.isfinite(fsd)
            or not np.isreal(fsd)
            or fsd <= 0.0
        ):
            error_message = """FunctionLogger:InvalidNoiseValue
                The returned estimated SD (second function output)
                must be a finite, positive real-valued scalar
                (returned SD:{})"""
            raise ValueError(error_message.format(str(fsd)))

        self.cache_count += 1
        fval, idx = self._record(
            x_orig,
            x,
            log_like_val,
            log_prior_val_orig,
            log_p0_val_orig,
            fval_orig,
            fsd,
            fun_evaltime,
        )
        return fval, fsd, idx

    def finalize(self):
        """
        Remove unused caching entries.
        """
        self.X_orig = self.X_orig[: self.Xn + 1]
        self.y_orig = self.y_orig[: self.Xn + 1]

        self.log_likes = self.log_likes[: self.Xn + 1]
        self.log_priors = self.log_priors[: self.Xn + 1]
        self.log_priors_orig = self.log_priors_orig[: self.Xn + 1]
        self.log_p0s = self.log_p0s[: self.Xn + 1]
        self.log_p0s_orig = self.log_p0s_orig[: self.Xn + 1]

        # in the original matlab version X and Y get deleted
        self.X = self.X[: self.Xn + 1]
        self.y = self.y[: self.Xn + 1]

        if self.noise_flag:
            self.S_orig = self.S_orig[: self.Xn + 1]
            self.S = self.S[: self.Xn + 1]
        self.X_flag = self.X_flag[: self.Xn + 1]
        self.fun_evaltime = self.fun_evaltime[: self.Xn + 1]

    def _expand_arrays(self, resize_amount: int = None):
        """
        A private function to extend the rows of the object attribute arrays.

        Parameters
        ----------
        resize_amount : int, optional
            The number of additional rows, by default expand current table
            by 50%.
        """

        if resize_amount is None:
            resize_amount = int(np.max((np.ceil(self.Xn / 2), 1)))

        self.X_orig = np.append(
            self.X_orig, np.full([resize_amount, self.D], np.nan), axis=0
        )
        self.y_orig = np.append(
            self.y_orig, np.full([resize_amount, 1], np.nan), axis=0
        )
        self.log_likes = np.append(
            self.log_likes, np.full([resize_amount, 1], np.nan), axis=0
        )
        self.log_priors_orig = np.append(
            self.log_priors_orig, np.full([resize_amount, 1], np.nan), axis=0
        )
        self.log_priors = np.append(
            self.log_priors, np.full([resize_amount, 1], np.nan), axis=0
        )
        self.log_p0s_orig = np.append(
            self.log_p0s_orig, np.full([resize_amount, 1], np.nan), axis=0
        )
        self.log_p0s = np.append(
            self.log_p0s, np.full([resize_amount, 1], np.nan), axis=0
        )

        self.X = np.append(
            self.X, np.full([resize_amount, self.D], np.nan), axis=0
        )
        self.y = np.append(self.y, np.full([resize_amount, 1], np.nan), axis=0)

        if self.noise_flag:
            self.S_orig = np.append(
                self.S_orig, np.full([resize_amount, 1], np.nan), axis=0
            )
            self.S = np.append(
                self.S, np.full([resize_amount, 1], np.nan), axis=0
            )
            self.log_likes_exact = np.append(
                self.log_likes_exact,
                np.full([resize_amount, 1], np.nan),
                axis=0,
            )
        self.X_flag = np.append(
            self.X_flag, np.full((resize_amount,), False, dtype=bool)
        )
        self.fun_evaltime = np.append(
            self.fun_evaltime, np.full([resize_amount, 1], np.nan), axis=0
        )
        self.nevals = np.append(
            self.nevals, np.full([resize_amount, 1], 0), axis=0
        )

    def _record(
        self,
        x_orig: float,
        x: float,
        log_like_val: float,
        log_prior_val_orig: float,
        log_p0_val_orig: float,
        fval_orig: float,
        fsd: float,
        fun_evaltime: float,
        log_like_val_exact: float = None,
    ):
        """
        A private method to save function values to class attributes.

        Parameters
        ----------
        x_orig : float
            The point at which the function has been evaluated
            (in original space).
        x : float
            The point at which the function has been evaluated
            (in transformed space).
        log_like_val: float
            The log likelihood value.
        log_prior_val_orig: float
            The log prior value in the original space.
        log_p0_val_orig: float
            The log p0 value in the original space.
        fval_orig : float
            The result of the evaluation.
        fsd : float
            The (estimated) SD of the returned value
            (if heteroskedastic noise handling is on).
        fun_evaltime : float
            The duration of the time it took to evaluate the function.

        Returns
        -------
        fval : float
            The result of the evaluation.
        idx : int
            The index of the last updated entry.

        Raises
        ------
        ValueError
            Raise if there is more than one match for a duplicate entry.
        """
        duplicate_flag = (self.X == x).all(axis=1)
        if np.any(duplicate_flag):
            if np.sum(duplicate_flag) > 1:
                raise ValueError("More than one match for duplicate entry.")
            idx = np.argwhere(duplicate_flag)[0, 0]
            N = self.nevals[idx]
            if fsd is not None:
                tau_n = 1 / self.S_orig[idx] ** 2
                tau_1 = 1 / fsd**2
                self.y_orig[idx] = (
                    tau_n * self.y_orig[idx] + tau_1 * fval_orig
                ) / (tau_n + tau_1)
                self.S_orig[idx] = 1 / np.sqrt(tau_n + tau_1)
                self.S[idx] = self.beta * self.S_orig[idx]

                self.log_likes[idx] = (
                    tau_n * self.log_likes[idx] + tau_1 * log_like_val
                ) / (tau_n + tau_1)
            else:
                # print(self.log_likes[idx], log_like_val)
                # print(self.y_orig[idx], fval_orig)
                assert np.allclose(
                    (N * self.y_orig[idx] + fval_orig) / (N + 1),
                    self.y_orig[idx],
                )
                assert np.allclose(
                    (N * self.log_likes[idx] + log_like_val) / (N + 1),
                    self.log_likes[idx],
                )
                self.y_orig[idx] = (N * self.y_orig[idx] + fval_orig) / (N + 1)

                self.log_likes[idx] = (
                    N * self.log_likes[idx] + log_like_val
                ) / (N + 1)

            fval = self.y_orig[
                idx
            ].item()  # item() to convert to scalar, or else it's an array and may be wrongly modified in place.
            if self.transform_parameters:
                # TODO: check reshaping?
                log_det_value = (
                    self.parameter_transformer.log_abs_det_jacobian(
                        np.reshape(x, (1, x.shape[0]))
                    )[0]
                )
                fval += log_det_value

            self.y[idx] = fval
            self.fun_evaltime[idx] = (
                N * self.fun_evaltime[idx] + fun_evaltime
            ) / (N + 1)
            self.nevals[idx] += 1
            assert np.allclose(
                self.log_likes[idx] + self.log_priors_orig[idx],
                self.y_orig[idx],
            )
            return fval, idx
        else:
            self.Xn += 1
            if self.Xn > self.X_orig.shape[0] - 1:
                self._expand_arrays()

            # record function time
            if not np.isnan(fun_evaltime):
                self.fun_evaltime[self.Xn] = fun_evaltime
                self.total_fun_evaltime += fun_evaltime

            self.X_orig[self.Xn] = x_orig
            self.X[self.Xn] = x
            self.y_orig[self.Xn] = fval_orig

            self.log_likes[self.Xn] = log_like_val
            if log_like_val_exact is not None:
                self.log_likes_exact[self.Xn] = log_like_val_exact
            self.log_priors_orig[self.Xn] = log_prior_val_orig
            self.log_priors[self.Xn] = log_prior_val_orig
            self.log_p0s_orig[self.Xn] = log_p0_val_orig
            self.log_p0s[self.Xn] = log_p0_val_orig

            fval = fval_orig
            if self.transform_parameters:
                log_det_value = (
                    self.parameter_transformer.log_abs_det_jacobian(
                        np.reshape(x, (1, x.shape[0]))
                    )[0]
                )
                self.log_priors[self.Xn] += log_det_value
                self.log_p0s[self.Xn] += log_det_value
                fval += log_det_value
            else:
                self.log_priors[self.Xn] = log_prior_val_orig
                self.log_p0s[self.Xn] = log_p0_val_orig

            self.y[self.Xn] = fval
            if fsd is not None:
                self.S_orig[self.Xn] = fsd
                self.S[self.Xn] = self.beta * fsd
            self.X_flag[self.Xn] = True
            self.nevals[self.Xn] += 1
            self.ymax_ind = np.nanargmax(self.y[self.X_flag])
            self.ymax = self.y[self.X_flag][self.ymax_ind]
            idx = self.Xn
            assert np.allclose(
                self.log_likes[idx] + self.log_priors_orig[idx],
                self.y_orig[idx],
            )
            return fval, self.Xn

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)

        # Avoid infinite recursion in deepcopy
        memo[id(self)] = result
        # Copy class properties:
        for k, v in self.__dict__.items():
            if k == "fun":  # Avoid deepcopy of log-joint function
                # (interferes with benchmark logging)
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def delete(self, count):
        assert count >= 0
        assert self.Xn + 1 > count
        self.Xn -= count
        self.X_flag[self.Xn + 1 :] = False

    @property
    def X_valid(self):
        return self.X[self.X_flag, :]

    @property
    def y_valid(self):
        return self.y[self.X_flag]


def log_function(func):
    """A decorator to automate the logging of function evaluations.

    The decorator is designed to be applied to a ``Task.log_joint()`` method. It
    records the locations of the function evaluations and their returned value,
    the estimated noise (if the task is noisy), and the wall-clock time before
    and after function evaluation.

    Parameters
    ----------
    self : benchflow.tasks.Task
        The ``Task`` which owns the ``log_joint()`` method.
    theta : np.ndarray
        The array of input point(s), of dimension `(D,)` or `(n,D)`, where
        `D` is the problem dimension.
    *args : any
        Additional arguments are passed on to the wrapped function.
    **kwargs : any
        Additional keyword arguments are passed on to the wrapped function.

    Returns
    -------
    wrapper : callable
        The modified method, with logging features.
    """

    @wraps(func)
    def wrapper(self, theta, *args, **kwargs):
        start = timer()
        # Evaluated point:
        self._log["theta"].append(theta)
        if self.is_noisy:
            # log density and estimated noise:
            log_y, noise_est = func(self, theta, *args, **kwargs)
            self._log["log_y"].extend(log_y.ravel())
            if np.ndim(noise_est > 0):
                self._log["noise_est"].extend(noise_est.ravel())
            else:
                self._log["noise_est"].append(noise_est)
            # Number of function evaluations:
            self._log["fun_evals"] += 1
            # Wall-clock time:
            self._log["t_before"].append(start - self._log["t0"])
            self._log["t_after"].append(timer() - self._log["t0"])
            return log_y, noise_est
        else:
            # log-density:
            log_y = func(self, theta, *args, **kwargs)
            self._log["log_y"].extend(log_y.ravel())
            # Number of function evaluations:
            self._log["fun_evals"] += 1
            # Wall-clock time
            self._log["t_before"].append(start - self._log["t0"])
            self._log["t_after"].append(timer() - self._log["t0"])
            return log_y

    return wrapper
