import copy
from typing import Optional, Union

import numpy as np
import torch
from plum import dispatch
from torch import Tensor
from tqdm import tqdm

from .function_logger import FunctionLogger
from .options import Options


def get_initial_samples(
    optim_state,
    function_logger: FunctionLogger,
    logger,
    options,
):
    parameter_transformer = function_logger.parameter_transformer
    x0 = optim_state["cache"]["x_orig"]
    provided_sample_count, D = x0.shape
    # TODO: add support for the case of no initial points, by sampling from the prior or plausible bounds. For now we require providing reasonable initial points.
    assert options["init_design"] == "provided", (
        "prior/plausible not implemented yet."
    )

    assert provided_sample_count > 0, "No initial points provided."

    Xs = np.copy(x0[:provided_sample_count])

    log_likes = np.copy(
        optim_state["cache"]["log_likes"][:provided_sample_count]
    )
    log_priors_orig = np.copy(
        optim_state["cache"]["log_priors_orig"][:provided_sample_count]
    )
    # if the uncertainty_level is 2 the user needs to fill in
    # the cache for the noise S (not just for y) at each x0
    if optim_state["uncertainty_handling_level"] == 2:
        S_orig = np.copy(
            optim_state["cache"]["S_orig"][:provided_sample_count]
        )
    idx_remove = np.full(provided_sample_count, True)
    # Remove points from starting cache
    optim_state["cache"]["x_orig"] = np.delete(
        optim_state["cache"]["x_orig"], idx_remove, 0
    )
    optim_state["cache"]["log_likes"] = np.delete(
        optim_state["cache"]["log_likes"], idx_remove, 0
    )
    optim_state["cache"]["log_priors_orig"] = np.delete(
        optim_state["cache"]["log_priors_orig"], idx_remove, 0
    )
    if optim_state["uncertainty_handling_level"] == 2:
        optim_state["cache"]["S_orig"] = np.delete(
            optim_state["cache"]["S_orig"], idx_remove, 0
        )

    Xs = parameter_transformer(Xs)
    # Delete points with nan (outside of original bounds)
    inds = ~np.isnan(Xs).any(axis=1)
    Xs = Xs[inds]
    logger.debug(f"{np.sum(inds)}/{np.size(inds)} points are kept.")

    for idx in tqdm(range(Xs.shape[0])):
        if np.isnan(log_likes[idx]):  # Function value is not available
            function_logger(Xs[idx])
        else:
            if optim_state["uncertainty_handling_level"] == 0:
                function_logger.add(
                    Xs[idx],
                    log_likes[idx],
                    log_priors_orig[idx],
                    log_priors_orig[idx],
                )
            elif optim_state["uncertainty_handling_level"] == 2:
                function_logger.add(
                    Xs[idx],
                    log_likes[idx],
                    log_priors_orig[idx],
                    log_priors_orig[idx],
                    S_orig[idx],
                )
            else:
                raise NotImplementedError

    return function_logger, optim_state


def _get_training_data(
    function_logger: FunctionLogger,
    options: Options,
    add_noise=0.0,
    torch_device=None,
    return_std=False,
):
    """
    Get training data for building GP surrogate.

    Parameters
    ==========
    function_logger : FunctionLogger
        Function logger from the VBMC instance which we are calling this from.

    Returns
    =======
    x_train, ndarray
        Training inputs.
    y_train, ndarray
        Training targets.
    s2_train, ndarray, optional
        Training data noise variance, if noise is used.
    t_train, ndarray
        Array of the times it took to evaluate the function on the training
        data.
    """

    x_train = function_logger.X[function_logger.X_flag, :]
    y_train = function_logger.y[function_logger.X_flag]
    if function_logger.noise_flag:
        s2_train = function_logger.S[function_logger.X_flag] ** 2
    else:
        s2_train = options["tol_noise"] ** 2 * np.ones_like(y_train)

    s2_train += add_noise

    if options["noise_shaping"]:
        s2_train_ns = noise_shaping(s2_train, y_train, options)
    else:
        s2_train_ns = np.array([0.0])
    s2_train_raw = np.maximum(s2_train, options["tol_noise"] ** 2)
    s2_train_ns = np.maximum(s2_train_ns, options["tol_noise"] ** 2)
    s_train_ns = np.sqrt(s2_train_ns)
    s_train_raw = np.sqrt(s2_train_raw)
    t_train = function_logger.fun_evaltime[function_logger.X_flag]

    if torch_device is not None:
        x_train = torch.from_numpy(x_train).to(torch_device)
        y_train = torch.from_numpy(y_train).to(torch_device)
        s2_train_ns = torch.from_numpy(s2_train_ns).to(torch_device)
        s2_train_raw = torch.from_numpy(s2_train_raw).to(torch_device)
        s_train_ns = torch.sqrt(s2_train_ns)
        s_train_raw = torch.sqrt(s2_train_raw)
    if return_std:
        return x_train, y_train, s_train_ns, s_train_raw, t_train
    return x_train, y_train, s2_train_ns, s2_train_raw, t_train


@dispatch
def noise_shaping(
    s2: Union[np.ndarray, None],
    y: np.ndarray,
    options: Union[Options, dict],
    ymax: Optional[float] = None,
):
    # Increase noise for low density points
    if s2 is None:
        s2 = options["tol_noise"] ** 2 * np.ones_like(y)
    else:
        assert np.all(s2 >= 0)

    if ymax is None:
        ymax = np.max(y)
    frac = np.minimum(1, (ymax - y) / options["noise_shaping_threshold"])
    if options.get("linear_shaping"):
        sigma_shape = (
            options["noise_shaping_min"] * (1 - frac)
            + frac * options["noise_shaping_med"]
        )
    else:
        if options.get("noise_shaping_med") == 0:
            assert options.get("noise_shaping_min") == 0
            sigma_shape = 0.0
        else:
            min_lnsigma = np.log(options["noise_shaping_min"])
            med_lnsigma = np.log(options["noise_shaping_med"])
            sigma_shape = np.exp(min_lnsigma * (1 - frac) + frac * med_lnsigma)

    delta_y = np.maximum(0, ymax - y - options["noise_shaping_threshold"])
    sigma_shape += options["noise_shaping_factor"] * delta_y

    sn2extra = sigma_shape**2

    s2s = s2 + sn2extra
    # Excessive difference between low and high noise might cause numerical
    # instabilities, so we give the option of capping the ratio
    if not np.isinf(options["noise_shaping_max_ratio"]):
        maxs2 = np.min(s2s) * options["noise_shaping_max_ratio"]
        s2s = np.minimum(s2s, maxs2)
    return s2s


@dispatch
def noise_shaping(  # noqa: F811
    s2: Union[Tensor, None],
    y: Tensor,
    options: Union[Options, dict],
    ymax: Optional[float] = None,
):
    # Increase noise for low density points
    if s2 is None:
        s2 = options["tol_noise"] ** 2 * torch.ones_like(y)
    else:
        assert torch.all(s2 >= 0)

    if ymax is None:
        ymax = torch.max(y)
    frac = torch.minimum(
        torch.ones_like(y), (ymax - y) / options["noise_shaping_threshold"]
    )
    if options.get("linear_shaping"):
        sigma_shape = (
            options["noise_shaping_min"] * (1 - frac)
            + frac * options["noise_shaping_med"]
        )
    else:
        if options.get("noise_shaping_med") == 0:
            assert options.get("noise_shaping_min") == 0
            sigma_shape = 0.0
        else:
            min_lnsigma = np.log(options["noise_shaping_min"])
            med_lnsigma = np.log(options["noise_shaping_med"])
            sigma_shape = torch.exp(
                min_lnsigma * (1 - frac) + frac * med_lnsigma
            )

    delta_y = torch.maximum(
        torch.zeros_like(y), ymax - y - options["noise_shaping_threshold"]
    )
    sigma_shape += options["noise_shaping_factor"] * delta_y

    sn2extra = sigma_shape**2

    s2s = s2 + sn2extra
    # Excessive difference between low and high noise might cause numerical
    # instabilities, so we give the option of capping the ratio
    if not np.isinf(options["noise_shaping_max_ratio"]):
        maxs2 = torch.min(s2s) * options["noise_shaping_max_ratio"]
        s2s = torch.minimum(s2s, maxs2)
    return s2s


def warp_input(
    mean, cov, optim_state, function_logger, parameter_transformer, options
):
    r"""Compute input warping of variables and update the cached points in
    function_logger accordingly.

    A whitening transformation: a rotation and
    rescaling of the inference space such that the new covariance matrix
    is diagonal.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix.
    optim_state : dict
        The dictionary recording the current optimization state.
    function_logger : FunctionLogger
        The record including cached function values.

    Returns
    -------
    parameter_transformer : ParameterTransformer
        A ParameterTransformer object representing the new transformation
        between original coordinates and inference space coordinates, with the
        input warping applied.
    optim_state : dict
        An updated copy of the original optimization state dict.
    function_logger : FunctionLogger
        An updated copy of the original function logger.
    """
    old_parameter_transformer = parameter_transformer
    new_parameter_transformer = copy.deepcopy(parameter_transformer)
    optim_state = copy.deepcopy(optim_state)
    function_logger = copy.deepcopy(function_logger)
    D = cov.shape[0]

    delta = new_parameter_transformer.delta
    R_mat = new_parameter_transformer.R_mat
    scale = new_parameter_transformer.scale
    if R_mat is None:
        R_mat = np.eye(D)
    if scale is None:
        scale = np.ones(D)
    cov = R_mat @ np.diag(scale) @ cov @ np.diag(scale) @ R_mat.T
    cov = np.diag(delta) @ cov @ np.diag(delta)

    # Remove low-correlation entries
    if options["warp_roto_corr_thresh"] > 0:
        vp_corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        mask_idx = np.abs(vp_corr) <= options["warp_roto_corr_thresh"]
        cov[mask_idx] = 0

    # Regularization of covariance matrix towards diagonal
    w_reg = options["warp_cov_reg"]
    w_reg = np.max([0, np.min([1, w_reg])])
    cov = (1 - w_reg) * cov + w_reg * np.diag(np.diag(cov))

    # Compute whitening transform (rotoscaling)
    U, s, __ = np.linalg.svd(cov)
    if np.linalg.det(U) < 0:
        U[:, 0] = -U[:, 0]
    scale = np.sqrt(s + np.finfo(np.float64).eps)
    new_parameter_transformer.R_mat = U
    new_parameter_transformer.scale = scale

    # Update shift and scaling and plausible bounds:
    new_parameter_transformer.mu = np.zeros(D)
    new_parameter_transformer.delta = np.ones(D)
    # Recenter
    tmp_parameter_transformer = copy.deepcopy(new_parameter_transformer)
    tmp_parameter_transformer.R_mat = None
    tmp_parameter_transformer.scale = None
    mean_orig = old_parameter_transformer.inverse(mean)
    mean = tmp_parameter_transformer(mean_orig)
    new_parameter_transformer.mu = mean
    assert np.allclose(new_parameter_transformer(mean_orig), 0)

    Nrnd = 100000
    xx = (
        np.random.rand(Nrnd, D)
        * (optim_state["pub_orig"] - optim_state["plb_orig"])
        + optim_state["plb_orig"]
    )
    yy = new_parameter_transformer(xx)

    # Quantile-based estimate of plausible bounds
    [plb_tran, pub_tran] = np.quantile(yy, [0.05, 0.95], axis=0)
    delta_temp = pub_tran - plb_tran
    plb_tran = plb_tran - delta_temp / 9
    pub_tran = pub_tran + delta_temp / 9

    optim_state["plb_tran"] = plb_tran.reshape((1, D))
    optim_state["pub_tran"] = pub_tran.reshape((1, D))

    # Adjust stored points after warping
    X_flag = function_logger.X_flag
    X_orig = function_logger.X_orig[X_flag, :]
    y_orig = function_logger.y_orig[X_flag].T
    log_prior_orig = function_logger.log_priors_orig[X_flag].T
    log_p0s_orig = function_logger.log_p0s_orig[X_flag].T
    X = new_parameter_transformer(X_orig)
    dy = new_parameter_transformer.log_abs_det_jacobian(X)
    y = y_orig + dy
    log_prior = log_prior_orig + dy
    log_p0s = log_p0s_orig + dy
    function_logger.X[X_flag, :] = X
    function_logger.y[X_flag] = y.T
    function_logger.log_priors[X_flag] = log_prior.T
    function_logger.log_p0s[X_flag] = log_p0s.T
    function_logger.parameter_transformer = new_parameter_transformer
    function_logger._update_values()

    return new_parameter_transformer, optim_state, function_logger
