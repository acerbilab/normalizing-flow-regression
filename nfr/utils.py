from functools import wraps
from typing import Callable, Optional, Union

import corner
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import wandb.sdk
import wandb.sdk.lib
from torch import Tensor


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
    hpd_N = round(hpd_frac * N)
    indices = order[0:hpd_N]
    hpd_X = X[indices]
    hpd_y = y[indices]

    if hpd_N > 0:
        hpd_range = np.max(hpd_X, axis=0) - np.min(hpd_X, axis=0)
    else:
        hpd_range = np.full((D), np.NaN)

    return hpd_X, hpd_y, hpd_range, indices


def _value_and_grad(f: Callable, x: Union[np.ndarray, Tensor]):
    if np.isnan(x).any():
        raise RuntimeError(
            f"{np.isnan(x).sum()} elements of the {x.size} element array "
            f"`x` are NaN."
        )
    X = torch.from_numpy(x).contiguous().requires_grad_(True)
    fval = f(X)
    # compute gradient w.r.t. the inputs (does not accumulate in leaves)
    gradf = _arrayify(torch.autograd.grad(fval, X)[0].contiguous().view(-1))
    if np.isnan(gradf).any():
        msg = (
            f"{np.isnan(gradf).sum()} elements of the {x.size} element "
            "gradient array `gradf` are NaN. "
            "This often indicates numerical issues."
        )
        raise RuntimeError(msg)
    fval = fval.item()
    return fval, gradf


def _arrayify(X: Tensor) -> np.ndarray:
    r"""Convert a torch.Tensor (any dtype or device) to a numpy (double) array.

    Args:
        X: The input tensor.

    Returns:
        A numpy array of double dtype with the same shape and data as `X`.
    """
    return X.cpu().detach().contiguous().double().clone().numpy()


def corner_fun(fun, x, plot_ranges, vmin=None, vmax=None, N_1d=100, N_2d=100):
    x = x.squeeze()
    D = np.size(x)
    for i in range(D):
        assert x[i] <= plot_ranges[i][1] and x[i] >= plot_ranges[i][0]

    K = D
    xs_all_1d = [
        np.linspace(plot_ranges[i][0], plot_ranges[i][1], N_1d)
        for i in range(D)
    ]
    xs_all_2d = [
        np.linspace(plot_ranges[i][0], plot_ranges[i][1], N_2d)
        for i in range(D)
    ]

    ys_all_1d = []
    for dim in range(D):
        xs = xs_all_1d[dim]
        points = np.tile(x, (N_1d, 1))
        points[:, dim] = xs
        ys = fun(points)
        ys_all_1d.append(ys)
    reverse = False
    fig = None
    # Some magic numbers for pretty axis layout.
    factor = 2.0  # size of one side of one panel
    if reverse:
        lbdim = 0.2 * factor  # size of left/bottom margin
        trdim = 0.5 * factor  # size of top/right margin
    else:
        lbdim = 0.5 * factor  # size of left/bottom margin
        trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.05  # w/hspace size
    plotdim = factor * K + factor * (K - 1.0) * whspace
    dim = lbdim + plotdim + trdim

    # Create a new figure if one wasn't provided.
    new_fig = True
    if fig is None:
        fig, axes = plt.subplots(K, K, figsize=(dim, dim))

    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )

    for i in range(K):
        for j in range(K):
            if reverse:
                ax = axes[K - i - 1, K - j - 1]
            else:
                ax = axes[i, j]
            if j > 0 and j != i:
                ax.set_yticklabels([])
            if i < K - 1:
                ax.set_xticklabels([])
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                ax.yaxis.tick_right()
                ax.plot(xs_all_1d[i], ys_all_1d[i])
                continue
            else:
                dim_1 = j
                dim_2 = i
                xs_1, xs_2 = np.meshgrid(xs_all_2d[dim_1], xs_all_2d[dim_2])
                points = np.tile(x, (np.size(xs_1), 1))
                points[:, dim_1] = xs_1.ravel()
                points[:, dim_2] = xs_2.ravel()
                ys = fun(points)
                ys = ys.reshape(N_2d, N_2d)
                im = ax.pcolormesh(xs_1, xs_2, ys, vmin=vmin, vmax=vmax)
    if K == 2:
        rec = [1 - 1 / (2 * K), 1 / (K) + 0.1, 0.03, 2 / (3 * K)]
    else:
        rec = [1 - 1 / (2 * K), 0.5 + 1 / (2 * K), 0.03, 4 / (3 * K)]
    cbar_ax = fig.add_axes(rec)
    fig.colorbar(im, cax=cbar_ax)
    return fig


def handle_0D_1D_input(patched_kwargs, patched_argpos, return_scalar=False):
    """
    A decorator that handles 0D, 1D inputs and transforms them to 2D.

    Parameters
    ----------
    kwarg : list of str
        The names of the keyword arguments that should be handeled.
    argpos : list of int
        The positions of the arguments that should be handeled.
    return_scalar : bool, optional
        If the input is 1D the function should return a scalar,
        by default False.
    """

    def decorator(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            for idx, patched_kwarg in enumerate(patched_kwargs):
                if patched_kwarg in kwargs:
                    # for keyword arguments
                    input_dims = np.ndim(kwargs.get(patched_kwarg))
                    kwargs[patched_kwarg] = np.atleast_2d(
                        kwargs.get(patched_kwarg)
                    )

                elif len(args) > patched_argpos[idx]:
                    # for positional arguments
                    arg_list = list(args)
                    input_dims = np.ndim(args[patched_argpos[idx]])
                    arg_list[patched_argpos[idx]] = np.atleast_2d(
                        args[patched_argpos[idx]]
                    )
                    args = tuple(arg_list)

            res = function(self, *args, **kwargs)

            # return value 1D or scalar when boolean set
            if input_dims == 1:
                # handle functions with multiple return values
                if type(res) is tuple:
                    returnvalues = list(res)
                    returnvalues = [o.flatten() for o in returnvalues]
                    if return_scalar:
                        returnvalues = [o[0] for o in returnvalues]
                    return tuple(returnvalues)

                elif return_scalar and np.ndim(res) != 0:
                    return res.flatten()[0]
                elif np.ndim(res) != 0:
                    return res.flatten()

            return res

        return wrapper

    return decorator


def plot_with_extra_data(
    surrogate,
    n_samples: int = int(1e4),
    title: Optional[str] = None,
    X=None,
    plot_data: bool = False,
    highlight_data: Optional[list] = None,
    plot_style: Optional[dict] = None,
    figure_size: tuple[int] = (6, 6),
    extra_data: Optional[np.ndarray] = None,
    parameter_transformer=None,
    subspace: int = None,
):
    """
    Same as plot. Except here extra_data can be added for plotting. Mostly
    for debugging purpose.
    """
    if parameter_transformer is None:

        class IndentityTransform:
            def __call__(self, X):
                return X

            def inverse(self, X):
                return X

        parameter_transformer = IndentityTransform()
    # generate samples
    if isinstance(surrogate, np.ndarray):
        Xs = surrogate
    else:
        Xs = surrogate.sample(
            n_samples
        )  # For flows the samples are in transformed space
    if isinstance(Xs, torch.Tensor):
        Xs = Xs.cpu().detach().numpy()
    Xs = parameter_transformer.inverse(Xs)

    D = Xs.shape[1]
    if subspace is not None:
        D = subspace
    Xs = Xs[:, :D]

    # cornerplot with samples of the surrogate
    fig = plt.figure(figsize=figure_size, dpi=100)
    labels = [f"$x_{i}$" for i in range(D)]
    corner_style = {"fig": fig, "labels": labels}

    if plot_style is None:
        plot_style = {}

    if "corner" in plot_style:
        corner_style.update(plot_style.get("corner"))

    # suppress warnings for small datasets with quiet=True
    fig = corner.corner(Xs, quiet=True, **corner_style)

    # style of the gp data
    data_style = {"s": 15, "color": "blue", "facecolors": "none"}

    if "data" in plot_style:
        data_style.update(plot_style.get("data"))

    highlighted_data_style = {
        "s": 15,
        "color": "orange",
    }

    if "highlight_data" in plot_style:
        highlighted_data_style.update(plot_style.get("highlight_data"))

    axes = np.array(fig.axes).reshape((D, D))

    # plot gp data
    if X is not None:
        # highlight nothing when argument is None
        if highlight_data is None or len(highlight_data) == 0:
            highlight_data = np.array([False] * len(X))
            normal_data = ~highlight_data
        else:
            normal_data = [i for i in range(len(X)) if i not in highlight_data]

        orig_X_norm = parameter_transformer.inverse(X[normal_data])
        orig_X_highlight = parameter_transformer.inverse(X[highlight_data])

        for r in range(1, D):
            for c in range(D - 1):
                if r > c:
                    if plot_data:
                        axes[r, c].scatter(
                            orig_X_norm[:, c], orig_X_norm[:, r], **data_style
                        )
                    axes[r, c].scatter(
                        orig_X_highlight[:, c],
                        orig_X_highlight[:, r],
                        **highlighted_data_style,
                    )

    # style of the gp data
    extra_data_style = {"s": 15, "color": "red"}
    if "extra_data" in plot_style:
        extra_data_style.update(plot_style.get("extra_data"))
    if extra_data is not None:
        orig_extra_data = parameter_transformer.inverse(extra_data)
        orig_extra_data = orig_extra_data[:, :D]

        for r in range(1, D):
            for c in range(D - 1):
                if r > c:
                    axes[r, c].scatter(
                        orig_extra_data[:, c],
                        orig_extra_data[:, r],
                        **extra_data_style,
                    )

    if title is not None:
        fig.suptitle(title)

    # adjust spacing between subplots
    fig.tight_layout(pad=0.5)

    return fig


def get_torch_model_size(model):
    param_num = 0
    param_size = 0
    for param in model.parameters():
        param_num += param.nelement()
        param_size += param.nelement() * param.element_size()
    buffer_num = 0
    buffer_size = 0
    for buffer in model.buffers():
        buffer_num += buffer.nelement()
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    # print('model size: {:.3f}MB'.format(size_all_mb))
    return size_all_mb, param_num, buffer_num


def wandb_log(data: dict):
    if wandb.run is not None and not isinstance(
        wandb.run, wandb.sdk.lib.disabled.RunDisabled
    ):
        wandb.log(data)
    else:
        # Print dictionary as string
        print(str(data))
