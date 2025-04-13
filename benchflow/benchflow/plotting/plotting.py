from pathlib import PurePosixPath

import matplotlib.pyplot as plt
import numpy as np
import yaml

from benchflow.utilities.io import load_result


def plot_results(plot_yaml, sources=None, save_figs=True):
    """Plot the results from ``benchflow`` run(s).

    Parameters
    ----------
    plot_yaml : str
        The path to the ``.yaml`` file describing what figures to plot. (See
        ``example.yaml`` in ``benchflow/plotting``).
    sources : str or [str], optional
        The path(s) to the directories containing the data from ``benchflow``
        run(s). If not provided, should be included in the ``.yaml`` file under
        the key ``sources``.
    save_figs : bool
        Whether to save the figures to files.
    """
    with open(plot_yaml, "r") as f:
        yaml_dict = yaml.full_load(f)
    # Dummy plot to avoid over-writing rcParams
    plt.plot([], [])
    plt.close()

    output = yaml_dict.get("defaults", {}).get("output", "test.svg")
    output = PurePosixPath(output)
    fig_num = 0
    for plot in yaml_dict["figures"]:
        current_options = get_current_options(sources, yaml_dict, plot)
        # Set overall plotting options:
        for key, val in current_options.get("rcParams", {}).items():
            plt.rcParams[key] = val
        fig_size = current_options.get("size")
        if fig_size is not None:
            plt.rcParams["figure.figsize"] = fig_size

        plot_type = list(plot.keys())[0]
        if "subplot" in plot_type:  # Multiple subplots
            __, rows, cols = plot_type.split(" ")
            rows, cols = int(rows), int(cols)
            fig, axes = plt.subplots(rows, cols, squeeze=False)

            # Record artist handles and labels for plotting legend:
            handles = {}
            legend_idx = None
            for subplot in list(plot.values())[0]["axes"]:
                row, col = list(subplot.keys())[0].split(" ")
                row, col = int(row), int(col)
                current_options, handles, legend_idx = plot_axis(
                    axes,
                    sources,
                    yaml_dict,
                    plot,
                    subplot,
                    row,
                    col,
                    handles=handles,
                    legend_idx=legend_idx,
                )
            finish_subplot(axes, rows, cols, handles, legend_idx)
        else:
            # Single plot, instead of subplots
            fig, axes = plt.subplots(squeeze=False)
            for algorithm in list(plot.values())[0].get("algorithms", [None]):
                current_options, __, __ = plot_axis(
                    axes, sources, yaml_dict, plot, algorithm
                )
        plt.suptitle(current_options.get("title"))
        plt.tight_layout()

        # Save plot to file, if requested:
        current_filename = current_options.get("filename")
        if current_filename is None and save_figs:
            fig.savefig(output.stem + "-" + str(fig_num) + output.suffix)
        elif save_figs:
            fig.savefig(current_filename)
        fig_num += 1


def plot_axis(
    axes,
    sources,
    yaml_dict,
    plot,
    subplot=None,
    row=0,
    col=0,
    handles={},
    legend_idx=None,
):
    """Plot a single axis in a plot/subplot.

    axes : plt.Axes
        The set of axes on which to plot.
    sources : str or [str], optional
        The path(s) to the directories containing the data from ``benchflow``
        run(s). If not provided, should be included in the ``.yaml`` file under
        the key ``sources``.
    yaml_dict : dict
        The entire yaml dictionary describing all figures and defaults.
    plot : dict
        The subset of the yaml dictionary describing the current plot.
    subplot : dict, optional
        The subset of the yaml dictionary describing the current subplot.
        Default empty.
    row : int, optional
        The index of the current subplot row. Default 0.
    col : int, optional
        The index of the current subplot col. Default 0.
    handles : dict, optional
        A dictionary where keys are string names of plotted algorithms, and
        values are tuples of associated plot artists. Used if legend is to be
        plotted in a separate panel. Should start empty (used for bookkeeping).
    legend_idx : (int, int), optional
        The tuple of indices specifying the index of the separate legend, if
        specified. Should start empty (used for bookkeeping).
    """
    if subplot is None:
        current_options = get_current_options(sources, yaml_dict, plot)
    else:
        current_options = get_current_options(
            sources, yaml_dict, plot, subplot
        )

    if current_options.get("legend") == "here":
        # Insert the legend here
        axes[row, col].set_frame_on(False)
        axes[row, col].get_xaxis().set_visible(False)
        axes[row, col].get_yaxis().set_visible(False)
        return current_options, handles, (row, col)
    if current_options.get("empty"):
        # Don't plot anything
        axes[row, col].set_frame_on(False)
        axes[row, col].get_xaxis().set_visible(False)
        axes[row, col].get_yaxis().set_visible(False)
        return current_options, handles, legend_idx
    else:
        # Otherwise plot a sub-plot
        for algorithm in current_options.get("algorithms", [None]):
            results = flatten_results(load_results(current_options["sources"]))
            data, current_tasks = extract_metrics(
                results,
                current_options["xaxis"],
                current_options["yaxis"],
                algorithm,
                current_options.get("task"),
            )

            if data == []:
                continue
            draw = current_options.get("draw", "all, median")
            if "all" in draw:  # Plot all lines individually
                for x, y in data:
                    (h,) = axes[row, col].plot(x, y)
                    if not handles.get(algorithm):
                        handles[algorithm] = set([h])
            if "quantile" in draw:  # Fill between quantiles
                lq, hq = current_options.get("quantiles", (0.05, 0.95))
                x, qs = get_quantiles(data, lq, hq)

                default_color = current_options.get("algo_colors", {}).get(
                    algorithm
                )
                quantile_color = current_options.get(
                    "quantile_color", default_color
                )
                alpha = current_options.get("quantile_alpha", 0.333)
                label = current_options.get(
                    "quantile_label", f"{lq:.1%} to {hq:.1%} Quantiles"
                )
                h = axes[row, col].fill_between(
                    x,
                    qs[0, :],
                    qs[1, :],
                    color=quantile_color,
                    label=label,
                    alpha=alpha,
                )
                update_legend(handles, algorithm, "patch", h, label)
            if "median" in draw:  # Plot median in bold
                x, y = get_quantiles(data, 0.5)

                default_color = current_options.get("algo_colors", {}).get(
                    algorithm
                )
                median_color = current_options.get(
                    "median_color", default_color
                )
                median_linewidth = current_options.get("median_width", 3)
                label = current_options.get("median_label", "Median")
                (h,) = axes[row, col].plot(
                    x,
                    y[0, :],
                    color=median_color,
                    linewidth=median_linewidth,
                    label=label,
                )
                update_legend(handles, algorithm, "line", h, label)
            if "confidence" in draw:
                # Bootstrap confidence intervals for median
                lci, hci = current_options.get(
                    "confidence_intervals", (0.025, 0.975)
                )
                xs = data[0][0]
                ys = np.vstack([y for __, y in data])
                n_bootstrap = current_options.get("n_bootstrap", 10000)
                medians = np.zeros((n_bootstrap, ys.shape[1]))
                for n in range(n_bootstrap):
                    idx = np.random.choice(
                        ys.shape[0], size=ys.shape[0], replace=True
                    )
                    medians[n] = np.median(ys[idx], axis=0)
                qs = np.quantile(medians, (lci, hci), axis=0)
                default_color = current_options.get("algo_colors", {}).get(
                    algorithm
                )
                ci_color = current_options.get("ci_color", default_color)
                alpha = current_options.get("ci_alpha", 0.333)
                label = current_options.get(
                    "ci_label", f"{lci:.1%} to {hci:.1%} Median C.I."
                )
                h = axes[row, col].fill_between(
                    xs,
                    qs[0, :],
                    qs[1, :],
                    color=ci_color,
                    label=label,
                    alpha=alpha,
                )
                update_legend(handles, algorithm, "patch", h, label)

        if current_options.get("ref_line"):
            ref_line_color = current_options.get("ref_line_color", "black")
            axes[row, col].axhline(
                current_options["ref_line"],
                color=ref_line_color,
                linestyle="dashed",
            )
        # Set subplot options
        yscale = current_options.get("yscale", None)
        xscale = current_options.get("xscale", None)
        if yscale:
            axes[row, col].set_yscale(yscale)
        if xscale:
            axes[row, col].set_xscale(xscale)

        current_title = current_options.get("subplot_title")
        if current_title is None:
            current_title = ", ".join(list(map(pretty_print, current_tasks)))
        current_xlabel = current_options.get("xlabel")
        if current_xlabel is None:
            current_xlabel = pretty_print(
                current_options.get("xaxis", "Iterations")
            )
        current_ylabel = current_options.get("ylabel")
        if current_ylabel is None:
            current_ylabel = pretty_print(
                current_options["yaxis"], abbreviate=True
            )
        axes[row, col].set_title(current_title)
        axes[row, col].set_xlabel(current_xlabel)
        axes[row, col].set_ylabel(current_ylabel)

        if current_options.get("legend", True):
            axes[row, col].legend(*get_legend(handles))

        return current_options, handles, legend_idx


def finish_subplot(axes, rows, cols, handles, legend_idx):
    """Finalize a set of subplots.

    Parameters
    ----------
    axes : plt.Axes
        The set of axes on which to plot.
    rows : int
        The total number of rows in the subplots.
    cols : int, optional
        The total number of columns in the subplots.
    handles : dict
        A dictionary where keys are string names of plotted algorithms, and
        values are tuples of associated plot artists. Used if legend is to be
        plotted in a separate panel (if ``legend_idx`` is not ``None``).
    legend_idx : (int, int) or None
        The tuple of indices specifying the index of the separate legend, if
        specified.
    """
    # Plot separate legend, if needed
    if legend_idx is not None:
        row, col = legend_idx
        axes[row, col].legend(*get_legend(handles), loc="upper left")
    # Erase duplicate x,y-axis labels:
    for c in range(cols):
        unique_col_labels = len(
            set(
                [
                    axes[r, c].get_xlabel()
                    for r in range(rows)
                    if axes[r, c].get_xlabel() != ""
                ]
            )
        )
        if unique_col_labels == 1:
            labeled = False
            for r in range(rows - 1, -1, -1):
                if axes[r, c].get_xlabel() != "" and not labeled:
                    labeled = True
                else:
                    axes[r, c].set_xlabel(None)
    for r in range(rows):
        unique_row_labels = len(
            set(
                [
                    axes[r, c].get_ylabel()
                    for c in range(cols)
                    if axes[r, c].get_ylabel() != ""
                ]
            )
        )
        if unique_row_labels == 1:
            labeled = False
            for c in range(cols):
                if axes[r, c].get_ylabel() != "" and not labeled:
                    labeled = True
                else:
                    axes[r, c].set_ylabel(None)


def get_quantiles(xys, *args):
    """Get quantiles from a pair of x and y axis values."""
    if xys == []:
        return [], []
    xs = xys[0][0]
    ys = np.vstack([y for __, y in xys])
    qs = np.quantile(ys, args, axis=0)
    return xs, qs


def get_current_options(sources, yaml_dict, plot, subplot={"": {}}):
    """Get options hierarchically from yaml dictionary.

    Priority of option values increases with specificity:
        ``yaml_dict`` < ``plot`` < ``subplot``
    """
    subplot_options = list(subplot.values())[0]
    plot_options = list(plot.values())[0]
    global_options = yaml_dict.get("defaults", {})

    current_options = {}
    for option in (
        list(subplot_options.keys())
        + list(plot_options.keys())
        + list(global_options.keys())
    ):
        value = subplot_options.get(
            option, plot_options.get(option, global_options.get(option))
        )
        if value is not None:
            current_options[option] = value
    if sources is None:
        value = subplot_options.get(
            "sources",
            plot_options.get("sources", global_options.get("sources")),
        )
        if value is not None:
            current_options["sources"] = value
    else:
        current_options["sources"] = sources

    if current_options.get("algo_colors") is None:
        algo_colors = {}
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, algo in enumerate(current_options.get("algorithms", [None])):
            algo_colors[algo] = color_cycle[i]
        current_options["algo_colors"] = algo_colors
    return current_options


def extract_metrics(result, xaxis, metric, algorithm, task):
    """Extract the appropriate values for plotting a given metric and task."""
    data = []
    found_tasks = set()
    for r in result:
        right_algorithm = (
            algorithm is None or r["config"]["algorithm"]["class"] == algorithm
        )
        right_task = task is None or r["config"]["task"]["name"] == task
        if right_algorithm and right_task:
            if xaxis is None:
                y = r["metrics"][metric]
                data.append((range(len(y)), y))
            else:
                xvals = r["metrics"].get(xaxis, [])
                data.append(dedup_x(xvals, r["metrics"][metric]))
            found_tasks.add(r["config"]["task"]["name"])
    return data, found_tasks


def pretty_print(string, abbreviate=False):
    """Pretty-print a string, possibly returning an abbreviated version."""
    maps = {
        "lml": ("Log Marginal Likelihood", "LML"),
        "lml_sd": ("LML Standard Deviation", "LML Std. Dev."),
        "lml_error": ("Absolute LML Error", "LML Loss"),
        "mmtv": ("Mean Marginal Total Variation", "MMTV"),
        "mtv": ("Mean Marginal Total Variation", "MMTV"),
        "gskl": ("Gaussianized Symmetrized K-L Divergence", "gsKL"),
        "fun_evals": ("Function Evaluations", "Fun. Evals."),
        # Tasks:
        "goris": "Neuronal (V1)",
        "goris_alternate": "Neuronal (V2)",
        "rosenbrock": "Rosenbrock",
        "lumpy": "Lumpy",
        "cigar": "Cigar",
        "student_t": "Student's T",
    }

    strings = maps.get(string)
    if strings is None:
        return string
    elif type(strings) == str:
        return strings
    else:
        if abbreviate:
            return strings[1]
        else:
            return strings[0]


def load_results(paths):
    """Load ``benchflow`` results from a path or list of paths."""
    results = []
    if type(paths) != list:
        paths = [paths]
    for path in paths:
        results.append(load_result(dirpath=path))
    return results


def flatten_results(results):
    """Flatten a list of results to share the same structure.

    Results from a ``benchflow`` multirun are structured with an "extra"
    top-level key corresponding to the sub-run that yielded them. Since the
    result's ``"config"`` key contains all relevant information anyway, we
    discard the sub-run index and flatten the dictionary to a list.
    """
    flat_results = []
    for i, res in enumerate(results):
        if res.get("config") is not None:
            res["run"] = i
            flat_results.append(res)
        else:
            for r in res.values():
                r["run"] = i
                flat_results.append(r)
    return flat_results


def dedup_x(x, y, callable_=np.mean):
    """De-duplicate repeated x-values."""
    values = {}
    y = np.array(y)

    if y.ndim == 2:  # Take mean of mmtv metric
        y = np.mean(y, axis=1)

    # Stack y-values according to (possibly duplicate) x-value keys
    for i, xx in enumerate(x):
        values[xx] = values.get(xx, []) + [y[i]]
    # Average the duplicate y-values (default: mean)
    for key, val in values.items():
        values[key] = callable_(val)

    # Return as 1-to-1 arrays
    new_ys = np.array(
        [q for _, q in sorted(zip(values.keys(), values.values()))]
    )
    xs = np.array(sorted(values.keys()))
    return xs, new_ys


def update_legend(handles, algorithm, handle_type, h, label):
    if not handles.get(algorithm):
        handles[algorithm] = {}
    if not handles[algorithm].get(handle_type):
        handles[algorithm][handle_type] = h
    if not handles[algorithm].get(label):
        handles[algorithm][label] = set([label])
    else:
        handles[algorithm][label].add(label)


def get_legend(handles):
    artist_tuples = []
    for val in handles.values():
        artists = [val.get("line"), val.get("patch")]
        artist_tuples.append(tuple([a for a in artists if a is not None]))
    labels = list(map(pretty_print, handles.keys()))
    return artist_tuples, labels
