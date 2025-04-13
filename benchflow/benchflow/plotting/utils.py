import matplotlib.pyplot as plt
import numpy as np
from corner import corner


def corner_plot(gt_samples, algo_samples, title=None, txt=None, save_as=None):
    fig = corner(
        gt_samples,
        color="tab:orange",
        hist_kwargs={"density": True},
    )
    corner(
        algo_samples,
        fig=fig,
        color="tab:blue",
        contour_kwargs=dict(linestyles="dashed"),
        hist_kwargs={"density": True},
    )
    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=12)
    lgd = fig.legend(
        labels=["Ground truth", "Flow"],
        loc="upper center",
        bbox_to_anchor=(0.8, 0.8),
    )
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    if txt is not None:
        text_art = fig.text(
            0.0, -0.10, txt, wrap=True, horizontalalignment="left", fontsize=12
        )
        extra_artists = (text_art, lgd)
    else:
        extra_artists = (lgd,)
    if save_as is not None:
        fig.savefig(
            save_as,
            dpi=300,
            bbox_extra_artists=extra_artists,
            bbox_inches="tight",
        )
    return fig


def corner_plot_with_data(
    samples,
    title=None,
    train_X=None,
    highlight_data=None,
    plot_style=None,
    figure_size=None,
    extra_data=None,
):
    D = samples.shape[1]
    if figure_size is None:
        figure_size = (3 * D, 3 * D)
    fig = plt.figure(figsize=figure_size, dpi=100)
    labels = ["$x_{}$".format(i) for i in range(D)]
    corner_style = dict({"fig": fig, "labels": labels})

    if plot_style is None:
        plot_style = dict()

    if "corner" in plot_style:
        corner_style.update(plot_style.get("corner"))

    # suppress warnings for small datasets with quiet=True
    fig = corner(samples, quiet=True, **corner_style)

    # style of the gp data
    data_style = dict({"s": 15, "color": "blue", "facecolors": "none"})

    if "data" in plot_style:
        data_style.update(plot_style.get("data"))

    highlighted_data_style = dict(
        {
            "s": 15,
            "color": "orange",
        }
    )
    axes = np.array(fig.axes).reshape((D, D))

    # plot train data
    if train_X is not None:
        # highlight nothing when argument is None
        if highlight_data is None or highlight_data.size == 0:
            highlight_data = np.array([False] * len(train_X))
            normal_data = ~highlight_data
        else:
            normal_data = [
                i for i in range(len(train_X)) if i not in highlight_data
            ]

        orig_X_norm = train_X[normal_data]
        orig_X_highlight = train_X[highlight_data]

        for r in range(1, D):
            for c in range(D - 1):
                if r > c:
                    axes[r, c].scatter(
                        orig_X_norm[:, c], orig_X_norm[:, r], **data_style
                    )
                    axes[r, c].scatter(
                        orig_X_highlight[:, c],
                        orig_X_highlight[:, r],
                        **highlighted_data_style,
                    )

    if title is not None:
        fig.suptitle(title)

    # adjust spacing between subplots
    fig.tight_layout(pad=0.5)
    return fig
