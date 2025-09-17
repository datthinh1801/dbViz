import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
from matplotlib.colors import LinearSegmentedColormap

from .utils import get_predictions


def plot_decision_boundaries(
    model: nn.Module,
    pivot_labels: list[torch.Tensor],
    plane_loader: data.DataLoader,
    num_classes: int,
    plot_pivots: bool = True,
    plane_size: int = 500,
    dpi: int = 240,
    device: torch.device = torch.device("cpu"),
):
    """
    Plot the decision boundaries of a model on a plane.

    Args:
        model (Union[nn.Module, pl.LightningModule]): The model to plot the decision boundaries of.
        pivot_labels (list[torch.Tensor]): The labels of the pivot samples.
        plane_loader (data.DataLoader): The data loader for the plane.
        num_classes (int): The number of classes.
        plot_pivots (bool): Whether to plot the pivot samples.
        plane_size (int): The size of the plane.
        dpi (int): The resolution of the figure in dots per inch.
    Returns:
        plt.Figure: The plot figure.
    """
    col_map = plt.colormaps["tab10"]
    cmaplist = [col_map(i) for i in range(num_classes)]
    col_map = LinearSegmentedColormap.from_list(
        "decision_boundary_colormap", cmaplist, N=num_classes
    )

    fig, ax = plt.subplots(dpi=dpi)

    # make region
    xx = plane_loader.dataset.coeff_x.reshape(plane_size, plane_size)
    yy = plane_loader.dataset.coeff_y.reshape(plane_size, plane_size)
    zz = get_predictions(model, plane_loader, device).reshape(plane_size, plane_size)
    zz = zz.cpu().numpy()
    ax.scatter(xx, yy, c=zz, alpha=0.8, s=0.1)

    # scatter plot the pivot samples
    if plot_pivots:
        ax.scatter(
            plane_loader.dataset.x_coords,
            plane_loader.dataset.y_coords,
            c=pivot_labels,
            cmap=col_map,
            edgecolors="k",
        )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.margins(0, 0)
    return fig
