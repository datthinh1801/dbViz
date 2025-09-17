import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
from typing import List, Union
import matplotlib

def plot_decision_boundary(
    model: torch.nn.Module,
    data_points: Union[torch.Tensor, List[torch.Tensor]],
    resolution: int = 500,
    plot_range_expansion_l: float = 0.1,
    plot_range_expansion_r: float = 0.1,
    device: str = 'cpu'
) -> matplotlib.figure.Figure:
    """
    Plots the decision boundary of a model for a given set of 3 data points.

    Args:
        model (torch.nn.Module): The PyTorch model to plot.
        data_points (Union[torch.Tensor, List[torch.Tensor]]): A list of 3 tensors or a single tensor
            of shape (3, C, H, W).
        resolution (int, optional): The resolution of the plot. Defaults to 500.
        plot_range_expansion_l (float, optional): The left expansion of the plot range. Defaults to 0.1.
        plot_range_expansion_r (float, optional): The right expansion of the plot range. Defaults to 0.1.
        device (str, optional): The device to run the model on. Defaults to 'cpu'.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    if isinstance(data_points, torch.Tensor):
        if data_points.shape[0] != 3:
            raise ValueError("Input tensor must have a batch size of 3.")
        images = [data_points[i] for i in range(3)]
    else:
        if len(data_points) != 3:
            raise ValueError("Input list must contain 3 tensors.")
        images = data_points

    model.to(device)

    planeloader = make_planeloader(images, resolution, plot_range_expansion_l, plot_range_expansion_r)
    preds = decision_boundary(model, planeloader, device)

    # Get the labels of the 3 images
    with torch.no_grad():
        image_tensors = torch.stack(images).to(device)
        labels = model(image_tensors).argmax(dim=1).cpu().numpy()

    fig = _plot_decision_boundary_internal(preds, planeloader, images, labels)
    return fig

def get_plane(img1, img2, img3):
    ''' Calculate the plane (basis vecs) spanned by 3 images
    Input: 3 image tensors of the same size
    Output: two (orthogonal) basis vectors for the plane spanned by them, and
    the second vector (before being made orthogonal)
    '''
    a = img2 - img1
    b = img3 - img1
    a_norm = torch.dot(a.flatten(), a.flatten()).sqrt()
    a = a / a_norm
    first_coef = torch.dot(a.flatten(), b.flatten())
    #first_coef = torch.dot(a.flatten(), b.flatten()) / torch.dot(a.flatten(), a.flatten())
    b_orthog = b - first_coef * a
    b_orthog_norm = torch.dot(b_orthog.flatten(), b_orthog.flatten()).sqrt()
    b_orthog = b_orthog / b_orthog_norm
    second_coef = torch.dot(b.flatten(), b_orthog.flatten())
    #second_coef = torch.dot(b_orthog.flatten(), b.flatten()) / torch.dot(b_orthog.flatten(), b_orthog.flatten())
    coords = [[0,0], [a_norm,0], [first_coef, second_coef]]
    return a, b_orthog, b, coords


class plane_dataset(torch.utils.data.Dataset):
    def __init__(self, base_img, vec1, vec2, coords, resolution=500,
                    range_l=.1, range_r=.1):
        self.base_img = base_img
        self.vec1 = vec1
        self.vec2 = vec2
        self.coords = coords
        self.resolution = resolution
        x_bounds = [coord[0] for coord in coords]
        y_bounds = [coord[1] for coord in coords]

        self.bound1 = [torch.min(torch.tensor(x_bounds)), torch.max(torch.tensor(x_bounds))]
        self.bound2 = [torch.min(torch.tensor(y_bounds)), torch.max(torch.tensor(y_bounds))]

        len1 = self.bound1[-1] - self.bound1[0]
        len2 = self.bound2[-1] - self.bound2[0]

        #list1 = torch.linspace(self.bound1[0] - 0.1*len1, self.bound1[1] + 0.1*len1, int(resolution))
        #list2 = torch.linspace(self.bound2[0] - 0.1*len2, self.bound2[1] + 0.1*len2, int(resolution))
        list1 = torch.linspace(self.bound1[0] - range_l*len1, self.bound1[1] + range_r*len1, int(resolution))
        list2 = torch.linspace(self.bound2[0] - range_l*len2, self.bound2[1] + range_r*len2, int(resolution))

        grid = torch.meshgrid([list1,list2])

        self.coefs1 = grid[0].flatten()
        self.coefs2 = grid[1].flatten()

    def __len__(self):
        return self.coefs1.shape[0]

    def __getitem__(self, idx):
        return self.base_img + self.coefs1[idx] * self.vec1 + self.coefs2[idx] * self.vec2

def make_planeloader(images, resolution, plot_range_expansion_l, plot_range_expansion_r):
    a, b_orthog, b, coords = get_plane(images[0], images[1], images[2])

    planeset = plane_dataset(images[0], a, b_orthog, coords, resolution=resolution, range_l=plot_range_expansion_l, range_r=plot_range_expansion_r)

    planeloader = torch.utils.data.DataLoader(
        planeset, batch_size=256, shuffle=False, num_workers=0)
    return planeloader


def decision_boundary(net, loader, device):
    net.eval()
    predicted_labels = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            inputs = inputs.to(device)
            outputs = net(inputs)
            for output in outputs:
                predicted_labels.append(output)
    return predicted_labels


def imscatter(x, y, image, ax=None, zoom=1):
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
    artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def _plot_decision_boundary_internal(preds, planeloader, images, labels, temp=1.0):
    from matplotlib.colors import LinearSegmentedColormap
    col_map = matplotlib.colormaps.get_cmap('tab10')
    cmaplist = [col_map(i) for i in range(col_map.N)]

    assert preds, "Predictions list cannot be empty."
    num_classes = preds[0].shape[0]
    classes = [f"Class {i}" for i in range(num_classes)]

    cmaplist = cmaplist[:num_classes]
    col_map = LinearSegmentedColormap.from_list('custom_colormap', cmaplist, N=num_classes)
    fig, ax1 = plt.subplots()
    import torch.nn as nn
    preds = torch.stack((preds))
    preds = nn.Softmax(dim=1)(preds / temp)
    val = torch.max(preds,dim=1)[0].cpu().numpy()
    class_pred = torch.argmax(preds, dim=1).cpu().numpy()
    x = planeloader.dataset.coefs1.cpu().numpy()
    y = planeloader.dataset.coefs2.cpu().numpy()
    label_color_dict = dict(zip([*range(num_classes)], cmaplist))

    color_idx = [label_color_dict[label] for label in class_pred]
    ax1.scatter(x, y, c=color_idx, alpha=val, s=0.1)

    coords = planeloader.dataset.coords

    for i, image in enumerate(images):
        img = image.cpu().numpy().transpose(1,2,0)
        # Clamp the image to [0, 1] range for visualization
        img = np.clip(img, 0, 1)

        if img.shape[0] > 32:
            from PIL import Image
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img).resize(size=(32, 32))
            img = np.array(img) / 255.0

        coord = coords[i]
        imscatter(coord[0], coord[1], img, ax1)

    patches = [mpatches.Patch(color=cmaplist[labels[i]], label=f'Image {i} (Class {labels[i]})') for i in range(len(labels))]
    plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    plt.title('Decision Boundary')

    return fig

if __name__ == '__main__':
    # Create a dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(3 * 32 * 32, 10)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            return self.fc1(x)

    # Create dummy data
    dummy_model = DummyModel()
    dummy_data = torch.rand(3, 3, 32, 32)

    # Plot the decision boundary
    fig = plot_decision_boundary(dummy_model, dummy_data)

    # Save the plot
    fig.savefig('decision_boundary_example.png')
    print("Saved example plot to decision_boundary_example.png")
