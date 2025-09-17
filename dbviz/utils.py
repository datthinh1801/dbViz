from typing import Tuple

import torch
import torch.nn as nn
import torch.utils.data as data

from .data import PlaneDataset


def get_random_samples(
    dataset: torch.utils.data.Dataset,
) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Get 3 random samples from 3 different classes from the dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to sample from.

    Returns:
        Tuple[list[torch.Tensor], list[torch.Tensor]]: A tuple containing a list of 3 sample tensors
        and their corresponding labels.
    """
    samples = []
    labels = []

    while len(samples) < 3:
        idx = torch.randint(0, len(dataset), (1,)).item()
        sample, label = dataset[idx]

        sample = sample.squeeze()
        # Ensure label is a tensor
        label = torch.tensor(label).squeeze()

        if label not in labels:
            samples.append(sample)
            labels.append(label)
    return samples, labels


def make_plane_loader(
    samples: list[torch.Tensor],
    batch_size: int,
    plane_size: int,
    range_l: float = 0.1,
    range_r: float = 0.1,
    num_workers: int = 0,
) -> data.DataLoader:
    """
    Make a plane from 3 samples. Then, create a data loader for the plane.

    Args:
        samples (list[torch.Tensor]): a list of 3 samples to make the plane
        batch_size (int): the batch size

    Returns:
        data.DataLoader: a data loader for the plane
    """
    assert len(samples) == 3, "The number of samples must be 3."
    x1, x2, x3 = samples
    plane_dataset = PlaneDataset(x1, x2, x3, plane_size, range_l, range_r)
    data_loader = data.DataLoader(
        plane_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return data_loader


def get_predictions(
    model: nn.Module,
    plane_loader: data.DataLoader,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """
    Get the predictions of a model on a plane.

    Args:
        model (nn.Module): The model to get the predictions of.
        plane_loader (data.DataLoader): The data loader for the plane.

    Returns:
        torch.Tensor: The predictions of the model on the plane.
    """
    model.eval()
    predictions = []
    for batch in plane_loader:
        with torch.no_grad():
            batch = batch.to(device)
            pred = model(batch)
            _, predicted_classes = torch.max(pred, dim=1)
            predictions.append(predicted_classes)
    return torch.cat(predictions, dim=0)
