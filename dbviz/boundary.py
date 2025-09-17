from typing import List

import torch


def decision_boundary(
    net: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Get the model predictions for all points in the plane defined by the DataLoader.

    Args:
        net (torch.nn.Module): The PyTorch model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader that provides points in the plane.
        device (torch.device): The device to run the model on.

    Returns:
        List[torch.Tensor]: A list of model predictions for each point.
    """
    net.eval()
    predicted_labels = []
    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            for output in outputs:
                predicted_labels.append(output)
    return predicted_labels
