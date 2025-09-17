# Decision Boundary Plotter

A simple library to plot the decision boundaries of a PyTorch model on a given set of 3 data points.

## Installation

This library requires `torch` and `matplotlib`. You can install them using pip:

```bash
pip install torch matplotlib
```

## Usage

The main function is `plot_decision_boundary`, which takes a PyTorch model and 3 data points as input and returns a `matplotlib.figure.Figure` object.

### `plot_decision_boundary`

```python
import torch
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
```

**Arguments**:

*   `model` (torch.nn.Module): The PyTorch model to plot.
*   `data_points` (Union[torch.Tensor, List[torch.Tensor]]): A list of 3 tensors or a single tensor of shape (3, C, H, W).
*   `resolution` (int, optional): The resolution of the plot. Defaults to 500.
*   `plot_range_expansion_l` (float, optional): The left expansion of the plot range. Defaults to 0.1.
*   `plot_range_expansion_r` (float, optional): The right expansion of the plot range. Defaults to 0.1.
*   `device` (str, optional): The device to run the model on. Defaults to 'cpu'.

**Returns**:

*   `matplotlib.figure.Figure`: The figure object containing the plot.

### Example

The following example shows how to use the library to plot the decision boundary of a dummy model.

```python
import torch
from decision_boundary_plotter import plot_decision_boundary

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
```
