# dbViz — Decision Boundary Visualizer

Minimal install and quickstart to plot decision boundaries for PyTorch models.

## Installation

```bash
uv sync
```

## Quickstart

A tiny example that selects three samples, builds a plane loader, and plots decision boundaries:

```python
import torch
import torchvision
import torchvision.transforms as transforms
from dbviz.utils import get_random_samples, make_plane_loader
from dbviz.plot import plot_decision_boundaries
import matplotlib.pyplot as plt

# 1. Load a dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Load a model (using an untrained model for this example)
model = torchvision.models.resnet18(weights=None, num_classes=10)
model.eval()

# 3. Pick three samples from your dataset
cifar10_samples, cifar10_labels = get_random_samples(testset)

# 4. Build a plane loader
plane_loader = make_plane_loader(cifar10_samples, batch_size=256, plane_size=500)

# 5. Plot and save
fig = plot_decision_boundaries(model, cifar10_labels, plane_loader, num_classes=len(classes), plane_size=500)
fig.savefig('decision_boundaries_cifar10.png')
plt.show()
```

## Acknowledgement
This package is inspired by:
- [dbViz](https://github.com/somepago/dbViz)