# dbViz — Decision Boundary Visualizer

Minimal install and quickstart to plot decision boundaries for PyTorch models.

## Installation

```bash
uv sync
```

## Quickstart

A tiny example that selects three samples, builds a plane loader, and plots decision boundaries:

```python
from dbviz.utils import get_random_samples, make_plane_loader
from dbviz.plot import plot_decision_boundaries
import matplotlib.pyplot as plt

# pick three samples from your dataset (example: CIFAR-10 testset)
cifar10_samples, cifar10_labels = get_random_samples(testset_cifar10)

# build a plane loader
plane_loader = make_plane_loader(cifar10_samples, batch_size=256, plane_size=500)

# plot and save
fig = plot_decision_boundaries(model, cifar10_labels, plane_loader, num_classes=len(classes), plane_size=500)
fig.savefig('decision_boundaries_cifar10.png')
plt.show()
```

That's it — `uv sync` to install deps, then run the quickstart snippet after you load your model and dataset.

