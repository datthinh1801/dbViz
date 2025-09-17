from dataclasses import dataclass
from typing import Tuple

import torch
import torch.utils.data as data


@dataclass
class Span:
    """
    A dataclass representing a span of 2 vectors
    """

    v1: torch.Tensor
    v2: torch.Tensor


@dataclass
class Coordinate:
    """
    A dataclass representing a coordinate in a plane.
    """

    x: torch.Tensor
    y: torch.Tensor

    def __post_init__(self):
        assert self.x.dim() == 0, "x must be a scalar tensor"
        assert self.y.dim() == 0, "y must be a scalar tensor"


@dataclass
class Bound:
    """
    A dataclass representing the bounds of a plane.
    """

    min_value: torch.Tensor
    max_value: torch.Tensor


class PlaneDataset(data.Dataset):
    """
    A dataset class representing a data plane for visualisation.
    """

    def __init__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        plane_size: int = 100,
        range_l: float = 0.1,
        range_r: float = 0.1,
    ) -> None:
        """
        Initialize the PlaneDataset with 3 base samples and a step size.
        The base samples are used to make the plane using Gram-Schmidt orthogonalization.

        Args:
            x1 (torch.Tensor): a base sample
            x2 (torch.Tensor): a sample that produces that first orthogonal vector of the plane
            x3 (torch.Tensor): a sample that produces the second orthogonal vector of the plane
            steps (float): step size to make the grid of samples in the plane
            range_l (float): a range factor to extend the plane to the minimum side
            range_r (float): a range factor to extend the plane to the maximum side
        """
        super().__init__()
        self.base_x = x1
        self.span, self.coords = self._get_plane(x1, x2, x3)

        self.x_coords = torch.tensor([coord.x for coord in self.coords])
        self.y_coords = torch.tensor([coord.y for coord in self.coords])

        x_bound = Bound(torch.min(self.x_coords), torch.max(self.x_coords))
        y_bound = Bound(torch.min(self.y_coords), torch.max(self.y_coords))

        x_len = x_bound.max_value - x_bound.min_value
        y_len = y_bound.max_value - y_bound.min_value

        grid = torch.meshgrid(
            [
                torch.linspace(
                    start=(x_bound.min_value - range_l * x_len).item(),
                    end=(x_bound.max_value + range_r * x_len).item(),
                    steps=plane_size,
                ),
                torch.linspace(
                    start=(y_bound.min_value - range_l * y_len).item(),
                    end=(y_bound.max_value + range_r * y_len).item(),
                    steps=plane_size,
                ),
            ]
        )

        self.coeff_x = grid[0].flatten()
        self.coeff_y = grid[1].flatten()

    def __len__(self) -> int:
        return len(self.coeff_x)

    def __getitem__(self, index: int) -> torch.Tensor:
        return (
            self.base_x
            + self.coeff_x[index] * self.span.v1
            + self.coeff_y[index] * self.span.v2
        )

    @staticmethod
    def _get_plane(
        x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
    ) -> Tuple[Span, list[Coordinate]]:
        """
        Gram-Schmidt orthogonalization of 2 vectors from 3 base samples.

        Args:
            x1 (torch.Tensor): a base sample
            x2 (torch.Tensor): a sample
            x3 (torch.Tensor): a sample

        Returns:
            Tuple[Span, list[Coordinate]]: A tuple containing the orthogonal basis vectors for the plane
            and the coordinates of the three input samples in that plane.
        """
        v1 = x2 - x1
        v2 = x3 - x1

        # u1 = v1 / ||v1||
        v1_norm = torch.dot(v1.flatten(), v1.flatten()).sqrt()
        u1 = v1 / v1_norm  # basis vector 1

        # v2 = c1 * u1 + c2 * u2 = c1 * u1 + c2 * u2
        # c1 = v2.v1 / v1.v1 = v2.u1
        c1 = torch.dot(u1.flatten(), v2.flatten())  # orthogonal projection of v2 on u1

        v2_orthog = v2 - c1 * u1  # component of v2 orthogonal to u1
        v2_orthog_norm = torch.dot(v2_orthog.flatten(), v2_orthog.flatten()).sqrt()
        u2 = v2_orthog / v2_orthog_norm  # basis vector 2

        # c2 = v2.v2_orthog / v2_orthog.v2_orthog = v2.u2
        c2 = torch.dot(v2.flatten(), u2.flatten())

        span = Span(u1, u2)

        # co-ordinates of three samples in the new basis
        # coords = [[0, 0], [v1_norm, 0], [c1, c2]]
        coords = [
            Coordinate(torch.scalar_tensor(0), torch.scalar_tensor(0)),
            Coordinate(v1_norm, torch.scalar_tensor(0)),
            Coordinate(c1, c2),
        ]
        return span, coords
