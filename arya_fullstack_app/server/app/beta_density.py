from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist


@dataclass(frozen=True, slots=True)
class BetaDensity:
    """
    Segment size density based on a Beta(alpha, beta) distribution.

    The Beta distribution is defined on [0, 1] and is parameterised by two
    shape parameters (alpha, beta).  Here it models how "large" each customer
    segment is relative to the others — i.e. the density of demand at a given
    preference position.

    Common shapes:
      alpha < 1, beta < 1  →  U-shaped  (mass at extremes)
      alpha = beta = 1     →  uniform
      alpha > 1, beta > 1  →  unimodal bell  (mass in the middle)
      alpha < beta         →  skewed left
      alpha > beta         →  skewed right
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(f"alpha and beta must be positive, got ({self.alpha}, {self.beta})")

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        """Probability density at x (or array of x values in [0, 1])."""
        return beta_dist.pdf(x, self.alpha, self.beta)

    def density_at(self, x: float) -> float:
        """Scalar density at a single point x in [0, 1]."""
        return float(beta_dist.pdf(x, self.alpha, self.beta))

    def sample_grid(self, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (x, density) arrays over a uniform grid of n_points on [0, 1].

        Useful for plotting or numerical integration.
        """
        x = np.linspace(0.0, 1.0, n_points)
        return x, self.pdf(x)

    def segment_weights(self, segment_positions: list[float]) -> list[float]:
        """
        Given a list of positions in [0, 1] representing segment midpoints,
        return the un-normalised density weight at each position.

        Normalise by dividing by sum if you want probabilities.
        """
        return [self.density_at(p) for p in segment_positions]
