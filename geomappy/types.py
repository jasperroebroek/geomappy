from typing import Literal, Protocol

import matplotlib.axes
from matplotlib.colorbar import Colorbar
from matplotlib.colorizer import ColorizingArtist
from matplotlib.legend import Legend
from numpy.typing import ArrayLike

LegendType = Colorbar | Legend
ExtendType = Literal['neither', 'both', 'min', 'max']
GridSpacer = float | tuple[float, float] | tuple[tuple[float], tuple[float]]


class LegendCreator(Protocol):
    def __call__(
        self,
        ax: matplotlib.axes.Axes,
        ca: ColorizingArtist,
        legend_ax: matplotlib.axes.Axes | None,
        labels: ArrayLike | None,
    ) -> LegendType | None: ...
