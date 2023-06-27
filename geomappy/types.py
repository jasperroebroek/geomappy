from enum import Enum
from typing import TypeVar, Tuple

Color = TypeVar('Color', str, Tuple[float, float, float], Tuple[float, float, float, float])
Colormap = TypeVar('Colormap')
LegendOrColorbar = TypeVar("LegendOrColorbar")


class Legend(Enum):
    NoLegend = 0
    Legend = 1
    Colorbar = 2
