from enum import Enum
from typing import NewType, Union, Iterable, Optional, TypeVar

import matplotlib
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap

Color = TypeVar('Color', str, Iterable)
ColorOrMap = NewType('ColorOrMap', Union[str, Iterable, Colormap])
Number = TypeVar('Number', int, float)
OptionalLegend = NewType("OptionalLegend", Optional[Union[Colorbar, matplotlib.legend.Legend]])


class Legend(Enum):
    NoLegend = 0
    Legend = 1
    Colorbar = 2
