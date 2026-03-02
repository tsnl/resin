import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import numpy.lib.stride_tricks as nps

from . import graph as rg


class Interp(ABC):
    pass
