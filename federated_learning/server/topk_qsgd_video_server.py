import os
import torch
import numpy as np
from functools import reduce
from collections import OrderedDict

from flwr.common.typing import FitIns, FitRes, Parameters
from flwr.common import (
    ndarray_to_bytes, bytes_to_ndarray,
    weights_to_parameters
)
from qsgd_video_server import QSGDVideoServer
from ..utils import qsgd

