"""
This is c3d model constructors interface
"""

from typing import *


def c3d(
	num_classes: Optional[int] = 487,
	dropout: Optional[float] = 0.25
	):
    """Construct original C3D network as described in [1].
    """
    ...


def c3d_bn(
	num_classes: Optional[int] = 400,
	dropout: Optional[float] = 0.25
	):
    """Construct the modified C3D network with batch normalization hosted in github Video Model Zoo.
    """
    ...
