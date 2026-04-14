from typing import Any, Tuple, Literal

from TypeArrays import Shared
from numpy.typing import NDArray
from numpy import dtype


type DType[T: Any] = Shared.DType[T] | dtype[T]
type Shape[*Axis] = Shared.Shape[*Axis]
type Array[D: DType, S: Shape] = Shared.Array[D, S] | NDArray[D]

