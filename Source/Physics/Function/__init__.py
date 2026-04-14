from pathlib import Path
from typing import Any, Tuple, Union
from jaxtyping import Shaped, Array
from Physics.Units import ureg, PintUnit
from Physics.Context import fw

class Function: 
    pass

class Function1D(Function):
    _value_a: Array
    _value_u: PintUnit
    _axis__a: Array
    _axis__u: PintUnit

    def __init__(self, _value_a: Array, _value_u: PintUnit, _axis__a: Array, _axis__u: PintUnit):
        self._value_a, self._value_u, self._axis__a, self._axis__u = _value_a, _value_u, _axis__a, _axis__u
    
    def saveNPZ(self, path: Union[str, Path]):
        import numpy as np
        if not isinstance(self._value_a, np.ndarray) or not isinstance(self._axis__a, np.ndarray): raise RuntimeError()
        np.savez_compressed(path,
            value_a = self._value_a,
            axis__a = self._axis__a,
            value_u = str(self._value_u),
            axis__u = str(self._axis__u),
        )

    @classmethod
    def loadNPZ(cls, path: Union[str, Path]):
        import numpy as np
        with np.load(path) as data:
            return cls(
                _value_a = data["_value_a"],
                _value_u = data["_value_u"],
                _axis__a = data["_axis__a"],
                _axis__u = data["_axis__u"]
            )

class TimeFunc(Function1D):

    _axis__u: PintUnit = ureg.Unit("s")

    def __init__(self, values: Array, axis: Array, unit_values: PintUnit):
        self._value_a = values
        self._axis__a = axis
        self._value_u = unit_values

    @classmethod
    def from_Function1D(cls, func: Function1D):
        return cls(func._value_a, func._axis__a, func._value_u)
    
    @classmethod
    def from_zeros(cls, unit_values: PintUnit, dtype: Any, shape_t: int = 10):
        values = fw().zeros(shape_t, dtype=dtype)
        axis = fw().arange(shape_t)
        return cls(values=values, axis=axis, unit_values=unit_values)
    
    @property
    def time(self) -> Tuple[PintUnit, Array]:
        return self._axis__u, self._axis__a
    
    @time.setter
    def time(self, value: Array):
        self._value_a = value

    @property
    def values(self) -> Tuple[PintUnit, Array]:
        return self._value_u, self._value_a

    @values.setter
    def values(self, value: Array):
        self._value_a = value

    def saveNPZ(self, path: Union[str, Path]):
        Function1D.saveNPZ(self, path)
        
    @classmethod
    def loadNPZ(cls, path: Union[str, Path]):
        return cls.from_Function1D(Function1D.loadNPZ(path))

class Function2D(Function):
    _matrx_a: Shaped[Array, "X Y"]
    _matrx_u: PintUnit
    _first_a: Shaped[Array, "X"]
    _first_u: PintUnit
    _sec___a: Shaped[Array, "Y"]
    _sec___u: PintUnit

    def __init__(self, 
                 _matrx_a: Shaped[Array, "X Y"], _matrx_u: PintUnit, 
                 _first_a: Shaped[Array, "X"], _first_u: PintUnit, 
                 _sec___a: Shaped[Array, "Y"], _sec___u: PintUnit):
        self._matrx_a = _matrx_a
        self._matrx_u = _matrx_u
        self._first_a = _first_a
        self._first_u = _first_u
        self._sec___a = _sec___a
        self._sec___u = _sec___u

    def saveNPZ(self, path: Union[str, Path]):
        import numpy as np
        if (not isinstance(self._matrx_a, np.ndarray) or 
            not isinstance(self._first_a, np.ndarray) or 
            not isinstance(self._sec___a, np.ndarray)): 
            raise RuntimeError("Arrays must be numpy ndarrays before saving.")
        
        np.savez_compressed(path,
            _matrx_a = self._matrx_a,
            _first_a = self._first_a,
            _sec___a = self._sec___a,
            _matrx_u = str(self._matrx_u),
            _first_u = str(self._first_u),
            _sec___u = str(self._sec___u),
        )

    @classmethod
    def loadNPZ(cls, path: Union[str, Path]):
        import numpy as np
        with np.load(path) as data:
            return cls(
                _matrx_a = data["_matrx_a"],
                _matrx_u = ureg.Unit(str(data["_matrx_u"])),
                _first_a = data["_first_a"],
                _first_u = ureg.Unit(str(data["_first_u"])),
                _sec___a = data["_sec___a"],
                _sec___u = ureg.Unit(str(data["_sec___u"]))
            )

class SpecFunc(Function2D):
    _first_u: PintUnit = ureg.Unit("s")
    _sec___u: PintUnit = ureg.Unit("Hz")
    
    def __init__(self, matrix: Shaped[Array, "T H"], time: Shaped[Array, "T"], freq: Shaped[Array, "H"], unit: PintUnit):
        self._matrx_a = matrix
        self._first_a = time
        self._sec___a = freq
        self._matrx_u = unit

    @classmethod
    def from_Function2D(cls, func: Function2D):
        return cls(func._matrx_a, func._first_a, func._sec___a, func._matrx_u)

    @classmethod
    def from_zeros(cls, unit: PintUnit, dtype: Any, shape_t: int = 2, shape_h: int = 2):
        matrix = fw().zeros((shape_t, shape_h), dtype=dtype)
        time = fw().arange(shape_t)
        freq = fw().arange(shape_h)
        return cls(matrix=matrix, time=time, freq=freq, unit=unit)

    @property
    def time(self) -> Tuple[PintUnit, Shaped[Array, "T"]]:
        return self._first_u, self._first_a
    
    @time.setter
    def time(self, value: Shaped[Array, "T"]): 
        self._first_a = value
    
    @property
    def freq(self) -> Tuple[PintUnit, Shaped[Array, "H"]]:
        return self._sec___u, self._sec___a
    
    @freq.setter
    def freq(self, value: Shaped[Array, "H"]):
        self._sec___a = value

    @property
    def values(self) -> Tuple[PintUnit, Shaped[Array, "T H"]]:
        return self._matrx_u, self._matrx_a
    
    @values.setter
    def values(self, value: Shaped[Array, "T H"]):
        self._matrx_a = value

    def saveNPZ(self, path: Union[str, Path]):
        Function2D.saveNPZ(self, path)
        
    @classmethod
    def loadNPZ(cls, path: Union[str, Path]):
        return cls.from_Function2D(Function2D.loadNPZ(path))
