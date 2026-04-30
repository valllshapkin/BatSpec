from pathlib import Path
from typing import Any, Iterator, Self, Callable, Union
from jaxtyping import Float, Shaped

# --- 1. Импорты специфичных для проекта библиотек ---
from BatSpec.Core.Physical.Units import PintUnit, Phisical, UREG, unit_mul
from BatSpec.Core.SaveIntegral import SaveIntegral

# Внедряем наш новый универсальный фреймворк
import MultiArray as ma
from MultiArray import ArrayContext

ArrayLike = Any 

# ==========================================
# БАЗОВЫЕ КЛАССЫ
# ==========================================

class Function:
    @property
    def _primary_tensor(self) -> ArrayLike:
        """Абстрактное свойство: возвращает главный тензор объекта для определения контекста."""
        raise NotImplementedError()

    @property
    def context(self) -> ArrayContext:
        """Единый контекст массива (фреймворк, устройство, тип данных)."""
        return ArrayContext.from_array(self._primary_tensor)


class Function1D(Function, SaveIntegral[Phisical[Shaped[ArrayLike, '...']]]):
    _value_a: Shaped[ArrayLike, '... x']
    _value_u: PintUnit
    _axis__a: Shaped[ArrayLike, 'x']
    _axis__u: PintUnit

    @property
    def _primary_tensor(self) -> ArrayLike:
        return self._value_a

    def __init__(self, values: Phisical[Shaped[ArrayLike, '... x']], axis: Phisical[Float[ArrayLike, "axis"]]):
        self._value_a, self._value_u = values
        self._axis__a, self._axis__u = axis
       
    @property
    def _d_axis__a(self) -> Phisical[float]:
        if len(self._axis__a) < 2:
            raise ValueError("Cannot calculate step size for an AnyArray with less than 2 elements.")
        step = float((self._axis__a[-1] - self._axis__a[0]) / (len(self._axis__a) - 1))
        return step, self._axis__u

    def _SaveIntegral_Energy(self) -> Phisical[Shaped[ArrayLike, '...']]:
        dA, dU = self._d_axis__a
        energy_tensor = ma.abs(self._value_a) ** 2
        integral = ma.sum(energy_tensor, axis=-1) * dA
        return integral, unit_mul(self._value_u, self._value_u, dU)

    def _SaveIntegral_Area(self) -> Phisical[Shaped[ArrayLike, '...']]:
        dA, dU = self._d_axis__a
        integral = ma.sum(self._value_a, axis=-1) * dA
        return integral, unit_mul(self._value_u, dU)

    def _FileSystem_Save(self, path: Path) -> None:
        raise NotImplementedError()
    
    @classmethod
    def _FileSystem_Load(cls, path: Path) -> Self:
        raise NotImplementedError()

    def __len__(self) -> int:
        if self._value_a.ndim <= 1:
            raise TypeError("Нет батчевого измерения.")
        return self._value_a.shape[0]

    def __getitem__(self, idx: Union[int, slice]) -> 'Function1D':
        if self._value_a.ndim <= 1:
            raise TypeError("Нет батчевого измерения.")
        return Function1D(
            values=(self._value_a[idx], self._value_u),
            axis=(self._axis__a, self._axis__u)
        )

    def __iter__(self) -> Iterator['Function1D']:
        for i in range(len(self)):
            yield self[i]

    def to_context(self, ctx: ArrayContext) -> 'Function1D':
        """Новый стандартный метод миграции данных."""
        new_value_a = ma.convert_to(self._value_a, ctx)
        
        # ИСПРАВЛЕНИЕ АРХИТЕКТУРЫ: Оси не должны перенимать dtype матрицы (особенно если матрица комплексная).
        # Передавая None в dtype, мы заставляем фреймворк сохранить естественный (вещественный) тип оси.
        axis_ctx = ArrayContext(ctx._framework, ctx._device, None)
        new_axis__a = ma.convert_to(self._axis__a, axis_ctx)
        
        return Function1D(
            values=(new_value_a, self._value_u), 
            axis=(new_axis__a, self._axis__u)
        )

    # ==========================================
    # Математические операторы (Math Overloads)
    # ==========================================

    def _math_op(self, other: Any, op: Callable, unit_op: Callable = None) -> Self:
        if isinstance(other, Function1D):
            new_val = op(self._value_a, other._value_a)
            new_u = unit_op(self._value_u, other._value_u) if unit_op else self._value_u
        else:
            new_val = op(self._value_a, other)
            new_u = self._value_u
            
        base = Function1D(
            values=(new_val, new_u), 
            axis=(self._axis__a, self._axis__u)
        )
        if type(self) is Function1D:
            return base
        return type(self).from_Function1D(base)

    def __add__(self, other): return self._math_op(other, lambda a, b: a + b)
    def __radd__(self, other): return self._math_op(other, lambda a, b: b + a)
    
    def __sub__(self, other): return self._math_op(other, lambda a, b: a - b)
    def __rsub__(self, other): return self._math_op(other, lambda a, b: b - a)
    
    def __mul__(self, other): return self._math_op(other, lambda a, b: a * b, lambda u1, u2: u1 * u2)
    def __rmul__(self, other): return self._math_op(other, lambda a, b: b * a, lambda u1, u2: u2 * u1)
    
    def __truediv__(self, other): return self._math_op(other, lambda a, b: a / b, lambda u1, u2: u1 / u2)
    def __rtruediv__(self, other): return self._math_op(other, lambda a, b: b / a, lambda u1, u2: u2 / u1)


class TimeFunc(Function1D):
    _axis__u: PintUnit = UREG.second

    def __init__(self, values: Phisical[Shaped[ArrayLike, '... time']], axis: Float[ArrayLike, 'time']):
        if values[0].shape[-1] != axis.shape[-1]: 
            raise ValueError(f"Последнее измерение values ({values[0].shape[-1]}) и длина axis ({axis.shape[-1]}) должны совпадать")
        self._value_a, self._value_u = values
        self._axis__a = axis

    @classmethod
    def from_Function1D(cls, func: Function1D) -> Self:
        return cls((func._value_a, func._value_u), func._axis__a)
    
    @property
    def time(self) -> Phisical[Float[ArrayLike, 'time']]:
        return self._axis__a, self._axis__u
    
    @time.setter
    def time(self, value: Float[ArrayLike, 'time']):
        if len(value) != len(self._axis__a):
            raise ValueError(f"New time AnyArray must have length {len(self._axis__a)}, got {len(value)}")
        self._axis__a = value

    @property
    def start(self) -> float:
        return float(self._axis__a[0])
    
    @property
    def end(self) -> float:
        return float(self._axis__a[-1])
    
    @property
    def values(self) -> Phisical[Shaped[ArrayLike, '... time']]:
        return self._value_a, self._value_u

    @values.setter
    def values(self, value: Phisical[Shaped[ArrayLike, '... time']]):
        if value[0].shape[-1] != self._axis__a.shape[-1]:
            raise ValueError(f"Новый массив значений должен иметь размер последнего измерения {self._axis__a.shape[-1]}, получено {value[0].shape[-1]}")
        self._value_a, self._value_u = value
        
    @property
    def dt(self) -> Phisical[float]:
        return self._d_axis__a

    @property
    def sr(self) -> int:
        dT, _ = self.dt
        return round(1.0 / dT)

    def _FileSystem_Save(self, path: Path):
        Function1D._FileSystem_Save(self, path)
        
    @classmethod
    def _FileSystem_Load(cls, path: Path):
        return cls.from_Function1D(Function1D._FileSystem_Load(path))

    def __getitem__(self, idx: Union[int, slice]) -> 'TimeFunc':
        return TimeFunc.from_Function1D(Function1D.__getitem__(self, idx))
    
    def __iter__(self) -> Iterator['TimeFunc']:
        for i in range(len(self)):
            yield self[i]

    def to_context(self, ctx: ArrayContext) -> 'TimeFunc':
        return TimeFunc.from_Function1D(Function1D.to_context(self, ctx))


class FreqFunc(Function1D):
    _axis__u: PintUnit = UREG.hertz

    def __init__(self, values: Phisical[Shaped[ArrayLike, '... freq']], axis: Float[ArrayLike, 'freq']):
        if values[0].shape[-1] != axis.shape[-1]: 
            raise ValueError(f"Последнее измерение values ({values[0].shape[-1]}) и длина axis ({axis.shape[-1]}) должны совпадать")
        self._value_a, self._value_u = values
        self._axis__a = axis

    @classmethod
    def from_Function1D(cls, func: Function1D) -> Self:
        return cls((func._value_a, func._value_u), func._axis__a)
    
    @property
    def freq(self) -> Phisical[Float[ArrayLike, 'freq']]:
        return self._axis__a, self._axis__u

    @freq.setter
    def freq(self, value: Float[ArrayLike, 'freq']):
        if len(value) != len(self._axis__a):
            raise ValueError(f"Новый массив частот должен иметь длину {len(self._axis__a)}, получено {len(value)}")
        self._axis__a = value

    @property
    def min_freq(self) -> float:
        return float(self._axis__a[0])
    
    @property
    def max_freq(self) -> float:
        return float(self._axis__a[-1])
    
    @property
    def values(self) -> Phisical[Shaped[ArrayLike, '... freq']]:
        return self._value_a, self._value_u

    @values.setter
    def values(self, value: Phisical[Shaped[ArrayLike, '... freq']]):
        if value[0].shape[-1] != self._axis__a.shape[-1]:
            raise ValueError(f"Новый массив значений должен иметь размер последнего измерения {self._axis__a.shape[-1]}, получено {value[0].shape[-1]}")
        self._value_a, self._value_u = value
        
    @property
    def df(self) -> Phisical[float]:
        return self._d_axis__a

    def _FileSystem_Save(self, path: Path):
        Function1D._FileSystem_Save(self, path)
        
    @classmethod
    def _FileSystem_Load(cls, path: Path) -> Self:
        return cls.from_Function1D(Function1D._FileSystem_Load(path))
    
    def __getitem__(self, idx: Union[int, slice]) -> 'FreqFunc':
        return FreqFunc.from_Function1D(Function1D.__getitem__(self, idx))
    
    def __iter__(self) -> Iterator['FreqFunc']:
        for i in range(len(self)):
            yield self[i]
    
    def to_context(self, ctx: ArrayContext) -> 'FreqFunc':
        return FreqFunc.from_Function1D(Function1D.to_context(self, ctx))


class Function2D(Function, SaveIntegral[Phisical[Shaped[ArrayLike, '...']]]):
    _matrx_a: Shaped[ArrayLike, '... x y']
    _matrx_u: PintUnit
    _first_a: Shaped[ArrayLike, 'x']
    _first_u: PintUnit
    _sec___a: Shaped[ArrayLike, 'y']
    _sec___u: PintUnit

    @property
    def _primary_tensor(self) -> ArrayLike:
        return self._matrx_a

    def __init__(self, 
                 matrix: Phisical[Shaped[ArrayLike, '... x y']], 
                 first: Phisical[Float[ArrayLike, "x"]], 
                 sec: Phisical[Float[ArrayLike, "y"]]):
        self._matrx_a, self._matrx_u = matrix
        self._first_a, self._first_u = first
        self._sec___a, self._sec___u = sec

    @property
    def _d_first_a(self) -> Phisical[float]:
        if len(self._first_a) < 2:
            raise ValueError("Cannot calculate step size for an array with less than 2 elements.")
        step = float((self._first_a[-1] - self._first_a[0]) / (len(self._first_a) - 1))
        return step, self._first_u

    @property
    def _d_sec___a(self) -> Phisical[float]:
        if len(self._sec___a) < 2:
            raise ValueError("Cannot calculate step size for an array with less than 2 elements.")
        step = float((self._sec___a[-1] - self._sec___a[0]) / (len(self._sec___a) - 1))
        return step, self._sec___u

    def _SaveIntegral_Area(self) -> Phisical[Shaped[ArrayLike, '...']]:
        f_da, f_du = self._d_first_a
        s_da, s_du = self._d_sec___a
        integral = ma.sum(self._matrx_a, axis=(-2, -1)) * f_da * s_da
        return integral, unit_mul(self._matrx_u, f_du, s_du)
    
    def _SaveIntegral_Energy(self) -> Phisical[Shaped[ArrayLike, '...']]:
        f_da, f_du = self._d_first_a
        s_da, s_du = self._d_sec___a
        energy_tensor = ma.abs(self._matrx_a) ** 2
        integral = ma.sum(energy_tensor, axis=(-2, -1)) * f_da * s_da
        return integral, unit_mul(self._matrx_u, self._matrx_u, f_du, s_du)

    def _FileSystem_Save(self, path: Path) -> None:
        raise NotImplementedError()
    
    @classmethod
    def _FileSystem_Load(cls, path: Path) -> Self:
        raise NotImplementedError()

    def __len__(self) -> int:
        if self._matrx_a.ndim <= 2:
            raise TypeError("Нет батчевого измерения.")
        return self._matrx_a.shape[0]

    def __getitem__(self, idx: Union[int, slice]) -> 'Function2D':
        if self._matrx_a.ndim <= 2:
            raise TypeError("Нет батчевого измерения.")
        return Function2D(
            matrix=(self._matrx_a[idx], self._matrx_u),
            first=(self._first_a, self._first_u),
            sec=(self._sec___a, self._sec___u)
        )

    def __iter__(self) -> Iterator['Function2D']:
        for i in range(len(self)):
            yield self[i]

    def to_context(self, ctx: ArrayContext) -> 'Function2D':
        new_matrx = ma.convert_to(self._matrx_a, ctx)
        
        # ИСПРАВЛЕНИЕ АРХИТЕКТУРЫ: Изолируем оси от комплексного типа матрицы
        axis_ctx = ArrayContext(ctx._framework, ctx._device, None)
        new_first = ma.convert_to(self._first_a, axis_ctx)
        new_sec   = ma.convert_to(self._sec___a, axis_ctx)

        return Function2D(
            matrix=(new_matrx, self._matrx_u),
            first=(new_first, self._first_u),
            sec=(new_sec, self._sec___u)
        )

    # ==========================================
    # Математические операторы (Math Overloads)
    # ==========================================

    def _math_op(self, other: Any, op: Callable, unit_op: Callable = None) -> Self:
        if isinstance(other, Function2D):
            new_mat = op(self._matrx_a, other._matrx_a)
            new_u = unit_op(self._matrx_u, other._matrx_u) if unit_op else self._matrx_u
        else:
            new_mat = op(self._matrx_a, other)
            new_u = self._matrx_u
            
        base = Function2D(
            matrix=(new_mat, new_u), 
            first=(self._first_a, self._first_u), 
            sec=(self._sec___a, self._sec___u)
        )
        if type(self) is Function2D:
            return base
        return type(self).from_Function2D(base)

    def __add__(self, other): return self._math_op(other, lambda a, b: a + b)
    def __radd__(self, other): return self._math_op(other, lambda a, b: b + a)
    
    def __sub__(self, other): return self._math_op(other, lambda a, b: a - b)
    def __rsub__(self, other): return self._math_op(other, lambda a, b: b - a)
    
    def __mul__(self, other): return self._math_op(other, lambda a, b: a * b, lambda u1, u2: u1 * u2)
    def __rmul__(self, other): return self._math_op(other, lambda a, b: b * a, lambda u1, u2: u2 * u1)
    
    def __truediv__(self, other): return self._math_op(other, lambda a, b: a / b, lambda u1, u2: u1 / u2)
    def __rtruediv__(self, other): return self._math_op(other, lambda a, b: b / a, lambda u1, u2: u2 / u1)


class SpecFunc(Function2D):
    _first_u: PintUnit = UREG.hertz
    _sec___u: PintUnit = UREG.second

    def __init__(self, 
                 matrix: Phisical[Shaped[ArrayLike, '... freq time']], 
                 freq: Float[ArrayLike, 'freq'],
                 time: Float[ArrayLike, 'time']):
        if matrix[0].shape[-2] != freq.shape[-1]:
            raise ValueError(f"Размер матрицы по оси частот ({matrix[0].shape[-2]}) не совпадает с длиной оси freq ({freq.shape[-1]})")
        if matrix[0].shape[-1] != time.shape[-1]:
            raise ValueError(f"Размер матрицы по оси времени ({matrix[0].shape[-1]}) не совпадает с длиной оси time ({time.shape[-1]})")
        
        self._matrx_a, self._matrx_u = matrix
        self._first_a = freq  
        self._sec___a = time  

    @classmethod
    def from_Function2D(cls, func: Function2D) -> Self:
        return cls(
            matrix=(func._matrx_a, func._matrx_u), 
            freq=func._first_a, 
            time=func._sec___a
        )

    @property
    def freq(self) -> Phisical[Float[ArrayLike, 'freq']]:
        return self._first_a, self._first_u
    
    @freq.setter
    def freq(self, value: Float[ArrayLike, 'freq']):
        if value.shape[-1] != self._first_a.shape[-1]:
            raise ValueError(f"Новая ось частот должна иметь длину {self._first_a.shape[-1]}, получено {value.shape[-1]}")
        self._first_a = value

    @property
    def time(self) -> Phisical[Float[ArrayLike, 'time']]:
        return self._sec___a, self._sec___u
    
    @time.setter
    def time(self, value: Float[ArrayLike, 'time']): 
        if value.shape[-1] != self._sec___a.shape[-1]:
            raise ValueError(f"Новая ось времени должна иметь длину {self._sec___a.shape[-1]}, получено {value.shape[-1]}")
        self._sec___a = value

    @property
    def values(self) -> Phisical[Shaped[ArrayLike, '... freq time']]:
        return self._matrx_a, self._matrx_u
    
    @values.setter
    def values(self, value: Phisical[Shaped[ArrayLike, '... freq time']]):
        if value[0].shape[-2] != self._first_a.shape[-1]:
            raise ValueError(f"Размер матрицы по частоте (-2) должен быть {self._first_a.shape[-1]}")
        if value[0].shape[-1] != self._sec___a.shape[-1]:
            raise ValueError(f"Размер матрицы по времени (-1) должен быть {self._sec___a.shape[-1]}")
        self._matrx_a, self._matrx_u = value

    @property
    def df(self) -> Phisical[float]:
        return self._d_first_a

    @property
    def dt(self) -> Phisical[float]:
        return self._d_sec___a

    def _FileSystem_Save(self, path: Path):
        Function2D._FileSystem_Save(self, path)
        
    @classmethod
    def _FileSystem_Load(cls, path: Path) -> Self:
        return cls.from_Function2D(Function2D._FileSystem_Load(path))

    def integrateOverTime(self) -> FreqFunc:
        dt_value, dt_unit = self.dt
        integrated_values = ma.sum(self._matrx_a, axis=-1) * dt_value
        new_unit = unit_mul(self._matrx_u, dt_unit)
        return FreqFunc(
            values=(integrated_values, new_unit),
            axis=self._first_a
        )

    def integrateOverFreq(self) -> TimeFunc:
        df_value, df_unit = self.df
        integrated_values = ma.sum(self._matrx_a, axis=-2) * df_value
        new_unit = unit_mul(self._matrx_u, df_unit)
        return TimeFunc(
            values=(integrated_values, new_unit),
            axis=self._sec___a
        )
    
    def cloneApply(self, func: Callable[[ArrayLike], ArrayLike], new_unit: PintUnit) -> 'SpecFunc':
        return SpecFunc(
            matrix=(func(self._matrx_a), new_unit), 
            freq=self._first_a,
            time=self._sec___a
        )

    def __getitem__(self, idx: Union[int, slice]) -> 'SpecFunc':
        return SpecFunc.from_Function2D(Function2D.__getitem__(self, idx))
    
    def __iter__(self) -> Iterator['SpecFunc']:
        for i in range(len(self)):
            yield self[i]

    def to_context(self, ctx: ArrayContext) -> 'SpecFunc':
        return SpecFunc.from_Function2D(Function2D.to_context(self, ctx))


class CorrelFunc(Function2D):
    _first_u: PintUnit = UREG.second  # Задержка (Delay)
    _sec___u: PintUnit = UREG.second  # Реальное время (Time)

    def __init__(self, 
                 matrix: Phisical[Shaped[ArrayLike, '... delay time']], 
                 delay: Float[ArrayLike, 'delay'],
                 time: Float[ArrayLike, 'time']):
        if matrix[0].shape[-2] != delay.shape[-1]:
            raise ValueError(f"Размер матрицы по оси задержки ({matrix[0].shape[-2]}) не совпадает с длиной оси delay ({delay.shape[-1]})")
        if matrix[0].shape[-1] != time.shape[-1]:
            raise ValueError(f"Размер матрицы по оси времени ({matrix[0].shape[-1]}) не совпадает с длиной оси time ({time.shape[-1]})")
        
        self._matrx_a, self._matrx_u = matrix
        self._first_a = delay  
        self._sec___a = time  

    @classmethod
    def from_Function2D(cls, func: Function2D) -> Self:
        return cls(
            matrix=(func._matrx_a, func._matrx_u), 
            delay=func._first_a, 
            time=func._sec___a
        )

    @property
    def delay(self) -> Phisical[Float[ArrayLike, 'delay']]:
        return self._first_a, self._first_u
    
    @delay.setter
    def delay(self, value: Float[ArrayLike, 'delay']):
        if value.shape[-1] != self._first_a.shape[-1]:
            raise ValueError(f"Новая ось задержки должна иметь длину {self._first_a.shape[-1]}, получено {value.shape[-1]}")
        self._first_a = value

    @property
    def time(self) -> Phisical[Float[ArrayLike, 'time']]:
        return self._sec___a, self._sec___u
    
    @time.setter
    def time(self, value: Float[ArrayLike, 'time']): 
        if value.shape[-1] != self._sec___a.shape[-1]:
            raise ValueError(f"Новая ось времени должна иметь длину {self._sec___a.shape[-1]}, получено {value.shape[-1]}")
        self._sec___a = value

    @property
    def values(self) -> Phisical[Shaped[ArrayLike, '... delay time']]:
        return self._matrx_a, self._matrx_u
    
    @values.setter
    def values(self, value: Phisical[Shaped[ArrayLike, '... delay time']]):
        if value[0].shape[-2] != self._first_a.shape[-1]:
            raise ValueError(f"Размер матрицы по задержке (-2) должен быть {self._first_a.shape[-1]}")
        if value[0].shape[-1] != self._sec___a.shape[-1]:
            raise ValueError(f"Размер матрицы по времени (-1) должен быть {self._sec___a.shape[-1]}")
        self._matrx_a, self._matrx_u = value

    def __getitem__(self, idx: Union[int, slice]) -> 'CorrelFunc':
        return CorrelFunc.from_Function2D(Function2D.__getitem__(self, idx))
    
    def __iter__(self) -> Iterator['CorrelFunc']:
        for i in range(len(self)):
            yield self[i]

    def to_context(self, ctx: ArrayContext) -> 'CorrelFunc':
        return CorrelFunc.from_Function2D(Function2D.to_context(self, ctx))