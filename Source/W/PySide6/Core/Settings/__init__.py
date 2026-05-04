import json
from enum import Enum
from typing import Any, Dict, List, Protocol, Type, overload, TypeVar, Generic, Tuple
from PySide6 import QtCore, QtGui # type: ignore

# ==========================================
# 1. АДАПТЕРЫ (СЕРИАЛИЗАТОРЫ)
# ==========================================
T = TypeVar('T')

class BaseAdapter(Generic[T]):
    def dump(self, value: T) -> Any: ...
    def load(self, raw_value: Any) -> T: ...

class EnumAdapter(BaseAdapter[Enum]):
    def __init__(self, enum_cls: Type[Enum], fallback: Enum):
        self.enum_cls = enum_cls
        self.fallback = fallback

    def dump(self, value: Enum) -> str:
        return value.name

    def load(self, raw_value: Any) -> Enum:
        try: return self.enum_cls[str(raw_value)]
        except (KeyError, ValueError): return self.fallback

class JsonAdapter(BaseAdapter[Dict[Any, Any] | List[Any]]):
    def dump(self, value: Dict[Any, Any] | List[Any]) -> str:
        return json.dumps(value, ensure_ascii=False)

    def load(self, raw_value: Any) -> Dict[Any, Any] | List[Any]:
        if not isinstance(raw_value, str): return {}
        try: return json.loads(raw_value)
        except json.JSONDecodeError: return {}

class LocalesAdapter(BaseAdapter[list[QtCore.QLocale]]):
    def dump(self, value: list[QtCore.QLocale]) -> list[str]:
        return [loc.name() for loc in value]

    def load(self, raw_value: Any) -> list[QtCore.QLocale]:
        if not raw_value: return []
        if isinstance(raw_value, str): raw_value = [raw_value]
        return [QtCore.QLocale(s) for s in raw_value]

class QtObjAdapter(BaseAdapter[T]):
    def __init__(self, obj_type: Type[T], fallback: T):
        self.obj_type = obj_type
        self.fallback = fallback

    def dump(self, value: T) -> Any: return value

    def load(self, raw_value: Any) -> T:
        if isinstance(raw_value, self.obj_type): return raw_value
        return self.fallback
    
class QColorAdapter(BaseAdapter[QtGui.QColor]):
    def __init__(self, fallback: QtGui.QColor):
        self.fallback = fallback

    def dump(self, value: QtGui.QColor) -> str: return value.name()

    def load(self, raw_value: Any) -> QtGui.QColor:
        if isinstance(raw_value, str) and QtGui.QColor.isValidColor(raw_value):
            return QtGui.QColor(raw_value)
        return self.fallback

# ==========================================
# 2. УНИВЕРСАЛЬНЫЙ ДЕСКРИПТОР
# ==========================================
class Settings(Protocol):
    @property
    def settings(self) -> QtCore.QSettings | None: ...

class SettingField[T]:
    def __init__(self, default: T, val_type: Type[T] | None = None, key: str | None = None, signal_ref: Any = None, adapter: BaseAdapter[T] | None = None) -> None:
        self.default = default
        self.val_type = val_type
        self.key = key
        self.adapter = adapter
        self._signal_ref = signal_ref
        self._signal_name: str | None = None

    def __set_name__(self, owner: type[Any], name: str) -> None:
        if self.key is None: self.key = name
        if self._signal_ref is not None:
            for attr_name, attr_value in vars(owner).items():
                if attr_value is self._signal_ref:
                    self._signal_name = attr_name
                    break

    def _get_settings(self, instance: Settings) -> QtCore.QSettings:
        st = getattr(instance, 'settings', None)
        return st if isinstance(st, QtCore.QSettings) else QtCore.QSettings()

    @overload
    def __get__(self, instance: None, owner: Type[Any]) -> 'SettingField[T]': ...
    @overload
    def __get__(self, instance: Settings, owner: Type[Any]) -> T: ...
    def __get__(self, instance: Settings | None, owner: Type[Any]) -> T | 'SettingField[T]':
        if instance is None: return self
        if self.key is None: raise RuntimeError("__set_name__ not colled")
        
        settings = self._get_settings(instance)
        raw_val = settings.value(self.key, None)

        if raw_val is None: return self.default
        if self.adapter: return self.adapter.load(raw_val)
        if self.val_type is not None: return settings.value(self.key, self.default, type=self.val_type) # type: ignore
        return raw_val

    def __set__(self, instance: Settings, value: T) -> None:
        current_value = self.__get__(instance, type(instance))
        if current_value == value: return

        settings = self._get_settings(instance)
        save_val = self.adapter.dump(value) if self.adapter else value
        settings.setValue(self.key, save_val) # type: ignore

        if self._signal_name and hasattr(instance, self._signal_name):
            getattr(instance, self._signal_name).emit(value)

# ==========================================
# 3. ФАБРИКА
# ==========================================
def Field[T](default: T, val_type: Type[T] | None = None, key: str | None = None, adapter: BaseAdapter[T] | None = None, sig_type: Type[Any] | None = None) -> Tuple[SettingField[T], Any]:
    if sig_type is None:
        sig_type = val_type if val_type is not None else type(default)
        if sig_type is list or sig_type is dict: sig_type = object

    sig = QtCore.Signal(sig_type)
    desc = SettingField[T](default=default, val_type=val_type, key=key, signal_ref=sig, adapter=adapter)
    return desc, sig
