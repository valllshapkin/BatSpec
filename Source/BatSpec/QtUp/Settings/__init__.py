import sys
from typing import List, Protocol, Type
from PySide6 import QtCore
from PySide6.QtWidgets import QApplication

class Settings(Protocol):
    settings: QtCore.QSettings | None

class SettingField[T]:
    """
    Дескриптор для удобной работы с QSettings.
    """
    def __init__(self, default:T | None = None, type: Type[T] | None = None, key: str | None = None):
        self.default = default
        self.type = type
        self.key: str = key # type: ignore
        self.name = None

    def __set_name__(self, owner: type, name: str):
        self.name = name
        if self.key is None: # type: ignore
            self.key = name

    def _get_settings(self, instance: Settings):
        if hasattr(instance, 'settings') and isinstance(instance.settings, QtCore.QSettings):
            return instance.settings
        return QtCore.QSettings()

    def __get__(self, instance: Settings | None, owner: Type[Settings]) -> T:
        if instance is None:
            return self
        
        settings = self._get_settings(instance)

        if self.type is not None:
            return settings.value(self.key, self.default, type=self.type)
        return settings.value(self.key, self.default)

    def __set__(self, instance: Settings, value: T):
        settings = self._get_settings(instance)
        settings.setValue(self.key, value)


class LocalesField:
    """
    Дескриптор для List[QLocale] — хранит как список строк вида 'en_US'.
    """
    def __init__(self, default: List[QtCore.QLocale] | None = None, key: str | None = None):
        self.default = default or []
        self.key: str = key  # type: ignore

    def __set_name__(self, owner: type, name: str):
        if self.key is None:
            self.key = name

    def _get_settings(self, instance) -> QtCore.QSettings:
        if hasattr(instance, 'settings') and isinstance(instance.settings, QtCore.QSettings):
            return instance.settings
        return QtCore.QSettings()

    def __get__(self, instance, owner) -> List[QtCore.QLocale]:
        if instance is None:
            return self

        settings = self._get_settings(instance)
        raw = settings.value(self.key, None)

        if raw is None:
            return self.default

        # QSettings может вернуть строку, если элемент один (ini-quirk)
        if isinstance(raw, str):
            raw = [raw]

        return [QtCore.QLocale(s) for s in raw]

    def __set__(self, instance, value: List[QtCore.QLocale]):
        settings = self._get_settings(instance)
        # Сохраняем как список строк: 'en_US', 'ru_RU', ...
        settings.setValue(self.key, [loc.name() for loc in value])
