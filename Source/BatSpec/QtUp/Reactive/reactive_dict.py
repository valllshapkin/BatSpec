from collections.abc import Callable, Hashable
from typing import Any
from PySide6.QtCore import QObject, Signal, Property

class ReactiveDict[K: Hashable, V](dict[K, V]):
    """
    Индустриальная реализация реактивного словаря.
    - Оптимизировано от холостых обновлений (No-op check).
    - Атомарные массовые операции (update, |=).
    - Exception-safe (если базовая операция падает, сигналы не летят).
    """

    class Signals(QObject):
        itemSet = Signal(object, object, object)      # key, new_value, old_value
        itemRemoved = Signal(object, object)          # key, old_value
        cleared = Signal()
        updated = Signal(list)                        # list of changed keys
        changed = Signal()                            # Universal trigger

        def __init__(self, parent_dict: 'ReactiveDict[K, V]', parent: QObject | None = None):
            super().__init__(parent)
            self._dict = parent_dict

        @Property("QVariantMap", notify=changed)
        def data(self) -> dict[K, V]:
            return dict(self._dict)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.signals = self.Signals(self)
        # Инициализируем через наш же update, чтобы при создании 
        # (если нужно) сразу отработали сигналы. Но обычно словарь 
        # создают пустым. Мы подавим initial update signals для оптимизации.
        super().__init__(*args, **kwargs)

    # ==================== Одиночные мутации ====================

    def __setitem__(self, key: K, value: V) -> None:
        is_existing = key in self
        old_value = self.get(key)
        
        # No-op optimization: игнорируем, если значение не изменилось
        if is_existing and old_value == value:
            return

        super().__setitem__(key, value)
        self.signals.itemSet.emit(key, value, old_value)
        self.signals.changed.emit()

    def __delitem__(self, key: K) -> None:
        # Вызовет KeyError штатно, если ключа нет. Сигналы не вылетят.
        old_value = super().pop(key) 
        self.signals.itemRemoved.emit(key, old_value)
        self.signals.changed.emit()

    def pop(self, key: K, *args: Any) -> Any:
        try:
            # Пытаемся удалить как обычно
            value = super().pop(key)
            self.signals.itemRemoved.emit(key, value)
            self.signals.changed.emit()
            return value
        except KeyError:
            # Если ключа нет, но передан default (через *args)
            if args:
                return args[0]
            raise  # Если default не передан, пробрасываем KeyError

    def popitem(self) -> tuple[K, V]:
        # Бросит KeyError если пуст (что нам и нужно)
        key, value = super().popitem()
        self.signals.itemRemoved.emit(key, value)
        self.signals.changed.emit()
        return key, value

    def setdefault(self, key: K, default: V | None = None) -> V:
        if key not in self:
            # Вызовет наш __setitem__, который испустит сигналы
            self[key] = default  # type: ignore
        return self[key]

    # ==================== Массовые мутации ====================

    def clear(self) -> None:
        if not self:  # No-op check
            return
            
        super().clear()
        self.signals.cleared.emit()
        self.signals.changed.emit()

    def update(self, *args: Any, **kwargs: Any) -> None:
        """
        Массовое обновление. Парсит любые аргументы через встроенный dict,
        выявляет реальную дельту и испускает атомарные сигналы.
        """
        new_data = dict(*args, **kwargs)
        if not new_data:
            return

        changed_keys: list[K] = []
        
        for k, v in new_data.items():
            is_existing = k in self
            old_value = self.get(k)
            
            if not is_existing or old_value != v:
                super().__setitem__(k, v)
                changed_keys.append(k)
                self.signals.itemSet.emit(k, v, old_value)

        # Вызываем changed() и updated() ТОЛЬКО если были реальные изменения
        if changed_keys:
            self.signals.updated.emit(changed_keys)
            self.signals.changed.emit()

    def __ior__(self, other: Any) -> 'ReactiveDict[K, V]':
        """Поддержка оператора in-place объединения `|=` (Python 3.9+)"""
        self.update(other)
        return self

    # ==================== Утилиты ====================

    def copy(self) -> 'ReactiveDict[K, V]':
        """copy.copy() должно возвращать новый независимый ReactiveDict"""
        return self.__class__(self)

    def connect_all(self, slot: Callable[..., Any]) -> None:
        """Утилита для логгирования или жесткого биндинга"""
        self.signals.itemSet.connect(slot)
        self.signals.itemRemoved.connect(slot)
        self.signals.cleared.connect(slot)
        self.signals.updated.connect(slot)
        self.signals.changed.connect(slot)


# ====================== Фабрика ======================

def make_reactive[K, V](d: dict[K, V]) -> ReactiveDict[K, V]:
    if type(d) is ReactiveDict:  # Строгая проверка типа вместо isinstance
        return d
    return ReactiveDict(d)