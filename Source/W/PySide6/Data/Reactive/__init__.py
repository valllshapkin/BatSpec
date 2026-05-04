from collections.abc import Hashable, Iterable
from typing import Any
from PySide6.QtCore import QObject, Signal, Property

class ReactiveDict[K: Hashable, V](dict[K, V]):
    class Signals(QObject):
        itemSet = Signal(object, object, object)
        itemRemoved = Signal(object, object)
        cleared = Signal()
        updated = Signal(list)
        changed = Signal()

        def __init__(self, parent_dict: 'ReactiveDict[K, V]', parent: QObject | None = None):
            super().__init__(parent)
            self._dict = parent_dict

        @Property("QVariantMap", notify=changed)
        def data(self) -> dict[K, V]: return dict(self._dict)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.signals = self.Signals(self)
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: K, value: V) -> None:
        is_existing = key in self
        old_value = self.get(key)
        if is_existing and old_value == value: return
        super().__setitem__(key, value)
        self.signals.itemSet.emit(key, value, old_value)
        self.signals.changed.emit()

    def __delitem__(self, key: K) -> None:
        old_value = super().pop(key) 
        self.signals.itemRemoved.emit(key, old_value)
        self.signals.changed.emit()

    def pop(self, key: K, *args: Any) -> Any:
        try:
            value = super().pop(key)
            self.signals.itemRemoved.emit(key, value)
            self.signals.changed.emit()
            return value
        except KeyError:
            if args: return args[0]
            raise 

    def clear(self) -> None:
        if not self: return
        super().clear()
        self.signals.cleared.emit()
        self.signals.changed.emit()

    def update(self, *args: Any, **kwargs: Any) -> None:
        new_data = dict(*args, **kwargs)
        if not new_data: return
        changed_keys: list[K] = []
        for k, v in new_data.items():
            if k not in self or self.get(k) != v:
                old_val = self.get(k)
                super().__setitem__(k, v)
                changed_keys.append(k)
                self.signals.itemSet.emit(k, v, old_val)
        if changed_keys:
            self.signals.updated.emit(changed_keys)
            self.signals.changed.emit()

def make_reactive_dict[K, V](d: dict[K, V]) -> ReactiveDict[K, V]:
    return d if type(d) is ReactiveDict else ReactiveDict(d)

class ReactiveList[T](list[T]):
    class Signals(QObject):
        itemAdded = Signal(int, object)
        itemRemoved = Signal(int, object)
        itemSet = Signal(int, object, object)
        cleared = Signal()
        updated = Signal()
        changed = Signal()

        def __init__(self, parent_list: 'ReactiveList[T]', parent: QObject | None = None):
            super().__init__(parent)
            self._list = parent_list

        @Property(list, notify=changed)
        def data(self) -> list[T]: return list(self._list)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.signals = self.Signals(self)
        super().__init__(*args, **kwargs)

    def append(self, item: T) -> None:
        super().append(item)
        self.signals.itemAdded.emit(len(self) - 1, item)
        self.signals.changed.emit()

    def remove(self, item: T) -> None:
        idx = self.index(item)
        super().remove(item)
        self.signals.itemRemoved.emit(idx, item)
        self.signals.changed.emit()

    def clear(self) -> None:
        if not self: return
        super().clear()
        self.signals.cleared.emit()
        self.signals.changed.emit()

    def extend(self, iterable: Iterable[T]) -> None:
        items = list(iterable)
        if not items: return
        super().extend(items)
        self.signals.updated.emit()
        self.signals.changed.emit()

def make_reactive_list[T](l: list[T]) -> ReactiveList[T]:
    return l if type(l) is ReactiveList else ReactiveList(l)
