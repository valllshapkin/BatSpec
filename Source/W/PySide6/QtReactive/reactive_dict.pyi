from collections.abc import Callable, Hashable
from typing import Any
from PySide6.QtCore import QObject, Signal

class ReactiveDict[K: Hashable, V](dict[K, V]):
    class Signals(QObject):
        itemSet: Signal
        itemRemoved: Signal
        cleared: Signal
        updated: Signal
        changed: Signal
        
        def __init__(self, parent_dict: 'ReactiveDict[K, V]', parent: QObject | None = ...) -> None: ...
        
        @property
        def data(self) -> dict[K, V]: ...

    signals: Signals

    # Мы явно указываем только то, что добавляем или меняем поведение
    def connect_all(self, slot: Callable[..., Any]) -> None: ...
    def copy(self) -> 'ReactiveDict[K, V]': ...

def make_reactive[K, V](d: dict[K, V]) -> ReactiveDict[K, V]: ...