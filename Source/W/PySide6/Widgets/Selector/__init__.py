from typing import Any
from PySide6 import QtCore, QtWidgets
from W.PySide6.Data.Reactive import ReactiveDict

class Selector(QtWidgets.QGroupBox):
    class Signals(QtCore.QObject):
        keySelected = QtCore.Signal(str)
        activeValueChanged = QtCore.Signal(object)

    def __init__(self, title: str = "Select:", parent: QtWidgets.QWidget | None = None):
        super().__init__(title, parent)
        self.signals = self.Signals(self)
        self._rdict: ReactiveDict[str, Any] | None = None
        self._current_key: str | None = None

        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setContentsMargins(4, 8, 4, 4)

        self.label = QtWidgets.QLabel(self.tr("Active Key:"))
        self.main_layout.addWidget(self.label)

        self.combo = QtWidgets.QComboBox(self)
        self.combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.main_layout.addWidget(self.combo)

        self.combo.currentIndexChanged.connect(self._on_combo_index_changed)

    def set_dictionary(self, rdict: ReactiveDict[str, Any]) -> None:
        if self._rdict is not None:
            self._rdict.signals.itemSet.disconnect(self._on_dict_item_set)
            self._rdict.signals.itemRemoved.disconnect(self._on_dict_item_removed)
            self._rdict.signals.cleared.disconnect(self._on_dict_cleared)

        self._rdict = rdict
        self._rdict.signals.itemSet.connect(self._on_dict_item_set)
        self._rdict.signals.itemRemoved.connect(self._on_dict_item_removed)
        self._rdict.signals.cleared.connect(self._on_dict_cleared)
        self._rebuild_ui()

    def active_key(self) -> str | None: return self._current_key

    def active_value(self) -> Any:
        if self._rdict is not None and self._current_key is not None:
            return self._rdict.get(self._current_key)
        return None

    def _rebuild_ui(self) -> None:
        with QtCore.QSignalBlocker(self.combo):
            self.combo.clear()
            if self._rdict: self.combo.addItems(list(self._rdict.keys()))
        self._sync_current_selection()

    def _on_dict_item_set(self, key: str, new_value: Any, old_value: Any) -> None:
        idx = self.combo.findText(key)
        if idx == -1:
            with QtCore.QSignalBlocker(self.combo): self.combo.addItem(key)
            self._sync_current_selection()
        elif key == self._current_key:
            self.signals.activeValueChanged.emit(new_value)

    def _on_dict_item_removed(self, key: str, old_value: Any) -> None:
        idx = self.combo.findText(key)
        if idx >= 0:
            with QtCore.QSignalBlocker(self.combo): self.combo.removeItem(idx)
            self._sync_current_selection()

    def _on_dict_cleared(self) -> None:
        with QtCore.QSignalBlocker(self.combo): self.combo.clear()
        self._sync_current_selection()

    def _on_combo_index_changed(self, index: int) -> None:
        self._sync_current_selection()

    def _sync_current_selection(self) -> None:
        new_key = self.combo.currentText() if self.combo.count() > 0 else None
        if new_key == self._current_key: return
        self._current_key = new_key
        
        if self._current_key is not None:
            self.signals.keySelected.emit(self._current_key)
            self.signals.activeValueChanged.emit(self.active_value())
        else:
            self.signals.keySelected.emit("")
            self.signals.activeValueChanged.emit(None)
