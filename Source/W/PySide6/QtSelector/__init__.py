from typing import Any

from PySide6 import QtCore, QtWidgets
from BatSpec.QtUp.Reactive.reactive_dict import ReactiveDict


class Selector(QtWidgets.QGroupBox):
    """
    Виджет-селектор для выбора активного ключа из ReactiveDict[str, Any].
    Автоматически обновляет QComboBox при добавлении/удалении ключей в словаре.
    """

    class Signals(QtCore.QObject):
        """Вложенный класс сигналов для сохранения архитектурного стиля."""
        # Испускается, когда пользователь (или система) меняет выбранный ключ
        keySelected = QtCore.Signal(str)
        # Испускается, если значение текущего выбранного ключа изменилось в самом словаре
        activeValueChanged = QtCore.Signal(object)

        def __init__(self, parent: QtCore.QObject | None = None):
            super().__init__(parent)


    def __init__(self, title: str = "Select:", parent: QtWidgets.QWidget | None = None):
        super().__init__(title, parent)
        
        # 1. Инициализация сигналов
        self.signals = self.Signals(self)
        
        # 2. Состояние
        self._rdict: ReactiveDict[str, Any] | None = None
        self._current_key: str | None = None

        # 3. Настройка UI
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setContentsMargins(4, 8, 4, 4)

        self.label = QtWidgets.QLabel(self.tr("Active Key:"))
        self.main_layout.addWidget(self.label)

        self.combo = QtWidgets.QComboBox(self)
        self.combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.main_layout.addWidget(self.combo)

        # 4. Подписка на действия пользователя в UI
        self.combo.currentIndexChanged.connect(self._on_combo_index_changed)

    # ==================== Публичный API ====================

    def set_dictionary(self, rdict: ReactiveDict[str, Any]) -> None:
        """Привязать новый реактивный словарь к виджету."""
        # Отписываемся от старого словаря, если был
        if self._rdict is not None:
            self._rdict.signals.itemSet.disconnect(self._on_dict_item_set)
            self._rdict.signals.itemRemoved.disconnect(self._on_dict_item_removed)
            self._rdict.signals.cleared.disconnect(self._on_dict_cleared)

        self._rdict = rdict

        # Подписываемся на новый (благодаря атомарности update(), 
        # нам достаточно слушать только гранулярные сигналы)
        self._rdict.signals.itemSet.connect(self._on_dict_item_set)
        self._rdict.signals.itemRemoved.connect(self._on_dict_item_removed)
        self._rdict.signals.cleared.connect(self._on_dict_cleared)

        self._rebuild_ui()

    def active_key(self) -> str | None:
        """Возвращает текущий выбранный ключ (или None, если словарь пуст)."""
        return self._current_key

    def active_value(self) -> Any:
        """Возвращает значение по текущему ключу (или None)."""
        if self._rdict is not None and self._current_key is not None:
            return self._rdict.get(self._current_key)
        return None

    def set_active_key(self, key: str) -> None:
        """Программно установить активный ключ."""
        idx = self.combo.findText(key)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)


    # ==================== Реакция на словарь (Data -> UI) ====================

    def _rebuild_ui(self) -> None:
        """Полная пересборка комбобокса (вызывается при смене словаря)."""
        # Блокируем сигналы UI, чтобы не спамить при добавлении элементов
        with QtCore.QSignalBlocker(self.combo):
            self.combo.clear()
            if self._rdict:
                self.combo.addItems(list(self._rdict.keys()))
        
        self._sync_current_selection()

    def _on_dict_item_set(self, key: str, new_value: Any, old_value: Any) -> None:
        """Обработчик добавления или изменения ключа."""
        idx = self.combo.findText(key)
        
        if idx == -1:
            # Ключ новый - добавляем в комбобокс
            with QtCore.QSignalBlocker(self.combo):
                self.combo.addItem(key)
            self._sync_current_selection()
        else:
            # Ключ уже есть (обновилось значение). 
            # Если это сейчас активный ключ, испускаем сигнал об изменении ЗНАЧЕНИЯ.
            if key == self._current_key:
                self.signals.activeValueChanged.emit(new_value)

    def _on_dict_item_removed(self, key: str, old_value: Any) -> None:
        """Обработчик удаления ключа."""
        idx = self.combo.findText(key)
        if idx >= 0:
            with QtCore.QSignalBlocker(self.combo):
                self.combo.removeItem(idx)
            self._sync_current_selection()

    def _on_dict_cleared(self) -> None:
        """Обработчик очистки словаря."""
        with QtCore.QSignalBlocker(self.combo):
            self.combo.clear()
        self._sync_current_selection()


    # ==================== Реакция на UI (UI -> Logic) ====================

    def _on_combo_index_changed(self, index: int) -> None:
        """Срабатывает, когда ПОЛЬЗОВАТЕЛЬ кликает в UI."""
        self._sync_current_selection()

    def _sync_current_selection(self) -> None:
        """
        Синхронизирует внутреннее состояние `_current_key` с UI 
        и испускает сигналы, если выбор реально изменился.
        """
        new_key = self.combo.currentText() if self.combo.count() > 0 else None
        
        # Если выбор остался прежним, ничего не делаем (No-op защита)
        if new_key == self._current_key:
            return

        self._current_key = new_key
        
        if self._current_key is not None:
            self.signals.keySelected.emit(self._current_key)
            self.signals.activeValueChanged.emit(self.active_value())
        else:
            # Если ключей больше нет (словарь пуст)
            self.signals.keySelected.emit("")
            self.signals.activeValueChanged.emit(None)