import sys
from typing import Any
from PySide6 import QtCore, QtWidgets

# Импортируем наши классы (пути из твоего проекта)
from BatSpec.QtUp.Reactive.reactive_dict import ReactiveDict
from BatSpec.QtUp.Selector import Selector


class DemoApp(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ReactiveDict & Selector Demo")
        self.resize(600, 500)

        # 1. Инициализация наших реактивных данных
        self.rdict = ReactiveDict[str, Any]({"player_1": "Alice", "player_2": "Bob"})

        # 2. Построение UI
        self._setup_ui()
        self._connect_signals()

        # 3. Подключение данных к селектору
        self.selector.set_dictionary(self.rdict)

        # Логгируем начальное состояние
        self.log_console.append("--- Система запущена ---")

    def _setup_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)

        # --- Зона 1: Селектор и отображение активного значения ---
        self.selector = Selector(title="Выбор активного элемента:", parent=self)
        main_layout.addWidget(self.selector)

        # Редактор значения выбранного ключа
        val_layout = QtWidgets.QHBoxLayout()
        val_layout.addWidget(QtWidgets.QLabel("Значение активного ключа:"))
        self.value_edit = QtWidgets.QLineEdit()
        val_layout.addWidget(self.value_edit)
        
        self.btn_save_val = QtWidgets.QPushButton("Сохранить изменение в Dict")
        val_layout.addWidget(self.btn_save_val)
        main_layout.addLayout(val_layout)

        main_layout.addSpacing(20)

        # --- Зона 2: Управление словарем ---
        ctrl_group = QtWidgets.QGroupBox("Манипуляции со словарем (API Тест)")
        ctrl_layout = QtWidgets.QGridLayout(ctrl_group)
        
        # Добавление/Изменение
        self.input_key = QtWidgets.QLineEdit()
        self.input_key.setPlaceholderText("Ключ (например, player_3)")
        self.input_val = QtWidgets.QLineEdit()
        self.input_val.setPlaceholderText("Значение")
        self.btn_add = QtWidgets.QPushButton("Установить (dict[k] = v)")
        
        ctrl_layout.addWidget(self.input_key, 0, 0)
        ctrl_layout.addWidget(self.input_val, 0, 1)
        ctrl_layout.addWidget(self.btn_add, 0, 2)

        # Удаление и массовые операции
        self.btn_del = QtWidgets.QPushButton("Удалить текущий ключ (del dict[k])")
        self.btn_clear = QtWidgets.QPushButton("Очистить словарь (clear)")
        self.btn_mass = QtWidgets.QPushButton("Массовое обновление (Atomicity Test)")
        
        ctrl_layout.addWidget(self.btn_del, 1, 0)
        ctrl_layout.addWidget(self.btn_clear, 1, 1)
        ctrl_layout.addWidget(self.btn_mass, 1, 2)

        main_layout.addWidget(ctrl_group)

        # --- Зона 3: Консоль логов ---
        main_layout.addWidget(QtWidgets.QLabel("Лог сигналов ReactiveDict:"))
        self.log_console = QtWidgets.QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: monospace;")
        main_layout.addWidget(self.log_console)

    def _connect_signals(self) -> None:
        # --- Подписка на UI Селектора ---
        self.selector.signals.keySelected.connect(self._on_key_selected)
        self.selector.signals.activeValueChanged.connect(self._on_value_changed)

        # --- Кнопки управления ---
        self.btn_save_val.clicked.connect(self._save_current_value)
        self.btn_add.clicked.connect(self._add_custom_item)
        self.btn_del.clicked.connect(self._delete_current_item)
        self.btn_clear.clicked.connect(self.rdict.clear)
        self.btn_mass.clicked.connect(self._run_mass_update)

        # --- ШПИОНАЖ ЗА СИГНАЛАМИ СЛОВАРЯ (Для логгера) ---
        self.rdict.signals.itemSet.connect(lambda k, v, old: self._log(f"[itemSet] '{k}' = {v} (было: {old})"))
        self.rdict.signals.itemRemoved.connect(lambda k, old: self._log(f"[itemRemoved] '{k}' (удалено: {old})"))
        self.rdict.signals.cleared.connect(lambda: self._log("[cleared] Словарь очищен"))
        self.rdict.signals.updated.connect(lambda keys: self._log(f"[updated] Массово изменены ключи: {keys}"))
        self.rdict.signals.changed.connect(lambda: self._log("[changed] *** СИГНАЛ CHANGED ***\n"))

    # ==================== Обработчики UI ====================

    def _log(self, text: str) -> None:
        self.log_console.append(text)
        # Автопрокрутка вниз
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_key_selected(self, key: str) -> None:
        if not key:
            self.value_edit.clear()
            self.value_edit.setEnabled(False)
            self.btn_save_val.setEnabled(False)
        else:
            self.value_edit.setEnabled(True)
            self.btn_save_val.setEnabled(True)

    def _on_value_changed(self, value: Any) -> None:
        # Обновляем QLineEdit, когда данные меняются в словаре
        if value is not None:
            self.value_edit.setText(str(value))
        else:
            self.value_edit.clear()

    # ==================== Манипуляции со словарем ====================

    def _save_current_value(self) -> None:
        key = self.selector.active_key()
        if key:
            # Изменяем существующий ключ!
            self.rdict[key] = self.value_edit.text()

    def _add_custom_item(self) -> None:
        key = self.input_key.text().strip()
        val = self.input_val.text().strip()
        if key:
            # Добавляем новый ключ или меняем старый
            self.rdict[key] = val
            # Очищаем инпуты для удобства
            self.input_key.clear()
            self.input_val.clear()

    def _delete_current_item(self) -> None:
        key = self.selector.active_key()
        if key and key in self.rdict:
            del self.rdict[key]

    def _run_mass_update(self) -> None:
        self._log("--- Запуск mass update... ---")
        # Массовое добавление. Обрати внимание, что player_1 останется без изменений!
        # Значит для него сигнал itemSet не сработает (No-op защита).
        self.rdict.update({
            "player_1": "Alice",         # No-op (значение не меняется)
            "enemy_1": "Goblin",         # Новый
            "enemy_2": "Orc",            # Новый
            "player_2": "Bob The Great"  # Измененный
        })


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Для красивого отображения под Windows 11/Mac
    app.setStyle("Fusion") 
    
    window = DemoApp()
    window.show()
    sys.exit(app.exec())