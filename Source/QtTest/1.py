from typing import TYPE_CHECKING
from PySide6 import QtWidgets

class ComponentLifecycle(QtWidgets.QWidget if TYPE_CHECKING else object):
    """
    Универсальный миксин, задающий строгий жизненный цикл виджета.
    Избавляет от монолитного __init__ и предотвращает SegFault.
    """
    
    def __init_state__(self):
        """1. Инициализация переменных, флагов, моделей данных."""
        pass

    def __init_graph__(self):
        """2. Построение визуального дерева (with b())."""
        pass

    def __init_signal__(self):
        """3. Подключение сигналов и событий."""
        pass

    def onLanguageChange(self):
        """4. Применение текстов и переводов."""
        pass

    def onThemeChange(self):
        """5. Применение стилей, цветов и иконок."""
        pass

    def __init_ready__(self):
        """6. Пост-инициализация: загрузка данных, запуск таймеров, обновление UI."""
        pass

    def __init__(self, *args, **kwargs) -> None:
        # 0. Инициализация C++ объекта (доходит до QWidget/QPushButton)
        super().__init__(*args, **kwargs)
        
        # Запускаем строгий конвейер инициализации
        self.__init_state__()
        self.__init_graph__()
        self.__init_signal__()
        
        # Динамические методы (вызываются и при старте, и при смене извне)
        self.onLanguageChange()
        self.onThemeChange()
        
        self.__init_ready__()

import sys
from PySide6 import QtWidgets, QtCore
from BatSpec.QtUp.Builder import build_node as b

class UserProfileCard(ComponentLifecycle, QtWidgets.QFrame):
    
    # ---------------------------------------------------------
    # 1. СОСТОЯНИЕ
    # ---------------------------------------------------------
    def __init_state__(self):
        # UI еще не существует. Подготавливаем данные.
        self.user_id = 42
        self.user_data = None
        self.is_loading = True

    # ---------------------------------------------------------
    # 2. ГРАФИКА
    # ---------------------------------------------------------
    def __init_graph__(self):
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        
        with b(self, QtWidgets.QVBoxLayout()) as layout:
            # Метка статуса
            with b(layout, QtWidgets.QLabel()) as self.lbl_status:
                self.lbl_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            
            # Данные пользователя
            with b(layout, QtWidgets.QFormLayout()) as form:
                with b(form, QtWidgets.QLabel()) as self.lbl_name_val:
                    self.lbl_name_key = QtWidgets.QLabel()
                    form.addRow(self.lbl_name_key, self.lbl_name_val)
                    
                with b(form, QtWidgets.QLabel()) as self.lbl_role_val:
                    self.lbl_role_key = QtWidgets.QLabel()
                    form.addRow(self.lbl_role_key, self.lbl_role_val)
            
            # Кнопка
            with b(layout, QtWidgets.QPushButton()) as self.btn_refresh:
                self.btn_refresh.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    # ---------------------------------------------------------
    # 3. СИГНАЛЫ
    # ---------------------------------------------------------
    def __init_signal__(self):
        self.btn_refresh.clicked.connect(self._fetch_data)

    # ---------------------------------------------------------
    # 4 & 5. ЛОКАЛЬ И ТЕМА
    # ---------------------------------------------------------
    def onLanguageChange(self):
        # Устанавливаем статические тексты
        self.lbl_name_key.setText("Username:")
        self.lbl_role_key.setText("Role:")
        self.btn_refresh.setText("Reload Profile")
        
        # Обновляем динамические тексты
        self._update_ui_state()

    def onThemeChange(self):
        self.setStyleSheet("""
            QFrame { background-color: #ffffff; border-radius: 8px; border: 1px solid #bdc3c7; }
            QLabel { color: #2c3e50; font-size: 14px; }
            QPushButton { background-color: #27ae60; color: white; border: none; border-radius: 4px; padding: 6px; }
            QPushButton:hover { background-color: #2ecc71; }
        """)

    # ---------------------------------------------------------
    # 6. ПОСТ-ИНИЦИАЛИЗАЦИЯ (ГОТОВНОСТЬ)
    # ---------------------------------------------------------
    def __init_ready__(self):
        # Графика построена, переводы применены. Можно грузить данные!
        self._fetch_data()

    # --- Бизнес-логика ---
    def _fetch_data(self):
        self.is_loading = True
        self._update_ui_state()
        
        # Имитация асинхронной загрузки из базы данных (через 1 секунду)
        QtCore.QTimer.singleShot(1000, self._on_data_loaded)

    def _on_data_loaded(self):
        # Имитация ответа от БД
        self.user_data = {"name": "Admin BatSpec", "role": "Superuser"}
        self.is_loading = False
        self._update_ui_state()

    def _update_ui_state(self):
        """Метод, который рисует актуальное состояние (state) на экране"""
        if self.is_loading:
            self.lbl_status.setText("Loading data...")
            self.lbl_name_val.setText("...")
            self.lbl_role_val.setText("...")
            self.btn_refresh.setEnabled(False)
        else:
            self.lbl_status.setText("Data loaded successfully!")
            self.lbl_name_val.setText(self.user_data.get("name", ""))
            self.lbl_role_val.setText(self.user_data.get("role", ""))
            self.btn_refresh.setEnabled(True)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Создаем окно, кладем туда нашу сложную карточку
    win = QtWidgets.QWidget()
    win.resize(300, 250)
    with b(win, QtWidgets.QVBoxLayout()) as l:
        with b(l, UserProfileCard()) as card: pass
        
    win.show()
    sys.exit(app.exec())