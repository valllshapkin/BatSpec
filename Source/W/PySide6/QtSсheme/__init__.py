from typing import TYPE_CHECKING, Any
from PySide6 import QtWidgets, QtCore, QtGui # type: ignore

class ComponentLifecycle(QtWidgets.QWidget if TYPE_CHECKING else object):
    """
    Универсальный миксин, задающий строгий жизненный цикл виджета.
    Избавляет от монолитного __init__ и предотвращает SegFault.
    """
    
    def __init_state__(self):
        """1. Инициализация переменных, флагов, моделей данных."""
        pass

    def __init_graph__(self):
        """2. Построение визуального дерева."""
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

    def onResize(self, event: QtGui.QResizeEvent):
        """7. Адаптивный дизайн (срабатывает при изменении размеров виджета)."""
        pass

    def onWindowStateChange(self, state: QtCore.Qt.WindowState):
        """8. Реакция на сворачивание/разворачивание окна."""
        pass

    def __init_ready__(self):
        """6. Пост-инициализация: загрузка данных, запуск таймеров, обновление UI."""
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # 0. Инициализация C++ объекта
        super().__init__(*args, **kwargs)
        
        # Запускаем строгий конвейер инициализации
        self.__init_state__()
        self.__init_graph__()
        self.__init_signal__()
        
        # Первичное применение темы и локали
        self.onLanguageChange()
        self.onThemeChange()
        
        self.__init_ready__()

    # ==========================================
    # ПЕРЕХВАТ НАИВНЫХ СОБЫТИЙ QT
    # ==========================================
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        if hasattr(super(), 'resizeEvent'):
            super().resizeEvent(event)
        self.onResize(event)

    def changeEvent(self, event: QtCore.QEvent) -> None:
        if hasattr(super(), 'changeEvent'):
            super().changeEvent(event)
            
        t = event.type()
        
        # Ловим смену языка ОС/Приложения
        if t == QtCore.QEvent.Type.LanguageChange:
            self.onLanguageChange()
            
        # Ловим смену палитры/темы
        elif t == QtCore.QEvent.Type.PaletteChange:
            self.onThemeChange()
            
        # Ловим разворачивание/сворачивание окна
        elif t == QtCore.QEvent.Type.WindowStateChange:
            window = self.window()
            if window:
                self.onWindowStateChange(window.windowState())
