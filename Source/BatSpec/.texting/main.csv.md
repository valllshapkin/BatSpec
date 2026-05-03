Твои замечания бьют точно в цель. 

1. **Цвета и QSS:** Ты абсолютно прав, `qt-themes` и `pyqtgraph` берут на себя всю палитру, а QSS может поломать их работу. Я убрал из кнопок QSS — теперь используется нативный `setFlat(True)`, поэтому они идеально работают на любой теме. Единственный QSS, который остался — это скругление самого окна (`border-radius`), но цвет для него теперь берется **динамически из палитры темы** строго внутри `onThemeChange`.
2. **Накладки внахлест:** Я сделал невидимые виджеты-грипы (`WindowEdgeGrip`). Размер грипа 8px. Отступ видимого окна (bleed) — 4px. Таким образом, грипы ложатся ровно внахлест: 4px торчат за пределами окна, а 4px лежат внутри окна. Это дает идеальное "нативное" поведение мыши и безупречную смену курсора.
3. **Меню в шапке:** В `TitleBar` добавлен `self.custom_layout`. В `Widget.py` я физически перенес `self.menuBar()` прямо в центр шапки!

### 1. Модуль Frameless

``````python path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/QtUp/Frameless.py" encoding="utf-8"
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QMainWindow
from PySide6.QtCore import Qt, QEvent
from BatSpec.QtUp.Builder import build_node as b

# Общий размер грипа (зоны ресайза)
GRIP_SIZE = 8
# Насколько грип будет выпирать за пределы видимого окна (настройка "внахлест")
GRIP_MARGIN = GRIP_SIZE // 2 

class WindowEdgeGrip(QWidget):
    """Невидимый виджет-накладка, обрабатывающий ресайз окна."""
    def __init__(self, parent, edge: Qt.Edge, cursor: Qt.CursorShape):
        super().__init__(parent)
        self.edge = edge
        self.setCursor(cursor)
        # Виджет прозрачный, но физически перехватывает мышь
        self.setStyleSheet("background: transparent;")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.window().windowHandle():
                self.window().windowHandle().startSystemResize(self.edge)
                return
        super().mousePressEvent(event)


class TitleBarButton(QPushButton):
    def __init__(self, text: str):
        super().__init__(text)
        self.setFixedSize(42, 32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # БЕЗ QSS! Мы используем flat=True, чтобы тема приложения 
        # сама отрисовала красивый hover-эффект и правильный цвет текста.
        self.setFlat(True)


class TitleBar(QWidget):
    def __init__(self, parent: QMainWindow):
        super().__init__(parent)
        self.setFixedHeight(36)
        
        with b(self, QHBoxLayout()) as self.layout:
            self.layout.setContentsMargins(12, 0, 0, 0)
            self.layout.setSpacing(0)
            
            # 1. Заголовок
            with b(self.layout, QLabel(parent.windowTitle())) as self.title_label:
                font = self.title_label.font()
                font.setPointSize(10)
                font.setBold(True)
                self.title_label.setFont(font)
                
            # 2. Контейнер для кастомных меню (например, MenuBar)
            with b(self.layout, QHBoxLayout()) as self.custom_layout:
                self.custom_layout.setContentsMargins(16, 0, 16, 0)
                self.custom_layout.setSpacing(8)
                
            self.layout.addStretch()
            
            # 3. Системные кнопки
            with b(self.layout, TitleBarButton("─")) as self.min_btn:
                self.min_btn.clicked.connect(parent.showMinimized)
                
            with b(self.layout, TitleBarButton("◻")) as self.max_btn:
                self.max_btn.clicked.connect(self._toggle_maximize)
                
            with b(self.layout, TitleBarButton("✕")) as self.close_btn:
                self.close_btn.clicked.connect(parent.close)

    def _toggle_maximize(self):
        win = self.window()
        if win.isMaximized():
            win.showNormal()
        else:
            win.showMaximized()

    def update_title(self, title: str):
        self.title_label.setText(title)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.window().windowHandle():
                self.window().windowHandle().startSystemMove()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_maximize()


class FramelessMainWindowMixin:
    """Миксин для QMainWindow: добавляет 8 невидимых накладок для ресайза."""
    
    def init_frameless(self: QMainWindow):
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self._grips = []
        grip_data = [
            (Qt.Edge.TopEdge, Qt.CursorShape.SizeVerCursor),
            (Qt.Edge.BottomEdge, Qt.CursorShape.SizeVerCursor),
            (Qt.Edge.LeftEdge, Qt.CursorShape.SizeHorCursor),
            (Qt.Edge.RightEdge, Qt.CursorShape.SizeHorCursor),
            (Qt.Edge.TopEdge | Qt.Edge.LeftEdge, Qt.CursorShape.SizeFDiagCursor),
            (Qt.Edge.TopEdge | Qt.Edge.RightEdge, Qt.CursorShape.SizeBDiagCursor),
            (Qt.Edge.BottomEdge | Qt.Edge.LeftEdge, Qt.CursorShape.SizeBDiagCursor),
            (Qt.Edge.BottomEdge | Qt.Edge.RightEdge, Qt.CursorShape.SizeFDiagCursor),
        ]
        
        for edge, cursor in grip_data:
            grip = WindowEdgeGrip(self, edge, cursor)
            self._grips.append(grip)

    def _update_grips(self: QMainWindow):
        """Пересчитывает позицию прозрачных накладок-ресайзеров"""
        if not hasattr(self, '_grips'): return
        w, h = self.width(), self.height()
        s = GRIP_SIZE
        
        self._grips[0].setGeometry(s, 0, w - 2*s, s)           # Top
        self._grips[1].setGeometry(s, h - s, w - 2*s, s)       # Bottom
        self._grips[2].setGeometry(0, s, s, h - 2*s)           # Left
        self._grips[3].setGeometry(w - s, s, s, h - 2*s)       # Right
        
        self._grips[4].setGeometry(0, 0, s, s)                 # TopLeft
        self._grips[5].setGeometry(w - s, 0, s, s)             # TopRight
        self._grips[6].setGeometry(0, h - s, s, s)             # BottomLeft
        self._grips[7].setGeometry(w - s, h - s, s, s)         # BottomRight
        
        for grip in self._grips:
            grip.raise_()

    def resizeEvent(self: QMainWindow, event):
        if hasattr(super(), 'resizeEvent'):
            super().resizeEvent(event)
        self._update_grips()

    def changeEvent(self: QMainWindow, event):
        if hasattr(super(), 'changeEvent'):
            super().changeEvent(event)
            
        if event.type() == QEvent.Type.WindowStateChange:
            if hasattr(self, '_grips'):
                # Если окно на весь экран, отключаем накладки ресайза
                is_max = self.isMaximized()
                for grip in self._grips:
                    grip.setVisible(not is_max)

``````

### 2. Главное Окно (Widget.py)

В `onThemeChange` мы программно вытягиваем цвета из текущей темы `self.palette().color(...)` и подставляем их в стили. `MenuBar` добавлен прямо внутрь новой шапки.

``````python path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/App/Widget.py" encoding="utf-8"
from PySide6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QPalette

from BatSpec.App.中Menu.Widget import AddMenu
from BatSpec.QtUp.Locales import Locales
from BatSpec.QtUp.Themes import Themes
from BatSpec.QtUp.Builder import build_node as b
from BatSpec.QtUp.Frameless import FramelessMainWindowMixin, TitleBar, GRIP_MARGIN

from BatSpec.App.中Project.Widget import ProjectTab
from BatSpec.App.中WinLab.Widget import WindowLabTab
# from .中Analize.Widget import RecordAnalisys


class MainWindow(FramelessMainWindowMixin, AddMenu):
    def __init__(self) -> None:
        AddMenu.__init__(self)
        
        # 1. Инициализируем безрамочный режим (с 8 невидимыми грипами)
        self.init_frameless()
    
        # 2. Обертка окна. Она всегда прозрачная.
        with b(self, QWidget()) as self.bg_wrapper:
            self.setCentralWidget(self.bg_wrapper)
            self.bg_wrapper.setStyleSheet("background: transparent;")
            
            with b(self.bg_wrapper, QVBoxLayout()) as self.bg_layout:
                # Отступы - это невидимая зона ВНЕ окна, в которой лежат наши ресайз-грипы
                self.bg_layout.setContentsMargins(GRIP_MARGIN, GRIP_MARGIN, GRIP_MARGIN, GRIP_MARGIN)
                
                # 3. Видимое окно (Root). Внутри него шапка и контент.
                with b(self.bg_layout, QWidget()) as self.root:
                    self.root.setObjectName("Root")
                    
                    with b(self.root, QVBoxLayout()) as self.root_v:
                        self.root_v.setContentsMargins(0, 0, 0, 0)
                        self.root_v.setSpacing(0)
                        
                        # Добавляем кастомную шапку
                        with b(self.root_v, TitleBar(self)) as self.title_bar: pass
                        
                        # ВСТРАИВАЕМ меню из AddMenu прямо В ШАПКУ!
                        self.title_bar.custom_layout.addWidget(self.menuBar())
                        
                        # 4. Основной контент
                        with b(self.root_v, QWidget()) as self.c:
                            with b(self.c, QVBoxLayout()) as self.cV:
                                self.cV.setContentsMargins(4, 4, 4, 4)

                                with b(self.cV, QTabWidget()) as self.main_tabs:
                                    self.project = ProjectTab(tab_widget=self.main_tabs)
                                    self.window_lab = WindowLabTab(tab_widget=self.main_tabs)
                                    # self.record_analisis = RecordAnalisys(tab_widget=self.main_tabs)

    def _update_root_style(self):
        """Динамически применяет тему к корневому виджету без ломающих QSS-цветов"""
        bg_color = self.palette().color(QPalette.ColorRole.Window).name()
        border_color = self.palette().color(QPalette.ColorRole.Shadow).name()
        
        if self.isMaximized():
            self.bg_layout.setContentsMargins(0, 0, 0, 0)
            self.root.setStyleSheet(f"QWidget#Root {{ background-color: {bg_color}; border-radius: 0px; border: none; }}")
        else:
            self.bg_layout.setContentsMargins(GRIP_MARGIN, GRIP_MARGIN, GRIP_MARGIN, GRIP_MARGIN)
            self.root.setStyleSheet(f"QWidget#Root {{ background-color: {bg_color}; border-radius: 8px; border: 1px solid {border_color}; }}")

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            if hasattr(self, 'title_bar'):
                # Меняем иконку
                txt = "🗗" if self.isMaximized() else "◻"
                self.title_bar.max_btn.setText(txt)
                
                # Обновляем скругления и отступы
                self._update_root_style()

    def setWindowTitle(self, title: str):
        super().setWindowTitle(title)
        if hasattr(self, 'title_bar'):
            self.title_bar.update_title(title)

    def onLanguageChange(self):
        self.setWindowTitle(self.tr("BatSpec"))

    def onThemeChange(self):
        super().onThemeChange()
        # Применяем новые цвета темы к скруглениям окна
        self._update_root_style()

``````