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
