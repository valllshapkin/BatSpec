from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, 
    QGraphicsDropShadowEffect, QFrame, QDialog
)
from PySide6.QtGui import QPalette, QPixmap
from PySide6.QtCore import Qt, QEvent
from W.PySide6.QtBuilder import build_node as b
from typing import TYPE_CHECKING

# Увеличили отступы, чтобы влезла красивая мягкая тень
GRIP_MARGIN = 6
GRIP_SIZE = 12

class WindowEdgeGrip(QWidget):
    def __init__(self, parent, edge: Qt.Edge, cursor: Qt.CursorShape):
        super().__init__(parent)
        self.edge = edge
        self.setCursor(cursor)
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
        self.setFlat(True)

class TitleBar(QWidget):
    def __init__(self, parent, icon_path: Path | None = None, fallback_icon: str = "🦇"):
        super().__init__(parent)
        self.setFixedHeight(36)
        
        with b(self, QHBoxLayout()) as self.layout:
            self.layout.setContentsMargins(12, 0, 0, 0)
            self.layout.setSpacing(8)
            
            # 1. ИКОНКА (Загрузка файла или фоллбэк)
            with b(self.layout, QLabel()) as self.icon_label:
                self.icon_label.setFixedSize(20, 20)
                if icon_path and icon_path.exists():
                    pixmap = QPixmap(str(icon_path)).scaled(
                        20, 20, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                    )
                    self.icon_label.setPixmap(pixmap)
                else:
                    self.icon_label.setText(fallback_icon)
                    self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # 2. ЗАГОЛОВОК
            with b(self.layout, QLabel(parent.windowTitle())) as self.title_label:
                font = self.title_label.font()
                font.setPointSize(10)
                font.setBold(True)
                self.title_label.setFont(font)
                
            # 3. КОНТЕЙНЕР ДЛЯ МЕНЮ (Выровнен по центру!)
            with b(self.layout, QHBoxLayout()) as self.custom_layout:
                self.custom_layout.setContentsMargins(8, 0, 8, 0)
                self.custom_layout.setSpacing(8)
                self.custom_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
                
            self.layout.addStretch()
            
            # 4. КНОПКИ УПРАВЛЕНИЯ ОКНОМ
            # Для диалогов (QDialog) кнопка сворачивания/разворачивания может быть не нужна,
            # но мы оставляем их для универсальности, либо прячем.
            if not isinstance(parent, QDialog):
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
        if event.button() == Qt.MouseButton.LeftButton and not isinstance(self.window(), QDialog):
            self._toggle_maximize()

class FramelessMixin(QWidget if TYPE_CHECKING else object):
    """Универсальный миксин для QMainWindow и QDialog"""
    
    def init_frameless(self, icon_path: Path | None = None, fallback_icon: str = "🦇"):
        """Настраивает флаги окна и создает грипы ресайза"""
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
            
        self._icon_path = icon_path
        self._fallback_icon = fallback_icon

    def build_frameless_ui(self) -> QVBoxLayout:
        """
        УТИЛИТА: Автоматически строит обертку, тень, скругления и шапку.
        Возвращает QVBoxLayout (content_layout), в который нужно добавлять ваш полезный UI.
        """
        with b(self, QWidget()) as self.bg_wrapper:
            # Для QMainWindow устанавливаем как CentralWidget, для QDialog просто оставляем
            if hasattr(self, 'setCentralWidget'):
                self.setCentralWidget(self.bg_wrapper)
            else:
                self.setLayout(QVBoxLayout())
                self.layout().setContentsMargins(0, 0, 0, 0)
                self.layout().addWidget(self.bg_wrapper)

            # self.bg_wrapper.setStyleSheet("background: transparent;")
            
            with b(self.bg_wrapper, QVBoxLayout()) as self.bg_layout:
                self.bg_layout.setContentsMargins(GRIP_MARGIN, GRIP_MARGIN, GRIP_MARGIN, GRIP_MARGIN)
                
                # ИСПОЛЬЗУЕМ QFrame и строгий ID для стилей, чтобы не ломать QTabWidget
                with b(self.bg_layout, QFrame()) as self.root:
                    self.root.setObjectName("FramelessRoot")
                    
                    # ДОБАВЛЯЕМ МЯГКУЮ ТЕНЬ
                    self.shadow_effect = QGraphicsDropShadowEffect(self.root)
                    self.shadow_effect.setBlurRadius(GRIP_MARGIN * 5)
                    self.shadow_effect.setOffset(0, 0)
                    self.root.setGraphicsEffect(self.shadow_effect)
                    
                    with b(self.root, QVBoxLayout()) as self.root_v:
                        self.root_v.setContentsMargins(0, 0, 0, 0)
                        self.root_v.setSpacing(0)
                        
                        # Добавляем нашу кастомную шапку
                        with b(self.root_v, TitleBar(self, self._icon_path, self._fallback_icon)) as self.title_bar: pass
                        
                        # Контейнер для пользовательского контента
                        with b(self.root_v, QWidget()) as self.content_widget:
                            with b(self.content_widget, QVBoxLayout()) as self.content_layout:
                                self.content_layout.setContentsMargins(4, 4, 4, 4)
                                
        self.update_frameless_theme()
        return self.content_layout

    def update_frameless_theme(self):
        """Обновляет стили рамки и цвет тени в зависимости от темы"""
        if not hasattr(self, 'root'): return
        
        bg_color = self.palette().color(QPalette.ColorRole.Window)
        border_color = self.palette().color(QPalette.ColorRole.Shadow)
        text_color = self.palette().color(QPalette.ColorRole.WindowText)
        
        # Настройка цвета тени (черная для светлой темы, светящаяся для темной)
        shadow_color = border_color
        # shadow_color.setAlpha(255) 
        self.shadow_effect.setColor(shadow_color)
        
        if self.isMaximized():
            self.bg_layout.setContentsMargins(0, 0, 0, 0)
            self.root.setStyleSheet(f"QFrame#FramelessRoot {{ background-color: {bg_color.name()}; border-radius: 0px; border: none; }}")
        else:
            self.bg_layout.setContentsMargins(GRIP_MARGIN, GRIP_MARGIN, GRIP_MARGIN, GRIP_MARGIN)
            self.root.setStyleSheet(f"QFrame#FramelessRoot {{ background-color: {bg_color.name()}; border-radius: 8px; border: 1px solid {border_color.name()}; }}")

    def _update_grips(self):
        if not hasattr(self, '_grips'): return
        w, h = self.width(), self.height()
        s = GRIP_SIZE
        
        self._grips[0].setGeometry(s, 0, w - 2*s, s)
        self._grips[1].setGeometry(s, h - s, w - 2*s, s)
        self._grips[2].setGeometry(0, s, s, h - 2*s)
        self._grips[3].setGeometry(w - s, s, s, h - 2*s)
        
        self._grips[4].setGeometry(0, 0, s, s)
        self._grips[5].setGeometry(w - s, 0, s, s)
        self._grips[6].setGeometry(0, h - s, s, s)
        self._grips[7].setGeometry(w - s, h - s, s, s)
        
        for grip in self._grips:
            grip.raise_()

    def resizeEvent(self, event):
        if hasattr(super(), 'resizeEvent'):
            super().resizeEvent(event)
        self._update_grips()

    def changeEvent(self, event):
        if hasattr(super(), 'changeEvent'):
            super().changeEvent(event)
            
        if event.type() == QEvent.Type.WindowStateChange:
            if hasattr(self, '_grips'):
                is_max = self.isMaximized()
                for grip in self._grips:
                    grip.setVisible(not is_max)
                    
            if hasattr(self, 'title_bar') and hasattr(self.title_bar, 'max_btn'):
                txt = "🗗" if self.isMaximized() else "◻"
                self.title_bar.max_btn.setText(txt)
                
            self.update_frameless_theme()
