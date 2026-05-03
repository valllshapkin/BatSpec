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
