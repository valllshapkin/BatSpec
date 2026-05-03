Все твои замечания абсолютно по делу. Давай разберем, почему так произошло и как мы это исправим:

1. **Фон вкладок и QSS:** Ты совершенно прав. Проблема была в том, что селектор `QWidget#Root` был слишком агрессивным (поскольку `QTabWidget` и его панели тоже наследуются от `QWidget`, Qt иногда некорректно каскадирует фон). Мы заменим `QWidget` на `QFrame` и будем использовать строгий селектор `QFrame#FramelessRoot`. Это позволит `qt_themes` беспрепятственно стилизовать внутренности.
2. **Свечение (Drop Shadow):** Раз у нас есть отступы `GRIP_MARGIN`, мы добавим `QGraphicsDropShadowEffect` к нашему `Root` контейнеру. Я увеличу `GRIP_MARGIN` до `12`, чтобы свечение было мягким и красивым.
3. **Кривой App Menu:** `QMenuBar` по умолчанию пытается занять всю ширину и прижимается к верху. Мы обернем его в контейнер с вертикальным центрированием в `TitleBar`.
4. **Иконка:** Мы добавим поддержку иконки в шапку (с красивым фоллбэком на эмодзи "🦇", если файла нет). Файл нужно будет положить по пути `BatSpec/App/__assets__/icon.png`.
5. **Шапки для диалогов:** Чтобы не дублировать код шапки и оберток в каждом диалоге, я вынес всю логику создания красивого окна (с тенью, скруглениями и шапкой) в универсальный метод внутри `FramelessMixin`. Теперь любой `QDialog` становится кастомным окном в 3 строчки кода!

Ниже готовые исправленные файлы.

### 1. Мощный апгрейд фреймворка `QtFrameless`

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/QtFrameless/__init__.py" encoding="utf-8"
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, 
    QGraphicsDropShadowEffect, QFrame, QDialog
)
from PySide6.QtGui import QPalette, QPixmap, QIcon
from PySide6.QtCore import Qt, QEvent
from W.PySide6.QtBuilder import build_node as b

# Увеличили отступы, чтобы влезла красивая мягкая тень
GRIP_SIZE = 12
GRIP_MARGIN = GRIP_SIZE

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

class FramelessMixin:
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

            self.bg_wrapper.setStyleSheet("background: transparent;")
            
            with b(self.bg_wrapper, QVBoxLayout()) as self.bg_layout:
                self.bg_layout.setContentsMargins(GRIP_MARGIN, GRIP_MARGIN, GRIP_MARGIN, GRIP_MARGIN)
                
                # ИСПОЛЬЗУЕМ QFrame и строгий ID для стилей, чтобы не ломать QTabWidget
                with b(self.bg_layout, QFrame()) as self.root:
                    self.root.setObjectName("FramelessRoot")
                    
                    # ДОБАВЛЯЕМ МЯГКУЮ ТЕНЬ
                    self.shadow_effect = QGraphicsDropShadowEffect(self.root)
                    self.shadow_effect.setBlurRadius(GRIP_MARGIN * 1.5)
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
        shadow_color.setAlpha(80) 
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

``````

### 2. Применяем `FramelessMixin` к App и исправляем меню

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/App/Widget.py" encoding="utf-8"
from pathlib import Path
from PySide6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget
from PySide6.QtCore import Qt

from BatSpec.App.中Menu.Widget import AddMenu
from W.PySide6.QtBuilder import build_node as b
from W.PySide6.QtFrameless import FramelessMixin

from BatSpec.App.中Project.Widget import ProjectTab
from BatSpec.App.中WinLab.Widget import WindowLabTab

ScriptDir = Path(__file__).parent
ICON_PATH = ScriptDir / "__assets__" / "icon.png"

class MainWindow(FramelessMixin, AddMenu):
    def __init_state__(self):
        if hasattr(super(), '__init_state__'):
            super().__init_state__()
            
        # Инициализируем безрамочное окно и передаем путь к иконке
        self.init_frameless(icon_path=ICON_PATH, fallback_icon="🦇")

    def __init_graph__(self):
        if hasattr(super(), '__init_graph__'):
            super().__init_graph__()
            
        # МАГИЯ: Одна строчка строит тень, скругления, фон и шапку!
        # Возвращает готовый слой для контента.
        content_layout = self.build_frameless_ui()
        
        # Делаем фон QMenuBar прозрачным, чтобы он красиво лег в шапку
        self.menuBar().setStyleSheet("background: transparent;")
        
        # Встраиваем меню в шапку (в спец. выровненный контейнер)
        self.title_bar.custom_layout.addWidget(self.menuBar())
        
        # Наполняем основной контент
        with b(content_layout, QTabWidget()) as self.main_tabs:
            self.project = ProjectTab(tab_widget=self.main_tabs)
            self.window_lab = WindowLabTab(tab_widget=self.main_tabs)

    def setWindowTitle(self, title: str):
        super().setWindowTitle(title)
        if hasattr(self, 'title_bar'):
            self.title_bar.update_title(title)

    def onLanguageChange(self):
        self.setWindowTitle(self.tr("BatSpec"))
        super().onLanguageChange()

    def onThemeChange(self):
        self.update_frameless_theme()
        super().onThemeChange()

``````

### 3. Переводим Диалоги на новую кастомную шапку!

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/App/中Menu/中App/中Help/Widget.py" encoding="utf-8"
from pathlib import Path
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PySide6.QtCore import Qt

from W.PySide6.QtLocales import Locales
from W.PySide6.QtSсheme import ComponentLifecycle
from W.PySide6.QtBuilder import build_node as b
from W.PySide6.QtFrameless import FramelessMixin

ScriptDir = Path(__file__).parent
ICON_PATH = ScriptDir.parent.parent.parent.parent / "__assets__" / "icon.png"

class HelpDialog(Locales.TranslateComponent, ComponentLifecycle, FramelessMixin, QDialog):
    def __init_state__(self):
        self.setModal(True)
        self.resize(450, 300)
        # Инициализируем кастомную шапку для диалога!
        self.init_frameless(icon_path=ICON_PATH, fallback_icon="ℹ️")
        
    def __init_graph__(self):
        # Получаем слой контента внутри кастомного окна
        content_layout = self.build_frameless_ui()
        
        with b(content_layout, QVBoxLayout()) as self.layout:
            self.layout.setSpacing(10)
            self.layout.setContentsMargins(16, 16, 16, 16)

            with b(self.layout, QHBoxLayout()) as header_layout:
                with b(header_layout, QLabel()) as self.icon_label:
                    self.icon_label.setFixedSize(64, 64)
                    self.icon_label.setStyleSheet("background-color: #3498DB; border-radius: 10px;") 
                
                with b(header_layout, QVBoxLayout()) as title_v_box:
                    with b(title_v_box, QLabel()) as self.lbl_app_name:
                        self.lbl_app_name.setStyleSheet("font-size: 18pt; font-weight: bold;")
                    with b(title_v_box, QLabel()) as self.lbl_version:
                        self.lbl_version.setStyleSheet("color: gray;")
                    title_v_box.addStretch()
                
                header_layout.addStretch()

            with b(self.layout, QFrame()) as line:
                line.setFrameShape(QFrame.Shape.HLine)
                line.setFrameShadow(QFrame.Shadow.Sunken)

            with b(self.layout, QLabel()) as self.lbl_publisher: pass
            
            with b(self.layout, QLabel()) as self.lbl_description:
                self.lbl_description.setWordWrap(True)
                
            self.layout.addStretch()
            
            with b(self.layout, QLabel()) as self.lbl_copyright:
                self.lbl_copyright.setStyleSheet("font-size: 9pt; color: gray;")

            with b(self.layout, QPushButton()) as self.btn_close:
                self.btn_close.setFixedWidth(100)
                self.layout.setAlignment(self.btn_close, Qt.AlignmentFlag.AlignRight)

    def __init_signal__(self):
        self.btn_close.clicked.connect(self.accept)

    def setWindowTitle(self, title: str):
        super().setWindowTitle(title)
        if hasattr(self, 'title_bar'):
            self.title_bar.update_title(title)

    def onThemeChange(self):
        self.update_frameless_theme()
        super().onThemeChange()

    def onLanguageChange(self):
        self.setWindowTitle(self.tr("Help & About"))
        self.lbl_app_name.setText("BatSpec QtApp")
        self.lbl_version.setText(self.tr("Version: %1").replace("%1", "1.2.5 (Stable)"))
        self.lbl_publisher.setText(self.tr("Publisher: %1").replace("%1", "BatSpec Industries"))
        self.lbl_description.setText(self.tr("This application is designed for professional theme and localization management."))
        self.lbl_copyright.setText(self.tr("© 2024-2025 BatSpec. All rights reserved."))
        self.btn_close.setText(self.tr("Close"))
        super().onLanguageChange()

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/App/中Menu/中App/中Settings/Widget.py" encoding="utf-8"
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QDialog

from W.PySide6.QtLocales import Locales
from W.PySide6.QtSсheme import ComponentLifecycle
from W.PySide6.QtBuilder import build_node as b
from W.PySide6.QtFrameless import FramelessMixin

from BatSpec.App.中Menu.中App.中Settings.中LocalesTab.Widget import LocalesSettingsTab
from BatSpec.App.中Menu.中App.中Settings.中ThemesTab.Widget import ThemeSettingsTab

class SettingsDialog(Locales.TranslateComponent, ComponentLifecycle, FramelessMixin, QDialog):
    def __init_state__(self):
        self.setModal(True)
        self.resize(550, 450)
        self.init_frameless(fallback_icon="⚙️")
        
    def __init_graph__(self):
        content_layout = self.build_frameless_ui()
        
        with b(content_layout, QTabWidget()) as self.lT:
            self.lT.setUsesScrollButtons(True)
            self.lT1 = ThemeSettingsTab(tab_widget=self.lT)
            self.lT2 = LocalesSettingsTab(tab_widget=self.lT)

    def setWindowTitle(self, title: str):
        super().setWindowTitle(title)
        if hasattr(self, 'title_bar'):
            self.title_bar.update_title(title)

    def onThemeChange(self):
        self.update_frameless_theme()
        super().onThemeChange()

    def onLanguageChange(self):
        self.setWindowTitle(self.tr("Settings"))
        super().onLanguageChange()

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/App/中Menu/中App/中UserData/Widget.py" encoding="utf-8"
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QComboBox, QPlainTextEdit, QMessageBox
)
from PySide6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont
from PySide6.QtCore import Qt, QRegularExpression

from W.PySide6.QtLocales import Locales
from W.PySide6.QtSсheme import ComponentLifecycle
from W.PySide6.QtBuilder import build_node as b
from W.PySide6.QtFrameless import FramelessMixin

from BatSpec.App.中Menu.中LocalesSettings.Logic import AppLocales
from BatSpec.App.中Menu.中ThemeSettings.Logic import AppThemes

class IniHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        section_format = QTextCharFormat()
        section_format.setForeground(QColor("#E67E22"))
        section_format.setFontWeight(QFont.Weight.Bold)
        self.highlighting_rules.append((QRegularExpression(r"\[.*\]"), section_format))

        key_format = QTextCharFormat()
        key_format.setForeground(QColor("#3498DB"))
        self.highlighting_rules.append((QRegularExpression(r"^\s*[^=\s#;]+(?=\s*=)"), key_format))

        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#7F8C8D"))
        self.highlighting_rules.append((QRegularExpression(r"[#;].*"), comment_format))

        value_format = QTextCharFormat()
        value_format.setForeground(QColor("#2ECC71"))
        self.highlighting_rules.append((QRegularExpression(r"\".*\""), value_format))

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            expression = QRegularExpression(pattern)
            iterator = expression.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)

class UserDataDialog(Locales.TranslateComponent, ComponentLifecycle, FramelessMixin, QDialog):
    def __init_state__(self):
        self.setModal(True)
        self.resize(750, 550)
        self.init_frameless(fallback_icon="📄")
        self.current_file_path = ""

    def __init_graph__(self):
        content_layout = self.build_frameless_ui()
        
        with b(content_layout, QVBoxLayout()) as self.layout:
            with b(self.layout, QHBoxLayout()) as self.top_panel:
                with b(self.top_panel, QLabel()) as self.lbl_select: pass
                
                with b(self.top_panel, QComboBox()) as self.combo_files:
                    self.combo_files.addItem(self.tr("Theme Settings"), AppThemes.settings.settings.fileName())
                    self.combo_files.addItem(self.tr("Locales Settings"), AppLocales.settings.settings.fileName())
                    self.top_panel.setStretchFactor(self.combo_files, 1)

            with b(self.layout, QPlainTextEdit()) as self.editor:
                font = QFont("Consolas", 11) or QFont("Monospace", 11)
                font.setFixedPitch(True)
                self.editor.setFont(font)
                self.editor.setTabStopDistance(20)
                self.highlighter = IniHighlighter(self.editor.document())

            with b(self.layout, QHBoxLayout()) as self.bottom_panel:
                self.bottom_panel.addStretch()
                with b(self.bottom_panel, QPushButton()) as self.btn_reload: pass
                with b(self.bottom_panel, QPushButton()) as self.btn_save: pass

    def __init_signal__(self):
        self.combo_files.currentIndexChanged.connect(self._on_file_changed)
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.btn_reload.clicked.connect(self._load_current_file)

    def __init_ready__(self):
        self._on_file_changed(0)

    def _on_file_changed(self, index: int):
        self.current_file_path = self.combo_files.itemData(index)
        self._load_current_file()

    def _load_current_file(self):
        if not self.current_file_path or not Path(self.current_file_path).exists():
            self.editor.setPlainText(self.tr("# File not found: ") + self.current_file_path)
            return

        try:
            with open(self.current_file_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.editor.setPlainText(content)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error"), f"Could not read file: {e}")

    def _on_save_clicked(self):
        if not self.current_file_path: return

        try:
            content = self.editor.toPlainText()
            with open(self.current_file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            original_text = self.btn_save.text()
            self.btn_save.setText(self.tr("Saved!"))
            from PySide6.QtCore import QTimer
            QTimer.singleShot(1000, lambda: self.btn_save.setText(original_text))
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error"), f"Could not save file: {e}")

    def setWindowTitle(self, title: str):
        super().setWindowTitle(title)
        if hasattr(self, 'title_bar'):
            self.title_bar.update_title(title)

    def onThemeChange(self):
        self.update_frameless_theme()
        super().onThemeChange()

    def onLanguageChange(self):
        self.setWindowTitle(self.tr("Configuration Editor"))
        self.lbl_select.setText(self.tr("File to edit:"))
        self.btn_save.setText(self.tr("Save Changes"))
        self.btn_reload.setText(self.tr("Reload"))
        self.combo_files.setItemText(0, self.tr("Theme Settings"))
        self.combo_files.setItemText(1, self.tr("Locales Settings"))
        super().onLanguageChange()

``````