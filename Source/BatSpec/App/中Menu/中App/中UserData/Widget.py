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
