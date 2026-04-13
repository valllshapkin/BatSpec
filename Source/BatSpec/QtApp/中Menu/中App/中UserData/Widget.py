import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QComboBox, QPlainTextEdit, QWidget, QMessageBox
)
from PySide6.QtGui import (
    QSyntaxHighlighter, QTextCharFormat, QColor, QFont
)
from PySide6.QtCore import Qt, QRegularExpression


from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales, Settings as LocalesSettings
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes, Settings as ThemeSettings
from BatSpec.QtApp.Shared.Component import Component

# --- Подсветка синтаксиса для INI/Conf ---
class IniHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        # Секции: [Section]
        section_format = QTextCharFormat()
        section_format.setForeground(QColor("#E67E22"))  # Оранжевый
        section_format.setFontWeight(QFont.Weight.Bold)
        self.highlighting_rules.append((QRegularExpression(r"\[.*\]"), section_format))

        # Ключи: key =
        key_format = QTextCharFormat()
        key_format.setForeground(QColor("#3498DB"))  # Голубой
        self.highlighting_rules.append((QRegularExpression(r"^\s*[^=\s#;]+(?=\s*=)"), key_format))

        # Комментарии: # или ;
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#7F8C8D"))  # Серый
        self.highlighting_rules.append((QRegularExpression(r"[#;].*"), comment_format))

        # Значения в кавычках
        value_format = QTextCharFormat()
        value_format.setForeground(QColor("#2ECC71"))  # Зеленый
        self.highlighting_rules.append((QRegularExpression(r"\".*\""), value_format))

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            expression = QRegularExpression(pattern)
            iterator = expression.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)


class UserDataDialog(*Component, QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setModal(True)
        self.resize(700, 500)
        
        self.layout = QVBoxLayout(self)
        self.current_file_path = ""

        # --- Верхняя панель выбора ---
        self.top_panel = QHBoxLayout()
        self.lbl_select = QLabel()
        self.combo_files = QComboBox()
        
        # Добавляем пути к файлам
        self.combo_files.addItem(self.tr("Theme Settings"), ThemeSettings.SETTING_PATH)
        self.combo_files.addItem(self.tr("Locales Settings"), LocalesSettings.SETTING_PATH)
        
        self.top_panel.addWidget(self.lbl_select)
        self.top_panel.addWidget(self.combo_files, 1)
        self.layout.addLayout(self.top_panel)

        # --- Редактор кода ---
        self.editor = QPlainTextEdit()
        # Устанавливаем моноширинный шрифт (как в VS Code / Notepad++)
        font = QFont("Consolas", 11) or QFont("Monospace", 11)
        font.setFixedPitch(True)
        self.editor.setFont(font)
        self.editor.setTabStopDistance(20)
        
        # Подключаем подсветку
        self.highlighter = IniHighlighter(self.editor.document())
        self.layout.addWidget(self.editor)

        # --- Нижняя панель кнопок ---
        self.bottom_panel = QHBoxLayout()
        self.btn_save = QPushButton()
        self.btn_reload = QPushButton()
        
        self.bottom_panel.addStretch()
        self.bottom_panel.addWidget(self.btn_reload)
        self.bottom_panel.addWidget(self.btn_save)
        self.layout.addLayout(self.bottom_panel)

        # --- Сигналы ---
        self.combo_files.currentIndexChanged.connect(self._on_file_changed)
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.btn_reload.clicked.connect(self._load_current_file)

        # Инициализация
        self.onLanguageChange()
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
        if not self.current_file_path:
            return

        try:
            content = self.editor.toPlainText()
            with open(self.current_file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            # Всплывающее уведомление (опционально)
            # QMessageBox.information(self, self.tr("Success"), self.tr("File saved successfully!"))
            
            # Показываем статус на кнопке на секунду (мини-фидбек)
            original_text = self.btn_save.text()
            self.btn_save.setText(self.tr("Saved!"))
            from PySide6.QtCore import QTimer
            QTimer.singleShot(1000, lambda: self.btn_save.setText(original_text))

        except Exception as e:
            QMessageBox.critical(self, self.tr("Error"), f"Could not save file: {e}")

    def onLanguageChange(self):
        self.setWindowTitle(self.tr("Configuration Editor"))
        self.lbl_select.setText(self.tr("File to edit:"))
        self.btn_save.setText(self.tr("Save Changes"))
        self.btn_reload.setText(self.tr("Reload"))
        
        # Обновляем названия в списке
        self.combo_files.setItemText(0, self.tr("Theme Settings"))
        self.combo_files.setItemText(1, self.tr("Locales Settings"))
        
        return super().onLanguageChange()

    def onThemeChange(self):
        # Если вы используете темные темы приложения, можно подправить цвета подсветки здесь
        return super().onThemeChange()