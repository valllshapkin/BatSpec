"""
Source/BatSpec/QtApp/中Analize/中SpecView/中Detection/Widget.py

UI модуля автодетекции.

Компоновка:
    AutoDetectWidget (QWidget)
    └── QSplitter (вертикальный)
        ├── [ВЕРХ] ScriptEditor (с toolbar)
        └── [НИЗ]  LogView + кнопка Run/Clear
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat, QTextDocument
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from .Logic import AUTO_STATE


# ── Подсветка синтаксиса Python ───────────────────────────────────────────────

class _PythonHighlighter(QSyntaxHighlighter):

    _KEYWORDS = frozenset({
        "False", "None", "True", "and", "as", "assert", "async", "await",
        "break", "class", "continue", "def", "del", "elif", "else", "except",
        "finally", "for", "from", "global", "if", "import", "in", "is",
        "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
        "while", "with", "yield",
    })

    def __init__(self, document: QTextDocument) -> None:
        super().__init__(document)

        def fmt(color: str, bold: bool = False) -> QTextCharFormat:
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold:
                f.setFontWeight(700)
            return f

        import re
        raw_rules = [
            (r"\b(" + "|".join(self._KEYWORDS) + r")\b", fmt("#cc99cd", bold=True)),
            (r'"""[\s\S]*?"""',                           fmt("#999999")),   # docstring
            (r"'''[\s\S]*?'''",                           fmt("#999999")),
            (r'"[^"\\\n]*(\\.[^"\\\n]*)*"',              fmt("#7ec699")),   # строки
            (r"'[^'\\\n]*(\\.[^'\\\n]*)*'",              fmt("#7ec699")),
            (r"#[^\n]*",                                  fmt("#6a9955")),   # комментарий
            (r"\b\d+(\.\d+)?\b",                         fmt("#f08d49")),   # числа
        ]
        self._compiled = [(re.compile(p, re.DOTALL), f) for p, f in raw_rules]

    def highlightBlock(self, text: str) -> None:
        for pattern, fmt in self._compiled:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


# ── ScriptEditor ───────────────────────────────────────────────────────────────

class ScriptEditor(QPlainTextEdit):
    """Редактор скрипта с моноширинным шрифтом и подсветкой синтаксиса."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setStyleSheet("background: #1e1e1e; color: #d4d4d4; border: none;")
        self.setPlainText(AUTO_STATE.script)

        self._hl = _PythonHighlighter(self.document())
        self.textChanged.connect(lambda: AUTO_STATE.setScript(self.toPlainText()))


# ── RunButton ──────────────────────────────────────────────────────────────────

class RunButton(QPushButton, AUTO_STATE.Trigger):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        QPushButton.__init__(self, "▶ Run", parent)
        self.clicked.connect(AUTO_STATE.run)

    def onRunningChanged(self, running: bool) -> None:
        self.setEnabled(not running)
        self.setText("⏳ Running…" if running else "▶ Run")


# ── LogView ────────────────────────────────────────────────────────────────────

class LogView(QPlainTextEdit, AUTO_STATE.Trigger):
    """Текстовый лог выполнения."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        QPlainTextEdit.__init__(self, parent)
        self.setReadOnly(True)
        font = QFont("Consolas", 9)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        self.setStyleSheet("background: #111; color: #aaa; border: none;")

    def onLogAppended(self, line: str) -> None:
        self.appendPlainText(line)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def onFinished(self, count: int) -> None:
        self.appendHtml(f'<span style="color:#4f4;">✓ Готово: {count} ROI</span>')

    def onError(self, _msg: str) -> None:
        self.appendHtml('<span style="color:#f44;">✗ Ошибка (см. выше)</span>')


# ── AutoDetectWidget ───────────────────────────────────────────────────────────

class AutoDetectWidget(QWidget):
    """
    Главный виджет модуля автодетекции.

    Верхняя часть — редактор скрипта.
    Нижняя часть — лог + кнопки Run / Clear log.

    Вся логика целевых групп и порогов живёт внутри скрипта:
    скрипт пишет напрямую в results["call"], results["context"] и т.д.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        QWidget.__init__(self, parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Toolbar ────────────────────────────────────────────────────────────
        tb = QWidget()
        tb_layout = QHBoxLayout(tb)
        tb_layout.setContentsMargins(6, 3, 6, 3)

        self._run_btn = RunButton()
        tb_layout.addWidget(self._run_btn)

        tb_layout.addSpacing(8)
        tb_layout.addWidget(QLabel("Detection script"))
        tb_layout.addStretch()

        hint = QLabel(
            '<span style="color:#888; font-size:11px;">'
            'results["call"] = bbox_to_roi(…)  |  results["context"] = …'
            '</span>'
        )
        hint.setTextFormat(Qt.TextFormat.RichText)
        tb_layout.addWidget(hint)

        root.addWidget(tb)

        # ── Основной сплиттер ──────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(splitter)

        # Редактор
        self._editor = ScriptEditor()
        splitter.addWidget(self._editor)

        # Нижняя панель: лог + кнопка очистки
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)

        log_toolbar = QWidget()
        ltl = QHBoxLayout(log_toolbar)
        ltl.setContentsMargins(6, 2, 6, 2)
        ltl.addWidget(QLabel("Log"))
        ltl.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        ltl.addWidget(clear_btn)
        bl.addWidget(log_toolbar)

        self._log = LogView()
        bl.addWidget(self._log)

        clear_btn.clicked.connect(self._log.clear)

        splitter.addWidget(bottom)
        splitter.setSizes([600, 200])