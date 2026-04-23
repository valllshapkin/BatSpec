from pathlib import Path
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                               QFileDialog, QTabWidget, QListWidget,
                               QListWidgetItem, QMessageBox)
from PySide6.QtCore import Slot, Signal, QThread, QSettings
from PySide6.QtGui import QIcon
import json
from typing import List, Optional

from BatSpec.QtUp.TabInst import TabInstance

from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Shared.Component import Component, ComponentSettings, GLOBAL_MARGINS

from ..Logic import RECORD_STATE

from pathlib import Path
ScriptDir = Path(__file__).parent


class PathChecker(QThread):
    """Поток для фоновой проверки актуальности путей"""
    
    paths_checked = Signal(dict)  # {path_str: exists}
    
    def __init__(self, paths: List[str]):
        super().__init__()
        self.paths = paths
        
    def run(self):
        results = {}
        for path_str in self.paths:
            results[path_str] = Path(path_str).exists()
        self.paths_checked.emit(results)


class Import(*Component, TabInstance):

    class Settings(ComponentSettings):
        COMPONENT_DIR = ScriptDir
    
    RECENT_FILES_KEY = "Record/Import/recent"
    MAX_RECENT_FILES = 10
    
    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        TabInstance.__init__(self, parent, tab_widget)
        
        self.settings = self.Settings().settings
        
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(*GLOBAL_MARGINS)
        
        # Кнопка выбора файла
        self.vB = QPushButton("Load WAV")          # будет переведено
        self.vB.clicked.connect(self.on_select_file)
        self.v.addWidget(self.vB)
        
        # Список недавних
        self.vL = QListWidget()
        self.vL.setMaximumHeight(200)
        self.vL.itemDoubleClicked.connect(self.on_recent_file_clicked)
        self.v.addWidget(self.vL)
        
        # Кнопка очистки
        self.vC = QPushButton("Clear history")     # будет переведено
        self.vC.clicked.connect(self.clear_recent_files)
        self.v.addWidget(self.vC)
        
        self.v.addStretch()
        
        # Загрузка и начальное обновление
        self.load_recent_files()
    
    def add_to_recent(self, file_path: Path):
        recent = self.get_recent_files()
        path_str = str(file_path.absolute())
        
        if path_str in recent:
            recent.remove(path_str)
        
        recent.insert(0, path_str)
        recent = recent[:self.MAX_RECENT_FILES]
        
        self.save_recent_files(recent)
        self.update_recent_list()
    
    def get_recent_files(self) -> List[str]:
        raw = self.settings.value(self.RECENT_FILES_KEY, "[]")
        if isinstance(raw, str):
            try:
                lst = json.loads(raw)
                if isinstance(lst, list):
                    return lst
            except:
                pass
        # fallback
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str) and raw:
            return [raw]
        return []

    def save_recent_files(self, recent: List[str]):
        self.settings.setValue(self.RECENT_FILES_KEY, json.dumps(recent))

    def update_recent_list(self):
        self.vL.clear()
        recent = self.get_recent_files()
        
        for path_str in recent:
            path = Path(path_str)
            item = QListWidgetItem()
            
            if path.exists():
                item.setIcon(QIcon.fromTheme("audio-x-generic"))
                item.setText(f"✓ {path.name}\n{path.parent}")
            else:
                item.setIcon(QIcon.fromTheme("dialog-warning"))
                item.setText(f"⚠ {path.name} (missing)\n{path.parent}")
            
            item.setData(256, path_str)
            self.vL.addItem(item)
    
    def check_paths_async(self):
        recent = self.get_recent_files()
        if recent:
            self.checker = PathChecker(recent)
            self.checker.paths_checked.connect(self.on_paths_checked)
            self.checker.start()
    
    @Slot(dict)
    def on_paths_checked(self, results: dict):
        self.update_recent_list()
        
        missing = [p for p, ex in results.items() if not ex]
        if missing and len(missing) <= 3:
            names = [Path(p).name for p in missing]
            QMessageBox.information(
                self,
                self.tr("Information"),
                self.tr("The following files are no longer available:\n{}").format("\n".join(names))
            )
    
    @Slot()
    def on_select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select WAV file"),
            "",
            self.tr("WAV files (*.wav)")
        )
        
        if path:
            file_path = Path(path)
            RECORD_STATE.setRecordFile(file_path)
            self.add_to_recent(file_path)
    
    @Slot(QListWidgetItem)
    def on_recent_file_clicked(self, item: QListWidgetItem):
        path_str = item.data(256)
        path = Path(path_str)
        
        if path.exists():
            RECORD_STATE.setRecordFile(path)
            self.add_to_recent(path)
        else:
            reply = QMessageBox.question(
                self,
                self.tr("File not found"),
                self.tr("File '{}' not found. Remove from list?").format(path.name),
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                recent = self.get_recent_files()
                if path_str in recent:
                    recent.remove(path_str)
                    self.save_recent_files(recent)
                    self.update_recent_list()
    
    @Slot()
    def clear_recent_files(self):
        reply = QMessageBox.question(
            self,
            self.tr("Clear history"),
            self.tr("Are you sure you want to clear recent files history?"),
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.save_recent_files([])
            self.update_recent_list()
    
    def load_recent_files(self):
        self.update_recent_list()
        self.check_paths_async()
    
    def onLanguageChange(self):
        self.setTabName(self.tr("Import"))
        
        self.vB.setText(self.tr("Load WAV"))
        self.vC.setText(self.tr("Clear history"))
        
        # Переводим сообщения в списке (если нужно)
        self.update_recent_list()
    
    def onThemeChange(self):
        self.update_recent_list()