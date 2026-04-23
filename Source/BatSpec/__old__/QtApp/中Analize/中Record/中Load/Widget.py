from pathlib import Path
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QStackedWidget,
                               QLabel, QListWidget, QListWidgetItem, QPushButton)
from PySide6.QtCore import Slot, Qt
from PySide6.QtGui import QIcon


from BatSpec.QtUp.TabInst import TabInstance

from BatSpec.QtApp.Shared.Component import Component, GLOBAL_MARGINS

from ..Logic import RECORD_STATE
from BatSpec.QtApp.中Project.Logic import PROJECT_STATE

class Load(*Component, PROJECT_STATE.Trigger, TabInstance):
    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        TabInstance.__init__(self, parent, tab_widget)
        
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(*GLOBAL_MARGINS)
        
        # QStackedWidget для переключения между "Заглушкой" и "Списком"
        self.vS = QStackedWidget()
        self.v.addWidget(self.vS)
        
        # --- СТРАНИЦА 0: Заглушка ---
        self.lbl_no_project = QLabel()
        self.lbl_no_project.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vS.addWidget(self.lbl_no_project)
        
        # --- СТРАНИЦА 1: Рабочая область со списком ---
        self.page_list = QWidget()
        self.page_list_layout = QVBoxLayout(self.page_list)
        self.page_list_layout.setContentsMargins(0, 0, 0, 0)
        
        self.list_records = QListWidget()
        self.list_records.itemDoubleClicked.connect(self.on_item_selected)
        self.page_list_layout.addWidget(self.list_records)
        
        self.btn_refresh = QPushButton()
        self.btn_refresh.clicked.connect(self.refresh_list)
        self.page_list_layout.addWidget(self.btn_refresh)
        
        self.vS.addWidget(self.page_list)

        # Инициализация
        self.update_ui_state()

    def onProjectNewLoaded(self):
        self.update_ui_state()

    @Slot()
    def update_ui_state(self):
        """Переключает экраны в зависимости от того, загружен ли проект"""
        if PROJECT_STATE.currentProject is None:
            self.vS.setCurrentIndex(0)
            self.list_records.clear()
        else:
            self.vS.setCurrentIndex(1)
            self.refresh_list()

    @Slot()
    def refresh_list(self):
        """Обновляет список записей из папки проекта"""
        self.list_records.clear()
        
        proj = PROJECT_STATE.currentProject
        if not proj:
            return
            
        records_dir = proj.folder_records
        
        # Защита: если папки еще нет, просто выходим
        if not records_dir.exists():
            return
            
        # Сканируем папку. Ищем структуру: folder_records / {name} / record.wav
        for item_dir in records_dir.iterdir():
            if item_dir.is_dir():
                wav_path = item_dir / "record.wav"
                
                if wav_path.exists() and wav_path.is_file():
                    # Создаем элемент списка. Имя - это название папки {name}
                    item = QListWidgetItem(item_dir.name)
                    item.setIcon(QIcon.fromTheme("audio-x-generic"))
                    
                    # Сохраняем полный путь к WAV файлу в data (роль 256, как в Import)
                    item.setData(256, str(wav_path.absolute()))
                    
                    self.list_records.addItem(item)

    @Slot(QListWidgetItem)
    def on_item_selected(self, item: QListWidgetItem):
        """Слот срабатывает при двойном клике по записи"""
        path_str = item.data(256)
        if path_str:
            wav_path = Path(path_str)
            if wav_path.exists():
                RECORD_STATE.setRecordFile(wav_path)
    
    def onLanguageChange(self):
        self.setTabName(self.tr("Load"))
        self.lbl_no_project.setText(self.tr("Проект не загружен"))
        self.btn_refresh.setText(self.tr("Обновить список"))
    
    def onThemeChange(self):
        # При смене темы можно перерисовать иконки (если они зависят от темы)
        pass