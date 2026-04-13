from typing import List
from pathlib import Path

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QListWidget, QWidget, QVBoxLayout, QTabWidget, QPushButton,
                               QLabel, QSpinBox, QDoubleSpinBox, QGroupBox,
                               QFileDialog, QMessageBox, QHBoxLayout, QTextEdit)
from PySide6.QtCore import Qt, Slot

from BatSpec.QtUp.TabInst import TabInstance
from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.Component import Component
import qt_themes

# Импорты новых стейтов
from .Logic import STFT_STATE
from ..中Record.Logic import RECORD_STATE
from BatSpec.QtApp.中Project.Logic import PROJECT_STATE

class StftModelTab(*Component, TabInstance, RECORD_STATE.Trigger):

    # ==========================================
    # --- ВИДЖЕТЫ ЗАГРУЗКИ ОКНА (SOURCES) ---
    # ==========================================

    class ButtonFromProject(PROJECT_STATE.Trigger, Locales.Trigger, QPushButton):
        class LoadDialog(Locales.Trigger, QDialog):
            def __init__(self, parent: QWidget, file_paths: List[Path]):
                QDialog.__init__(self, parent)
                self.file_paths = {p.name: p for p in file_paths} # Map Name -> Path
                
                self.layout = QVBoxLayout(self)
                self.lbl_desc = QLabel()
                self.layout.addWidget(self.lbl_desc)

                self.list_names = QListWidget()
                self.list_names.addItems(list(self.file_paths.keys()))
                self.layout.addWidget(self.list_names)

                self.btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                self.btn_box.accepted.connect(self.accept)
                self.btn_box.rejected.connect(self.reject)
                self.layout.addWidget(self.btn_box)

                self.onLanguageChange()
            
            def onLanguageChange(self):
                self.setWindowTitle(self.tr("Выбор окна"))
                self.lbl_desc.setText(self.tr("Выберите окно из проекта для загрузки:"))

            def getSelectedPath(self) -> Path:
                name = self.list_names.currentItem().text()
                return self.file_paths[name]

            def runAndVerify(self) -> bool:
                return self.exec() == QDialog.Accepted and self.list_names.currentItem()

        def __init__(self):
            QPushButton.__init__(self)
            self.clicked.connect(self.onClicked)

        def onClicked(self):
            if not PROJECT_STATE.currentProject: return
            
            # Ищем файлы в папке окон проекта
            folder = PROJECT_STATE.currentProject.folder_windows
            files = list(folder.glob("*.*")) # Предполагаем, что там лежат файлы окон

            if not files: 
                QMessageBox.information(self, self.tr("Пусто"), self.tr("В проекте нет сохраненных окон."))
                return

            dialog = self.LoadDialog(self, files)
            if not dialog.runAndVerify(): return

            # Передаем Path в новый стейт
            STFT_STATE.setWindowFile(dialog.getSelectedPath())

        def onProjectNewLoaded(self):
            self.setEnabled(PROJECT_STATE.isUploaded())
            self.onLanguageChange()

        def onLanguageChange(self):
            if PROJECT_STATE.isUploaded(): 
                self.setText(self.tr("Из Проекта"))
            else: 
                self.setText(self.tr("Из Проекта (Нет проекта)"))

    class ButtonFromWinLab(Locales.Trigger, QPushButton):
        def __init__(self):
            QPushButton.__init__(self)
            self.clicked.connect(self.onClicked)
            # TODO: Подписаться на изменения стейта WinLab, чтобы активировать/деактивировать кнопку

        def onLanguageChange(self):
            self.setText(self.tr("Из WinLab"))

        def onClicked(self): 
            # ЗАГЛУШКА
            # Здесь в будущем нужно получить Path экспортированного окна из WinLab
            # path = WINDOWLAB_STATE.getCurrentExportPath()
            # if path: STFT_STATE.setWindowFile(path)
            QMessageBox.information(self, self.tr("В разработке"), self.tr("Связь с WinLab еще не реализована."))

    class ButtonImportFile(Locales.Trigger, QPushButton):
        def __init__(self):
            QPushButton.__init__(self)
            self.clicked.connect(self.onClicked)

        def onLanguageChange(self):
            self.setText(self.tr("Импорт из файла"))

        def onClicked(self):
            path, _ = QFileDialog.getOpenFileName(self, self.tr("Импорт окна"), "", "All Files (*.*)")
            if path: 
                STFT_STATE.setWindowFile(Path(path))


    # ==========================================
    # --- ВИДЖЕТЫ ПАРАМЕТРОВ STFT ---
    # ==========================================

    class OverlapParameter(Locales.Trigger, QWidget):
        def __init__(self):
            QWidget.__init__(self)
            self.layout = QHBoxLayout(self)
            self.layout.setContentsMargins(0, 0, 0, 0)

            self.label = QLabel()
            self.spin = QDoubleSpinBox()
            self.spin.setRange(0.0, 0.99)
            self.spin.setSingleStep(0.05)
            self.spin.setValue(STFT_STATE.overlap)

            self.layout.addWidget(self.label)
            self.layout.addWidget(self.spin)

            # Сигналы
            self.spin.valueChanged.connect(STFT_STATE.setOverlap)
            STFT_STATE.signals.parametersChanged.connect(self.onStateChanged)

        def onStateChanged(self):
            # Защита от зацикливания
            if self.spin.value() != STFT_STATE.overlap:
                self.spin.blockSignals(True)
                self.spin.setValue(STFT_STATE.overlap)
                self.spin.blockSignals(False)

        def onLanguageChange(self):
            self.label.setText(self.tr("Перекрытие (Overlap):"))

    class BinsParameter(Locales.Trigger, QWidget):
        def __init__(self):
            QWidget.__init__(self)
            self.layout = QHBoxLayout(self)
            self.layout.setContentsMargins(0, 0, 0, 0)

            self.label = QLabel()
            self.spin = QSpinBox()
            self.spin.setRange(10, 10000)
            self.spin.setSingleStep(10)
            self.spin.setValue(STFT_STATE.bins)

            self.layout.addWidget(self.label)
            self.layout.addWidget(self.spin)

            # Сигналы
            self.spin.valueChanged.connect(STFT_STATE.setBins)
            STFT_STATE.signals.parametersChanged.connect(self.onStateChanged)

        def onStateChanged(self):
            if self.spin.value() != STFT_STATE.bins:
                self.spin.blockSignals(True)
                self.spin.setValue(STFT_STATE.bins)
                self.spin.blockSignals(False)

        def onLanguageChange(self):
            self.label.setText(self.tr("Бин (Bins):"))


    # ==========================================
    # --- ВИДЖЕТ УПРАВЛЕНИЯ (ACTION) ---
    # ==========================================

    class CalculateAction(Locales.Trigger, RECORD_STATE.Trigger, QPushButton):
        def __init__(self):
            QPushButton.__init__(self)
            self.setMinimumHeight(50)
            self.is_calculating = False
            
            self.clicked.connect(STFT_STATE.startCalculation)

            # Подписки на стейт STFT
            STFT_STATE.signals.calculationStarted.connect(self.onCalcStarted)
            STFT_STATE.signals.calculationFinished.connect(self.onCalcFinished)
            STFT_STATE.signals.calculationError.connect(self.onCalcError)
            STFT_STATE.signals.parametersChanged.connect(self.updateState)

        def onRecordChange(self):
            self.updateState()

        def updateState(self):
            if self.is_calculating: return
            
            has_record = RECORD_STATE.record is not None
            has_window = hasattr(STFT_STATE, 'window') and STFT_STATE.window is not None
            
            self.setEnabled(has_record and has_window)
            
            if STFT_STATE.isNewInput():
                self.setText(self.tr("ПЕРЕСЧИТАТЬ (Параметры изменены)"))
            else:
                self.setText(self.tr("ПОСЧИТАТЬ СПЕКТРОГРАММУ"))

        @Slot()
        def onCalcStarted(self):
            self.is_calculating = True
            self.setEnabled(False)
            self.setText(self.tr("Идёт расчёт..."))

        @Slot()
        def onCalcFinished(self):
            self.is_calculating = False
            self.updateState()

        @Slot(str)
        def onCalcError(self, err_msg: str):
            self.is_calculating = False
            self.setText(self.tr("ОШИБКА РАСЧЁТА"))
            self.updateState() # Восстановит кнопку при следующем действии

        def onLanguageChange(self):
            self.updateState()


    # ==========================================
    # --- ОСНОВНАЯ ВКЛАДКА (СБОРКА) ---
    # ==========================================

    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None):
        TabInstance.__init__(self, parent=parent, tab_widget=tab_widget)
        self.layout_main = QVBoxLayout(self)

        # Сообщение о том, что нет записи
        self.lbl_no_record = QLabel()
        self.lbl_no_record.setAlignment(Qt.AlignCenter)
        self.layout_main.addWidget(self.lbl_no_record)

        # Главный контейнер (скрывается, если нет записи)
        self.main_container = QWidget()
        self.layout_main.addWidget(self.main_container)
        self.layout_container = QVBoxLayout(self.main_container)
        self.layout_container.setContentsMargins(0, 0, 0, 0)

        # --- 1. Группа Окна ---
        self.group_window = QGroupBox()
        self.layout_container.addWidget(self.group_window)
        self.layout_group_win = QVBoxLayout(self.group_window)

        self.lbl_current_window = QLabel()
        self.layout_group_win.addWidget(self.lbl_current_window)

        self.layout_win_buttons = QHBoxLayout()
        self.layout_group_win.addLayout(self.layout_win_buttons)

        # Вставляем модульные кнопки загрузки
        self.layout_win_buttons.addWidget(self.ButtonFromWinLab())
        self.layout_win_buttons.addWidget(self.ButtonFromProject())
        self.layout_win_buttons.addWidget(self.ButtonImportFile())

        # --- 2. Группа Параметров ---
        self.group_params = QGroupBox()
        self.layout_container.addWidget(self.group_params)
        self.layout_params = QVBoxLayout(self.group_params)

        # Вставляем модульные параметры
        self.layout_params.addWidget(self.OverlapParameter())
        self.layout_params.addWidget(self.BinsParameter())

        # --- 3. Кнопка и Логи ---
        self.layout_container.addWidget(self.CalculateAction())

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.layout_container.addWidget(self.log_output)

        # --- ПОДПИСКИ ТАБЫ ---
        STFT_STATE.signals.parametersChanged.connect(self.update_info_labels)
        STFT_STATE.signals.calculationStarted.connect(lambda: self.log_output.append(self.tr("Начало расчёта...")))
        STFT_STATE.signals.calculationFinished.connect(lambda: self.log_output.append(self.tr("Расчёт успешно завершён.")))
        STFT_STATE.signals.calculationError.connect(self.on_calc_error)

        # Инициализация
        self.update_info_labels()
        self.onLanguageChange()

    def onRecordChange(self):
        """Реализация RECORD_STATE.Trigger для главной табы"""
        has_record = RECORD_STATE.record is not None
        self.lbl_no_record.setVisible(not has_record)
        self.main_container.setEnabled(has_record)
        self.update_info_labels()

    @Slot()
    def update_info_labels(self):
        # Обновляем текст с текущим окном
        win_name = getattr(STFT_STATE, 'window_name', None)
        if win_name:
            self.lbl_current_window.setText(self.tr("Текущее окно:") + f" {win_name}")
        else:
            self.lbl_current_window.setText(self.tr("Текущее окно: Не выбрано"))

    @Slot(str)
    def on_calc_error(self, err_msg: str):
        self.log_output.append(self.tr("Ошибка STFT:\n") + err_msg)

    def onLanguageChange(self):
        """Локализация только тех элементов, которые принадлежат самой табе"""
        self.setTabName(self.tr("STFT Model"))
        self.lbl_no_record.setText(self.tr("Запись не выбрана. Загрузите аудио файл."))
        self.group_window.setTitle(self.tr("Окно (Window)"))
        self.group_params.setTitle(self.tr("Параметры STFT"))
        self.update_info_labels()

    def onThemeChange(self):
        theme = qt_themes.get_theme()
        self.lbl_no_record.setStyleSheet(f"color: {theme.red.name()}; font-weight: bold;")
        self.log_output.setStyleSheet(f"color: {theme.red.name()}")