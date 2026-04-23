from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QDoubleSpinBox,
    QPushButton, QTextEdit, QGroupBox, QFormLayout, QTabWidget,
    QComboBox, QStackedWidget
)
from PySide6.QtCore import Qt, Slot

from BatSpec.QtUp.TabInst import TabInstance

from BatSpec.QtApp.Services.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Services.中ThemeSettings.Logic import Themes
import qt_themes

from .Logic import PROCESSING_STATE, ProcessingResult, PipelineMode

class ProcessingTab(Locales.TranslateComponent, Locales.Trigger, Themes.Trigger, TabInstance):
    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None):
        TabInstance.__init__(self, parent=parent, tab_widget=tab_widget)

        self.layout_main = QVBoxLayout(self)

        self.lbl_no_spec = QLabel()
        self.lbl_no_spec.setAlignment(Qt.AlignCenter)
        self.layout_main.addWidget(self.lbl_no_spec)

        self.main_container = QWidget()
        self.layout_main.addWidget(self.main_container)
        self.layout_container = QVBoxLayout(self.main_container)
        self.layout_container.setContentsMargins(0, 0, 0, 0)

        # ── Выбор пайплайна ────────────────────────────────────────────────
        self.group_pipeline = QGroupBox()
        self.layout_container.addWidget(self.group_pipeline)
        form_pipe = QFormLayout(self.group_pipeline)
        
        self.combo_pipeline = QComboBox()
        for mode in PipelineMode:
            self.combo_pipeline.addItem(mode.value, mode)
        self.combo_pipeline.setCurrentIndex(list(PipelineMode).index(PROCESSING_STATE.config['pipeline_mode']))
        form_pipe.addRow("Auto-Detection Mode:", self.combo_pipeline)

        # ── Стек параметров для разных пайплайнов ─────────────────────────
        self.stack_params = QStackedWidget()
        self.layout_container.addWidget(self.stack_params)

        # Страница 0: Skip
        page_skip = QWidget()
        self.stack_params.addWidget(page_skip)

        # Страница 1: Smoothness
        page_smooth = QWidget()
        form_smooth = QFormLayout(page_smooth)
        self.spin_percentile = QDoubleSpinBox()
        self.spin_percentile.setRange(1.0, 99.9)
        self.spin_percentile.setValue(PROCESSING_STATE.config['percentile'])
        form_smooth.addRow("Noise Percentile (%):", self.spin_percentile)
        self.stack_params.addWidget(page_smooth)

        # Страница 2: Echo
        page_echo = QWidget()
        form_echo = QFormLayout(page_echo)
        self.spin_echo_decay = QDoubleSpinBox()
        self.spin_echo_decay.setRange(0.0, 500.0); self.spin_echo_decay.setValue(PROCESSING_STATE.config['echo_decay'])
        self.spin_echo_n = QDoubleSpinBox()
        self.spin_echo_n.setRange(0.1, 5.0); self.spin_echo_n.setSingleStep(0.1); self.spin_echo_n.setValue(PROCESSING_STATE.config['echo_n'])
        self.spin_echo_beta = QDoubleSpinBox()
        self.spin_echo_beta.setRange(0.0, 50.0); self.spin_echo_beta.setValue(PROCESSING_STATE.config['echo_beta'])
        self.spin_echo_eps = QDoubleSpinBox()
        self.spin_echo_eps.setRange(0.0001, 1.0); self.spin_echo_eps.setDecimals(4); self.spin_echo_eps.setSingleStep(0.01); self.spin_echo_eps.setValue(PROCESSING_STATE.config['echo_eps'])
        
        form_echo.addRow("Decay Base (on 40kHz):", self.spin_echo_decay)
        form_echo.addRow("Freq Exponent (n):", self.spin_echo_n)
        form_echo.addRow("Time Power (beta):", self.spin_echo_beta)
        form_echo.addRow("Wiener Eps Factor:", self.spin_echo_eps)
        self.stack_params.addWidget(page_echo)

        # Переключение страниц
        self.combo_pipeline.currentIndexChanged.connect(self.stack_params.setCurrentIndex)
        self.stack_params.setCurrentIndex(self.combo_pipeline.currentIndex())

        # ── Кнопка ────────────────────────────────────────────────────────
        self.btn_recalculate = QPushButton()
        self.btn_recalculate.setMinimumHeight(40)
        self.btn_recalculate.setEnabled(False) 
        self.layout_container.addWidget(self.btn_recalculate)

        # ── Info & Log ────────────────────────────────────────────────────
        self.group_info = QGroupBox()
        self.layout_container.addWidget(self.group_info)
        self.form_info = QFormLayout(self.group_info)
        
        self.lbl_found_rois = QLabel("—")
        self.form_info.addRow("Detected ROIs:", self.lbl_found_rois)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.layout_container.addWidget(self.log_output)

        # ── Connections ───────────────────────────────────────────────────
        self.combo_pipeline.currentIndexChanged.connect(lambda idx: PROCESSING_STATE.set_param('pipeline_mode', self.combo_pipeline.itemData(idx)))
        self.spin_percentile.valueChanged.connect(lambda v: PROCESSING_STATE.set_param('percentile', v))
        self.spin_echo_decay.valueChanged.connect(lambda v: PROCESSING_STATE.set_param('echo_decay', v))
        self.spin_echo_n.valueChanged.connect(lambda v: PROCESSING_STATE.set_param('echo_n', v))
        self.spin_echo_beta.valueChanged.connect(lambda v: PROCESSING_STATE.set_param('echo_beta', v))
        self.spin_echo_eps.valueChanged.connect(lambda v: PROCESSING_STATE.set_param('echo_eps', v))

        self.btn_recalculate.clicked.connect(PROCESSING_STATE.recalculate)

        PROCESSING_STATE.readyToCalculate.connect(self._on_ready_to_calc)
        PROCESSING_STATE.calculationStarted.connect(self._on_calc_started)
        PROCESSING_STATE.calculationFinished.connect(self._on_calc_finished)
        PROCESSING_STATE.calculationError.connect(self._on_calc_error)

        self.onLanguageChange()

    @Slot()
    def _on_ready_to_calc(self):
        self.lbl_no_spec.setVisible(False)
        self.btn_recalculate.setEnabled(True)

    @Slot()
    def _on_calc_started(self):
        self.btn_recalculate.setEnabled(False)
        self.btn_recalculate.setText(self.tr("Processing..."))

    @Slot(object)
    def _on_calc_finished(self, result: ProcessingResult):
        self.btn_recalculate.setEnabled(True)
        self.btn_recalculate.setText(self.tr("Recalculate Bounds"))
        if result.is_manual:
            self.lbl_found_rois.setText(str(len(result.roi_boxes)))
            self.log_output.append(self.tr(f"Detection done. Found {len(result.roi_boxes)} ROIs."))

    @Slot(str)
    def _on_calc_error(self, err_msg: str):
        self.btn_recalculate.setEnabled(True)
        self.btn_recalculate.setText(self.tr("ERROR"))
        self.log_output.append(self.tr("Error:\n") + err_msg)

    def onLanguageChange(self):
        self.setTabName(self.tr("Detector Setup"))
        self.lbl_no_spec.setText(self.tr("Wait for STFT calculation..."))
        self.group_pipeline.setTitle(self.tr("Pipeline Selection"))
        self.group_info.setTitle(self.tr("Results"))
        if self.btn_recalculate.isEnabled():
            self.btn_recalculate.setText(self.tr("Calculate Bounds"))

    def onThemeChange(self):
        theme = qt_themes.get_theme()
        self.lbl_no_spec.setStyleSheet(f"color: {theme.red.name()}; font-weight: bold;")