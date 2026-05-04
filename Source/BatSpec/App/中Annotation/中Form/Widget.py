from PySide6.QtWidgets import QWidget, QStackedWidget, QFormLayout, QDoubleSpinBox, QComboBox, QLineEdit, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, Signal
from W.PySide6.Core.Builder import build_node as b

class ReactiveComboBox(QComboBox):
    def __init__(self, reactive_dict, label_attr: str, parent=None):
        super().__init__(parent)
        self.r_dict = reactive_dict
        self.label_attr = label_attr
        self.rebuild()
        self.r_dict.signals.changed.connect(self.rebuild)

    def rebuild(self):
        current_id = self.currentData()
        self.blockSignals(True)
        self.clear()
        self.addItem("--- Не выбрано ---", None)
        for k, v in self.r_dict.items():
            label = getattr(v, self.label_attr) or getattr(v, "name", "N/A")
            self.addItem(label, k)
        idx = self.findData(current_id)
        self.setCurrentIndex(idx if idx >= 0 else 0)
        self.blockSignals(False)


class PropertyForms(QStackedWidget):
    dataModified = Signal()
    
    def __init__(self, store):
        super().__init__()
        self.store = store
        self.lookups = store.lookups
        self.current_model = None
        self.current_typ = None
        self.setup_ui()

    def setup_ui(self):
        with b(self, QWidget()) as w_rec:
            with b(w_rec, QVBoxLayout()) as l_rec:
                with b(l_rec, QFormLayout()) as f_rec:
                    self.rec_filename = QLineEdit()
                    self.rec_detector = ReactiveComboBox(self.lookups.detectors, "name")
                    self.rec_habitat = ReactiveComboBox(self.lookups.habitats, "name")
                    self.rec_duration_s = QDoubleSpinBox(); self.rec_duration_s.setMaximum(999999)
                    self.rec_sample_rate_hz = QDoubleSpinBox(); self.rec_sample_rate_hz.setMaximum(9999999)
                    
                    f_rec.addRow("Файл:", self.rec_filename)
                    f_rec.addRow("Детектор:", self.rec_detector)
                    f_rec.addRow("Среда:", self.rec_habitat)
                    f_rec.addRow("Длительность (сек):", self.rec_duration_s)
                    f_rec.addRow("Sample Rate (Гц):", self.rec_sample_rate_hz)
                l_rec.addStretch()
        
        with b(self, QWidget()) as w_seq:
            with b(w_seq, QVBoxLayout()) as l_seq:
                with b(l_seq, QFormLayout()) as f_seq:
                    self.seq_species = ReactiveComboBox(self.lookups.species, "latin_name")
                    self.seq_context = ReactiveComboBox(self.lookups.contexts, "name")
                    self.seq_t_start = QDoubleSpinBox(); self.seq_t_start.setMaximum(999999)
                    self.seq_t_end = QDoubleSpinBox(); self.seq_t_end.setMaximum(999999)
                    self.seq_f_min = QDoubleSpinBox(); self.seq_f_min.setMaximum(200)
                    self.seq_f_max = QDoubleSpinBox(); self.seq_f_max.setMaximum(200)
                    self.seq_notes = QLineEdit()
                    
                    f_seq.addRow("Вид:", self.seq_species)
                    f_seq.addRow("Контекст:", self.seq_context)
                    f_seq.addRow("T Start:", self.seq_t_start)
                    f_seq.addRow("T End:", self.seq_t_end)
                    f_seq.addRow("F Min:", self.seq_f_min)
                    f_seq.addRow("F Max:", self.seq_f_max)
                    f_seq.addRow("Заметки:", self.seq_notes)
                l_seq.addStretch()
        
        with b(self, QWidget()) as w_call:
            with b(w_call, QVBoxLayout()) as l_call:
                with b(l_call, QFormLayout()) as f_call:
                    self.call_shape = ReactiveComboBox(self.lookups.shapes, "name")
                    self.call_t_start = QDoubleSpinBox(); self.call_t_start.setMaximum(999999)
                    self.call_t_end = QDoubleSpinBox(); self.call_t_end.setMaximum(999999)
                    self.call_f_min = QDoubleSpinBox(); self.call_f_min.setMaximum(200)
                    self.call_f_max = QDoubleSpinBox(); self.call_f_max.setMaximum(200)
                    self.call_peak_khz = QDoubleSpinBox(); self.call_peak_khz.setMaximum(200)
                    self.call_peak_ms = QDoubleSpinBox(); self.call_peak_ms.setMaximum(999999)
                    self.call_notes = QLineEdit()
                    
                    f_call.addRow("Форма:", self.call_shape)
                    f_call.addRow("T Start:", self.call_t_start)
                    f_call.addRow("T End:", self.call_t_end)
                    f_call.addRow("F Min:", self.call_f_min)
                    f_call.addRow("F Max:", self.call_f_max)
                    f_call.addRow("FmaxE:", self.call_peak_khz)
                    f_call.addRow("T FmaxE:", self.call_peak_ms)
                    f_call.addRow("Заметки:", self.call_notes)
                l_call.addStretch()
        
        with b(self, QLabel("Ничего не выбрано", alignment=Qt.AlignmentFlag.AlignCenter)):
            pass
            
        self.setCurrentIndex(3)
        self.bind_signals()

    def bind_signals(self):
        self.rec_filename.textChanged.connect(lambda v: self.update_model('filename', v))
        self.rec_detector.currentIndexChanged.connect(lambda: self.update_cb(self.rec_detector, 'detector_id'))
        self.rec_habitat.currentIndexChanged.connect(lambda: self.update_cb(self.rec_habitat, 'habitat_id'))
        self.rec_duration_s.valueChanged.connect(lambda v: self.update_model('duration_s', v))
        self.rec_sample_rate_hz.valueChanged.connect(lambda v: self.update_model('sample_rate_hz', int(v)))
        
        self.seq_species.currentIndexChanged.connect(lambda: self.update_cb(self.seq_species, 'species_id'))
        self.seq_context.currentIndexChanged.connect(lambda: self.update_cb(self.seq_context, 'context_id'))
        self.seq_t_start.valueChanged.connect(lambda v: self.update_model('t_start_ms', v))
        self.seq_t_end.valueChanged.connect(lambda v: self.update_model('t_end_ms', v))
        self.seq_f_min.valueChanged.connect(lambda v: self.update_model('f_min_khz', v))
        self.seq_f_max.valueChanged.connect(lambda v: self.update_model('f_max_khz', v))
        self.seq_notes.textChanged.connect(lambda v: self.update_model('notes', v))
        
        self.call_shape.currentIndexChanged.connect(lambda: self.update_cb(self.call_shape, 'shape_id'))
        self.call_t_start.valueChanged.connect(lambda v: self.update_model('t_start_ms', v))
        self.call_t_end.valueChanged.connect(lambda v: self.update_model('t_end_ms', v))
        self.call_f_min.valueChanged.connect(lambda v: self.update_model('f_min_khz', v))
        self.call_f_max.valueChanged.connect(lambda v: self.update_model('f_max_khz', v))
        self.call_peak_khz.valueChanged.connect(lambda v: self.update_model('peak_khz', v))
        self.call_peak_ms.valueChanged.connect(lambda v: self.update_model('peak_ms', v))
        self.call_notes.textChanged.connect(lambda v: self.update_model('notes', v))

    def update_model(self, field: str, value):
        if self.current_model:
            old_val = getattr(self.current_model, field)
            if old_val != value:
                setattr(self.current_model, field, value)
                self.dataModified.emit()

    def update_cb(self, combo: QComboBox, field: str):
        if self.current_model:
            old_val = getattr(self.current_model, field)
            new_val = combo.currentData()
            if old_val != new_val:
                setattr(self.current_model, field, new_val)
                self.dataModified.emit()

    def load_from(self, typ: str, model):
        if self.current_model and hasattr(self.current_model, 'changed'):
            try: self.current_model.changed.disconnect(self.on_model_changed)
            except Exception: pass

        self.current_model = model
        self.current_typ = typ

        if not model:
            self.setCurrentIndex(3)
            return
            
        self.current_model.changed.connect(self.on_model_changed)
        self.on_model_changed()

    def on_model_changed(self):
        model = self.current_model
        if not model: return
        
        if self.current_typ == "rec":
            self.setCurrentIndex(0)
            self.set_val(self.rec_filename, model.filename)
            self.set_cb(self.rec_detector, model.detector_id)
            self.set_cb(self.rec_habitat, model.habitat_id)
            self.set_val(self.rec_duration_s, model.duration_s or 0.0)
            self.set_val(self.rec_sample_rate_hz, model.sample_rate_hz or 0)
            
        elif self.current_typ == "seq":
            self.setCurrentIndex(1)
            self.set_cb(self.seq_species, model.species_id)
            self.set_cb(self.seq_context, model.context_id)
            self.set_val(self.seq_t_start, model.t_start_ms)
            self.set_val(self.seq_t_end, model.t_end_ms)
            self.set_val(self.seq_f_min, model.f_min_khz)
            self.set_val(self.seq_f_max, model.f_max_khz)
            self.set_val(self.seq_notes, model.notes or "")
            
        elif self.current_typ in ["call", "fmaxe", "curve"]:
            self.setCurrentIndex(2)
            self.set_cb(self.call_shape, model.shape_id)
            self.set_val(self.call_t_start, model.t_start_ms)
            self.set_val(self.call_t_end, model.t_end_ms)
            self.set_val(self.call_f_min, model.f_min_khz)
            self.set_val(self.call_f_max, model.f_max_khz)
            self.set_val(self.call_peak_khz, model.peak_khz or 0.0)
            self.set_val(self.call_peak_ms, model.peak_ms or 0.0)
            self.set_val(self.call_notes, model.notes or "")

    def set_val(self, widget, value):
        widget.blockSignals(True)
        if isinstance(widget, QDoubleSpinBox): widget.setValue(float(value))
        elif isinstance(widget, QLineEdit): widget.setText(str(value))
        widget.blockSignals(False)
        
    def set_cb(self, combo: QComboBox, data_val):
        combo.blockSignals(True)
        idx = combo.findData(data_val)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)
