from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QSplitter, QPushButton, 
    QMessageBox, QTreeWidgetItemIterator, QComboBox, QWidget, QTabWidget
)
from PySide6.QtCore import Qt

from W.PySide6.Core.Builder import build_node as b
from W.PySide6.Core.Lifecycle import ComponentLifecycle
from W.PySide6.Widgets.TabInstance import TabInstance

from BatAnnotation.QtModels import QtSequence, QtBatCall
from BatAnnotation.Tables import Recording
from BatSpec.App.中Annotation.Logic import ANNOTATION_STORE
from BatSpec.App.中Annotation.中Tree.Widget import AnnotationTreeWidget
from BatSpec.App.中Annotation.中Form.Widget import PropertyForms
from BatSpec.App.中Annotation.中Spectrogram.Widget import SpectrogramWidget

class AnnotationTab(ComponentLifecycle, TabInstance):
    """Вкладка разметки: Дерево + Спектрограмма + Формы."""
    
    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        super().__init__(parent, tab_widget)

    def __init_state__(self):
        self.store = ANNOTATION_STORE
        self._prev_cb_index = -1

    def __init_graph__(self):
        with b(self, QVBoxLayout()) as self.main_layout:
            with b(self.main_layout, QHBoxLayout()) as self.toolbar_top:
                with b(self.toolbar_top, QComboBox()) as self.cb_recording: pass
                self.toolbar_top.addStretch()

            with b(self.main_layout, QHBoxLayout()) as self.toolbar:
                with b(self.toolbar, QPushButton("➕ Секвенция (Контекст)")) as self.btn_add_seq: pass
                with b(self.toolbar, QPushButton("➕ Писк (Call)")) as self.btn_add_call: pass
                with b(self.toolbar, QPushButton("❌ Удалить")) as self.btn_del: pass
                self.toolbar.addStretch()
                with b(self.toolbar, QPushButton("💾 Сохранить в БД")) as self.btn_save: pass

            with b(self.main_layout, QSplitter(Qt.Orientation.Horizontal)) as self.splitter:
                with b(self.splitter, AnnotationTreeWidget(self.store)) as self.tree: pass
                with b(self.splitter, SpectrogramWidget(self.store.recording)) as self.plot: pass
                with b(self.splitter, PropertyForms(self.store)) as self.forms: pass
                
                self.splitter.setSizes([250, 700, 250])

    def __init_signal__(self):
        self.cb_recording.currentIndexChanged.connect(self.on_recording_changed)
        self.btn_add_seq.clicked.connect(self.add_sequence)
        self.btn_add_call.clicked.connect(self.add_call)
        self.btn_del.clicked.connect(self.delete_selected)
        self.btn_save.clicked.connect(lambda: self.save_data(show_info=True))
        
        self.tree.itemSelectionChanged.connect(self.on_tree_selection)
        self.plot.itemClicked.connect(self.on_plot_click)
        self.plot.dataModified.connect(self.mark_dirty)
        self.forms.dataModified.connect(self.mark_dirty)

    def __init_ready__(self):
        self.load_recording_list()
        if self.store.recording.recording_id:
            self.tree.build_from(self.store.recording)
            self.plot.set_selection("rec", self.store.recording)

    def mark_dirty(self):
        self.store.is_dirty = True

    def load_recording_list(self):
        self.cb_recording.blockSignals(True)
        self.cb_recording.clear()
        
        recs = self.store.db.query(Recording.recording_id, Recording.filename).all()
        for rec_id, fname in recs:
            self.cb_recording.addItem(fname, rec_id)
            
        idx = self.cb_recording.findData(self.store.recording.recording_id)
        if idx >= 0:
            self.cb_recording.setCurrentIndex(idx)
            self._prev_cb_index = idx
            
        self.cb_recording.blockSignals(False)

    def on_recording_changed(self, index: int):
        if self.store.is_dirty:
            reply = QMessageBox.question(
                self, "Несохраненные изменения",
                "У вас есть несохраненные изменения.\nСохранить их перед переключением?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            if reply == QMessageBox.StandardButton.Cancel:
                self.cb_recording.blockSignals(True)
                self.cb_recording.setCurrentIndex(self._prev_cb_index)
                self.cb_recording.blockSignals(False)
                return
            elif reply == QMessageBox.StandardButton.Yes:
                if not self.save_data(show_info=False):
                    self.cb_recording.blockSignals(True)
                    self.cb_recording.setCurrentIndex(self._prev_cb_index)
                    self.cb_recording.blockSignals(False)
                    return

        rec_id = self.cb_recording.itemData(index)
        if rec_id:
            self.store.load_recording(rec_id)
            self._prev_cb_index = index
            self.tree.clearSelection()
            self.plot.set_selection("rec", self.store.recording)

    def on_tree_selection(self):
        items = self.tree.selectedItems()
        if not items:
            self.forms.load_from("", None)
            self.plot.set_selection("rec", self.store.recording)
            return
            
        typ, model = items[0].data(0, Qt.ItemDataRole.UserRole)
        self.forms.load_from(typ, model)
        self.plot.set_selection(typ, model)

    def on_plot_click(self, typ, model):
        if not typ or not model or typ == "rec":
            self.tree.clearSelection()
            return

        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            i_typ, i_model = item.data(0, Qt.ItemDataRole.UserRole)
            if i_model == model and i_typ == typ:
                self.tree.setCurrentItem(item)
                return
            iterator += 1

    def add_sequence(self):
        vr = self.plot.getViewBox().viewRect()
        seq = QtSequence(self.store.recording) 
        seq.t_start_ms = vr.center().x() - 100
        seq.t_end_ms = vr.center().x() + 100
        seq.f_min_khz = 20
        seq.f_max_khz = 60
        self.store.recording.sequences.append(seq)
        self.mark_dirty()

    def add_call(self):
        items = self.tree.selectedItems()
        if not items:
            return QMessageBox.warning(self, "Внимание", "Выберите Sequence, куда добавить Call.")
            
        typ, model = items[0].data(0, Qt.ItemDataRole.UserRole)
        seq = model if typ == "seq" else model.parent()
        
        if not seq or not isinstance(seq, QtSequence):
            return QMessageBox.warning(self, "Внимание", "Сначала выделите Sequence в дереве.")
            
        vr = self.plot.getViewBox().viewRect()
        call = QtBatCall(seq) 
        call.t_start_ms = vr.center().x() - 10
        call.t_end_ms = vr.center().x() + 10
        call.f_min_khz = seq.f_min_khz + 5
        call.f_max_khz = seq.f_max_khz - 5
        
        call.peak_khz = call.f_min_khz + (call.f_max_khz - call.f_min_khz)/2
        call.peak_ms = call.t_start_ms + (call.t_end_ms - call.t_start_ms)/2
        call.signal_curves = {"main": [[call.t_start_ms, call.f_max_khz], [call.t_end_ms, call.f_min_khz]]}
        
        seq.calls.append(call)
        self.mark_dirty()

    def delete_selected(self):
        items = self.tree.selectedItems()
        if not items: return
        typ, model = items[0].data(0, Qt.ItemDataRole.UserRole)
        
        if typ == "seq":
            self.store.recording.sequences.remove(model)
        elif typ == "call":
            seq = model.parent()
            if seq and model in seq.calls:
                seq.calls.remove(model)
            else:
                for s in self.store.recording.sequences:
                    if model in s.calls:
                        s.calls.remove(model)
                        break
        elif typ in ["fmaxe", "curve"]:
            if typ == "fmaxe":
                model.peak_khz = 0.0
                model.peak_ms = 0.0
            else:
                model.signal_curves = None
            model.changed.emit()
            
        self.mark_dirty()

    def save_data(self, show_info=True) -> bool:
        try:
            self.store.save_recording_to_db()
            if show_info:
                QMessageBox.information(self, "Успех", "Данные сохранены!")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {e}")
            return False

    def onLanguageChange(self):
        if self.tab_widget:
            self.setTabName(self.tr("Annotation Editor"))
        super().onLanguageChange()
