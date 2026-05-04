from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QTreeWidgetItemIterator
from PySide6.QtCore import Qt
from BatAnnotation.QtModels import QtRecording, QtSequence, QtBatCall

class ReactiveTreeItem(QTreeWidgetItem):
    def __init__(self, parent, typ: str, model, lookups):
        super().__init__(parent)
        self.typ = typ
        self.model = model
        self.lookups = lookups
        
        self.setData(0, Qt.ItemDataRole.UserRole, (self.typ, self.model))
        self.model.changed.connect(self.update_text)
        
        if self.typ == "seq":
            self.lookups.species.signals.changed.connect(self.update_text)
            
        self.update_text()

    def update_text(self):
        if self.typ == "rec":
            self.setText(0, f"🎵 {self.model.filename}")
            
        elif self.typ == "seq":
            sp = self.lookups.species.get(self.model.species_id)
            sp_name = sp.latin_name if sp else "Неизвестно"
            self.setText(0, f"🦇 Секвенция: {self.model.t_start_ms:.0f}-{self.model.t_end_ms:.0f} [{sp_name}]")
            
        elif self.typ == "call":
            self.setText(0, f"🔊 Писк: {self.model.t_start_ms:.0f}-{self.model.t_end_ms:.0f}")
            
        elif self.typ == "fmaxe":
            val = f"{self.model.peak_khz:.1f}kHz" if self.model.peak_khz else "Нет"
            self.setText(0, f"🎯 fmaxe: {val}")
            
        elif self.typ == "curve":
            pts = len(self.model.signal_curves.get("main", [])) if self.model.signal_curves else 0
            self.setText(0, f"〰️ Кривая ({pts} узлов)")

    def cleanup(self):
        try: self.model.changed.disconnect(self.update_text)
        except Exception: pass
        
        if self.typ == "seq":
            try: self.lookups.species.signals.changed.disconnect(self.update_text)
            except Exception: pass


class AnnotationTreeWidget(QTreeWidget):
    def __init__(self, store):
        super().__init__()
        self.store = store
        self.lookups = store.lookups
        self.setHeaderHidden(True)
        
        self.store.recording.changed.connect(self._rebuild_if_new_recording)
        self._current_recording_id = None

    def _rebuild_if_new_recording(self):
        rec = self.store.recording
        if rec.recording_id != self._current_recording_id:
            self.build_from(rec)

    def build_from(self, recording: QtRecording):
        self._cleanup_all_items()
        self.clear()
        
        self._current_recording_id = recording.recording_id
        rec_item = ReactiveTreeItem(self, "rec", recording, self.lookups)
        
        recording.sequences.signals.itemAdded.connect(lambda idx, seq: self._add_sequence_node(rec_item, seq))
        recording.sequences.signals.itemRemoved.connect(lambda idx, seq: self._remove_node_by_model(seq))
        
        for seq in recording.sequences:
            self._add_sequence_node(rec_item, seq)
            
        self.expandAll()

    def _add_sequence_node(self, parent_item, seq: QtSequence):
        seq_item = ReactiveTreeItem(parent_item, "seq", seq, self.lookups)
        
        seq.calls.signals.itemAdded.connect(lambda idx, call: self._add_call_node(seq_item, call))
        seq.calls.signals.itemRemoved.connect(lambda idx, call: self._remove_node_by_model(call))
        
        for call in seq.calls:
            self._add_call_node(seq_item, call)

    def _add_call_node(self, parent_item, call: QtBatCall):
        call_item = ReactiveTreeItem(parent_item, "call", call, self.lookups)
        ReactiveTreeItem(call_item, "fmaxe", call, self.lookups)
        ReactiveTreeItem(call_item, "curve", call, self.lookups)
        parent_item.setExpanded(True)

    def _remove_node_by_model(self, model):
        iterator = QTreeWidgetItemIterator(self)
        while iterator.value():
            item = iterator.value()
            typ, item_model = item.data(0, Qt.ItemDataRole.UserRole)
            if item_model == model and typ in ["seq", "call"]:
                for i in range(item.childCount()):
                    child = item.child(i)
                    if isinstance(child, ReactiveTreeItem):
                        child.cleanup()
                        
                if isinstance(item, ReactiveTreeItem):
                    item.cleanup()
                item.parent().removeChild(item)
                return
            iterator += 1

    def _cleanup_all_items(self):
        iterator = QTreeWidgetItemIterator(self)
        while iterator.value():
            item = iterator.value()
            if isinstance(item, ReactiveTreeItem):
                item.cleanup()
            iterator += 1
