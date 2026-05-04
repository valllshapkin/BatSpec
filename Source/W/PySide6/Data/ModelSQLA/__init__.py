import logging
from typing import Any, List, Optional, Type
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtWidgets import QStyledItemDelegate, QSpinBox, QDoubleSpinBox, QComboBox, QWidget
from sqlalchemy.orm import Session, DeclarativeBase
from sqlalchemy.types import Enum as SqlaEnum

logger = logging.getLogger(__name__)

def _get_python_type(column: Any) -> Type[Any]:
    try: return column.type.python_type
    except NotImplementedError: return str

class SqlAlchemyDelegate(QStyledItemDelegate):
    def __init__(self, sqla_model: Type[DeclarativeBase], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.sqla_model = sqla_model
        self.columns = list(sqla_model.__mapper__.columns)

    def createEditor(self, parent: QWidget, option: Any, index: QModelIndex) -> Optional[QWidget]:
        col = self.columns[index.column()]
        py_type = _get_python_type(col)

        if isinstance(col.type, SqlaEnum) and col.type.enum_class:
            editor = QComboBox(parent)
            editor.addItems([e.name for e in col.type.enum_class])
            return editor

        if py_type is bool: return None
        if py_type is int:
            editor = QSpinBox(parent)
            editor.setRange(-2147483648, 2147483647)
            return editor
        if py_type is float:
            editor = QDoubleSpinBox(parent)
            editor.setRange(-1e9, 1e9)
            editor.setDecimals(4)
            return editor
        return super().createEditor(parent, option, index)

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        col = self.columns[index.column()]
        if isinstance(col.type, SqlaEnum) and isinstance(editor, QComboBox):
            value = index.model().data(index, Qt.ItemDataRole.EditRole)
            if value:
                idx = editor.findText(str(value))
                if idx >= 0: editor.setCurrentIndex(idx)
            return
        super().setEditorData(editor, index)

    def setModelData(self, editor: QWidget, model: QAbstractTableModel, index: QModelIndex) -> None:
        col = self.columns[index.column()]
        if isinstance(col.type, SqlaEnum) and isinstance(editor, QComboBox):
            model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)
            return
        super().setModelData(editor, model, index)

class SqlAlchemyTableModel(QAbstractTableModel):
    def __init__(self, db_session: Session, sqla_model: Type[DeclarativeBase]):
        super().__init__()
        self.db = db_session
        self.sqla_model = sqla_model
        self.columns = list(sqla_model.__mapper__.columns)
        self.records: List[Any] = []
        self.refresh()

    def refresh(self) -> None:
        self.beginResetModel()
        try: self.records = self.db.query(self.sqla_model).all()
        except Exception as e:
            logger.error(f"Failed to fetch records: {e}")
            self.records = []
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int: return len(self.records)
    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int: return len(self.columns)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.columns[section].name
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid(): return Qt.ItemFlag.NoItemFlags
        col_name = self.columns[index.column()].name
        py_type = _get_python_type(self.columns[index.column()])
        
        if self.columns[index.column()].primary_key or "id" in col_name.lower():
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
        if py_type is bool: flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid(): return None
        record = self.records[index.row()]
        col = self.columns[index.column()]
        val = getattr(record, col.name)
        py_type = _get_python_type(col)

        if role == Qt.ItemDataRole.CheckStateRole and py_type is bool:
            return Qt.CheckState.Checked if val else Qt.CheckState.Unchecked

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if py_type is bool: return None
            if val is None: return ""
            if isinstance(col.type, SqlaEnum): return val.name if hasattr(val, "name") else str(val)
            return str(val)
        return None

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid(): return False
        record = self.records[index.row()]
        col = self.columns[index.column()]
        py_type = _get_python_type(col)

        try:
            if role == Qt.ItemDataRole.CheckStateRole and py_type is bool:
                new_val = (value == Qt.CheckState.Checked.value or value == 2)
                setattr(record, col.name, new_val)
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
                return True
                
            if role == Qt.ItemDataRole.EditRole:
                if value == "": setattr(record, col.name, None)
                elif isinstance(col.type, SqlaEnum):
                    enum_member = col.type.enum_class[value]
                    setattr(record, col.name, enum_member)
                else: setattr(record, col.name, py_type(value))
                    
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
                return True
        except (ValueError, KeyError) as e:
            logger.warning(f"Validation error: {e}")
        return False

    def insertRows(self, row: int, count: int, parent: QModelIndex = QModelIndex()) -> bool:
        self.beginInsertRows(parent, row, row + count - 1)
        try:
            for _ in range(count):
                new_record = self.sqla_model()
                for col in self.columns:
                    if col.default and col.default.is_callable:
                        try: setattr(new_record, col.name, col.default.arg({})) 
                        except Exception: pass
                self.db.add(new_record)
                self.records.insert(row, new_record)
            self.endInsertRows()
            return True
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            self.db.rollback()
            self.endInsertRows()
            return False

    def removeRows(self, row: int, count: int, parent: QModelIndex = QModelIndex()) -> bool:
        self.beginRemoveRows(parent, row, row + count - 1)
        try:
            for i in range(count):
                self.db.delete(self.records[row + i])
            del self.records[row:row+count]
            self.endRemoveRows()
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            self.db.rollback()
            self.endRemoveRows()
            return False
