from typing import Type
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, 
    QTableView, QLabel, QMessageBox, QHeaderView, QAbstractItemView, QWidget, QTabWidget
)
from sqlalchemy.orm import DeclarativeBase

from W.PySide6.Core.Builder import build_node as b
from W.PySide6.Data.ModelSQLA import SqlAlchemyTableModel, SqlAlchemyDelegate
from W.PySide6.Core.Lifecycle import ComponentLifecycle
from W.PySide6.Widgets.TabInstance import TabInstance

from BatAnnotation.Lookup import Species, DetectorModel, HabitatType, ContextType, SignalShape
from BatSpec.App.中Annotation.Logic import ANNOTATION_STORE

class LookupsTab(ComponentLifecycle, TabInstance):
    """Вкладка для редактирования справочников БД напрямую через SQLAlchemy Models"""

    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        super().__init__(parent, tab_widget)

    def __init_state__(self):
        self.store = ANNOTATION_STORE
        self.db = self.store.db
        self.table_model = None
    
    def __init_graph__(self):
        with b(self, QVBoxLayout()) as self.main_layout:
            with b(self.main_layout, QHBoxLayout()) as self.top_layout:
                with b(self.top_layout, QLabel("Справочник:")): pass 
                
                with b(self.top_layout, QComboBox()) as self.cb_tables:
                    self.cb_tables.addItem("Виды (Species)", Species)
                    self.cb_tables.addItem("Детекторы (Detectors)", DetectorModel)
                    self.cb_tables.addItem("Местообитания (Habitats)", HabitatType)
                    self.cb_tables.addItem("Контексты (Contexts)", ContextType)
                    self.cb_tables.addItem("Формы (Shapes)", SignalShape)
                
                with b(self.top_layout, QPushButton("➕ Добавить")) as self.btn_add: pass
                with b(self.top_layout, QPushButton("❌ Удалить")) as self.btn_del: pass
                self.top_layout.addStretch()
                with b(self.top_layout, QPushButton("💾 Сохранить изменения в БД")) as self.btn_save: pass

            with b(self.main_layout, QTableView()) as self.table:
                self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
                self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

    def __init_signal__(self):
        self.cb_tables.currentIndexChanged.connect(self.load_table)
        self.btn_add.clicked.connect(self.add_row)
        self.btn_del.clicked.connect(self.delete_row)
        self.btn_save.clicked.connect(self.save_table)

    def __init_ready__(self):
        if self.cb_tables.count() > 0:
            self.load_table()

    def load_table(self):
        current_sqla_model: Type[DeclarativeBase] = self.cb_tables.currentData()
        if not current_sqla_model:
            return

        self.table_model = SqlAlchemyTableModel(self.db, current_sqla_model)
        self.table.setModel(self.table_model)
        
        delegate = SqlAlchemyDelegate(current_sqla_model, self.table)
        self.table.setItemDelegate(delegate)

    def add_row(self):
        if not self.table_model: return
        row = self.table_model.rowCount()
        if self.table_model.insertRows(row, 1):
            self.table.selectRow(row)

    def delete_row(self):
        if not self.table_model: return
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            QMessageBox.warning(self, "Внимание", "Выберите строку для удаления.")
            return
        for index in sorted(selected, key=lambda x: x.row(), reverse=True):
            self.table_model.removeRows(index.row(), 1)

    def save_table(self) -> bool:
        try:
            self.db.commit()
            self.store.sync_lookups_from_db() 
            QMessageBox.information(self, "Успех", "Справочники обновлены!")
            self.table_model.refresh()
            return True
        except Exception as e:
            self.db.rollback()
            QMessageBox.critical(self, "Ошибка БД", f"Не удалось сохранить.\n\nПодробности:\n{e}")
            return False

    def onLanguageChange(self):
        if self.tab_widget:
            self.setTabName(self.tr("Lookups (DB)"))
        super().onLanguageChange()
