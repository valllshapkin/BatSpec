import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLineEdit, QPushButton, QTreeView,
                               QFileDialog, QFileSystemModel)
from PySide6.QtCore import QDir

class FolderBrowser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Просмотрщик папок")
        self.resize(600, 400)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной вертикальный layout
        main_layout = QVBoxLayout(central_widget)

        # Горизонтальный layout для выбора папки
        top_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Путь к папке...")
        self.browse_button = QPushButton("Обзор")
        self.browse_button.clicked.connect(self.select_folder)

        top_layout.addWidget(self.path_edit)
        top_layout.addWidget(self.browse_button)

        # Дерево для отображения содержимого
        self.tree_view = QTreeView()
        # Создаём модель файловой системы
        self.model = QFileSystemModel()
        # Устанавливаем фильтр: показывать все файлы и папки, не показывать системные
        self.model.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot)
        # Не показываем столбцы размера, типа и даты изменения (оставляем только имя)
        self.tree_view.setModel(self.model)
        self.tree_view.setColumnHidden(1, True)  # размер
        self.tree_view.setColumnHidden(2, True)  # тип
        self.tree_view.setColumnHidden(3, True)  # дата изменения
        # При изменении пути в поле редактирования обновляем корень модели
        self.path_edit.textChanged.connect(self.set_root_path)

        # Добавляем виджеты в основной layout
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.tree_view)

        # По умолчанию показываем текущую рабочую папку
        self.path_edit.setText(QDir.currentPath())

    def select_folder(self):
        """Открывает диалог выбора папки и устанавливает путь в поле ввода."""
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку", self.path_edit.text()
        )
        if folder:
            self.path_edit.setText(folder)

    def set_root_path(self, path):
        """Устанавливает корневой путь модели для отображения в дереве."""
        if path and QDir(path).exists():
            index = self.model.setRootPath(path)
            self.tree_view.setRootIndex(index)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FolderBrowser()
    window.show()
    sys.exit(app.exec())