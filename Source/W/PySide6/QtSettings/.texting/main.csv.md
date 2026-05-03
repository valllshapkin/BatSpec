Для сохранения положения и размеров окна в индустрии Qt **не принято** сохранять отдельно `x`, `y`, `width` и `height`. 

Для этого используют встроенные методы окна: `saveGeometry()` и `restoreGeometry()`. Они возвращают бинарный объект `QByteArray`, который автоматически учитывает на каком мониторе находилось окно, какой там был DPI, и было ли окно развернуто на весь экран (Maximized).

Для таких объектов в нашем фреймворке уже есть `QtObjAdapter`, он идеально подходит для работы с `QByteArray`.

Вот готовый пример `2.py`, который демонстрирует этот паттерн:

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/QtSettings/__test__/2.py" encoding="utf-8"
import sys
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets # type: ignore
from W.PySide6.QtSettings import Field, QtObjAdapter

ScriptDir = Path(__file__).parent


class AppConfig(QtCore.QObject):
    settings = QtCore.QSettings(str(ScriptDir / "window.ini"), QtCore.QSettings.Format.IniFormat)

    # Используем QtObjAdapter для QByteArray. 
    # QByteArray — это бинарный массив, в котором Qt безопасно хранит геометрию.
    geometry, S_geometry = Field(
        default=QtCore.QByteArray(),
        adapter=QtObjAdapter(QtCore.QByteArray, fallback=QtCore.QByteArray()),
        sig_type=QtCore.QByteArray
    )
    
    # saveState сохраняет расположение панелей инструментов (ToolBars) и док-виджетов (DockWidgets)
    state, S_state = Field(
        default=QtCore.QByteArray(),
        adapter=QtObjAdapter(QtCore.QByteArray, fallback=QtCore.QByteArray()),
        sig_type=QtCore.QByteArray
    )


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.setWindowTitle("Сохранение геометрии окна")
        
        label = QtWidgets.QLabel(
            "Измените размер окна, переместите его и закройте.\n"
            "При следующем запуске оно появится точно на том же месте!", 
            self
        )
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(label)

        # ----------------------------------------------------
        # 1. ВОССТАНОВЛЕНИЕ ПРИ ЗАПУСКЕ
        # ----------------------------------------------------
        if not self.config.geometry.isEmpty():
            self.restoreGeometry(self.config.geometry)
        else:
            # Значения по умолчанию для самого первого запуска
            self.resize(600, 400)
            
        if not self.config.state.isEmpty():
            self.restoreState(self.config.state)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # ----------------------------------------------------
        # 2. СОХРАНЕНИЕ ПЕРЕД ЗАКРЫТИЕМ
        # ----------------------------------------------------
        # Записываем текущую геометрию в дескриптор, 
        # и он автоматически сохранит её в window.ini
        self.config.geometry = self.saveGeometry()
        self.config.state = self.saveState()
        
        super().closeEvent(event)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    QtCore.QCoreApplication.setOrganizationName("MyCompany")
    QtCore.QCoreApplication.setApplicationName("MyApp")

    # Инициализируем наш конфиг
    config = AppConfig()
    
    # Создаем и показываем окно
    window = MainWindow(config)
    window.show()
    
    sys.exit(app.exec())

``````

### Как это работает:
1. Когда окно закрывается (`closeEvent`), мы вызываем `self.saveGeometry()`, который возвращает текущую позицию и размеры в виде `QByteArray`.
2. Мы присваиваем это в `self.config.geometry`.
3. Наш дескриптор передает этот массив в `QSettings`. В файле `window.ini` это сохранится в виде бинарной строки `@Variant(\0\0\0\x12...)`. В данном случае бинарный формат — это благо, потому что его читает и понимает само ядро C++ библиотеки Qt.
4. При следующем запуске мы делаем `.isEmpty()` чтобы проверить, есть ли уже сохраненные данные. Если есть — передаем в `self.restoreGeometry()`, и окно моментально принимает нужный вид (без мерцаний и скачков на экране).