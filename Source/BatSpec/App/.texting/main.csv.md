Ты абсолютно прав насчет MRO. Твой предыдущий код *должен* был работать, и во многих случаях Python успешно "схлопывает" такую иерархию. Проблема MRO в связке с PySide/PyQt иногда бывает сложнее из-за того, как C++ классы и Python классы взаимодействуют. Иногда порядок, который кажется логичным для Python, конфликтует с внутренней иерархией Qt. Так что твое решение убрать дублирование — самый надежный и правильный путь.

А теперь к самому интересному! Создание кастомной "шапки" (title bar) — это отличная идея для придания приложению уникального вида.

### Концепция

Мы сделаем следующее:

1.  **Уберем стандартную рамку окна:** С помощью специального флага `Qt.WindowType.FramelessWindowHint` мы скажем операционной системе не рисовать стандартный заголовок и границы.
2.  **Создадим свой виджет-заголовок:** Это будет обычный `QWidget` с `QHBoxLayout`, в котором мы разместим иконку, название, меню (`QMenuBar`), "пружину" для отступа и наши собственные кнопки управления окном.
3.  **Реализуем перетаскивание:** Поскольку стандартного заголовка нет, нам нужно будет самим реализовать логику перетаскивания окна с помощью событий мыши (`mousePressEvent`, `mouseMoveEvent`).
4.  **Реализуем изменение размера:** У frameless-окна нет границ для ресайза. Мы добавим специальный виджет `QSizeGrip` в правый нижний угол.
5.  **Интегрируем это в архитектуру:** Чтобы не замусоривать `MainWindow` или `AddMenu`, мы создадим новый базовый класс `FramelessWindow`, который будет содержать всю эту логику.

Давай создадим новый модуль для этого функционала и обновим существующие классы.

---

### 1. Новый модуль `FramelessWindow`

Этот модуль будет содержать всю логику для создания окна без рамки.

``````python path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/QtUp/FramelessWindow.py" encoding="utf-8"
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
    QPushButton, QLabel, QSizeGrip, QMenuBar
)
from PySide6.QtCore import Qt, QPoint, QEvent
from PySide6.QtGui import QMouseEvent

from BatSpec.QtUp.Builder import build_node as b
from BatSpec.QtUp.Locales import Locales
from BatSpec.QtUp.Themes import Themes

class CustomTitleBar(Locales.TranslateComponent, Locales.Trigger, Themes.Trigger, QWidget):
    def __init__(self, parent_window: QMainWindow):
        super().__init__(parent_window)
        self.window = parent_window
        self.setFixedHeight(32) # Стандартная высота

        with b(self, QHBoxLayout()) as layout:
            layout.setContentsMargins(5, 0, 5, 0)
            layout.setSpacing(10)

            # Иконка и заголовок
            with b(layout, QLabel("Ico")) as self.icon_label: pass # Заглушка для иконки
            with b(layout, QLabel("Window Title")) as self.title_label: pass
            
            # Меню (будет добавлено из AddMenu)
            with b(layout, QMenuBar()) as self.menu_bar:
                layout.setStretchFactor(self.menu_bar, 1)

            # Кнопки управления окном
            with b(layout, QPushButton("_")) as self.btn_minimize:
                self.btn_minimize.setFixedSize(28, 28)
                self.btn_minimize.clicked.connect(self.window.showMinimized)

            with b(layout, QPushButton("[]")) as self.btn_maximize:
                self.btn_maximize.setFixedSize(28, 28)
                self.btn_maximize.clicked.connect(self.toggle_maximize)

            with b(layout, QPushButton("X")) as self.btn_close:
                self.btn_close.setFixedSize(28, 28)
                self.btn_close.clicked.connect(self.window.close)
                # Добавим стиль, чтобы было похоже на кнопку закрытия
                self.btn_close.setStyleSheet("background-color: #c42b1c; color: white;")
    
    def onLanguageChange(self):
        # Обновляем заголовок окна при смене языка
        self.title_label.setText(self.window.windowTitle())

    def toggle_maximize(self):
        if self.window.isMaximized():
            self.window.showNormal()
            self.btn_maximize.setText("[]")
        else:
            self.window.showMaximized()
            self.btn_maximize.setText("[=]")
            
    def update_window_title(self, title: str):
        self.title_label.setText(title)


class FramelessWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.old_pos = QPoint()

        # Центральная структура
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        with b(self.central_widget, QVBoxLayout()) as self.main_layout:
            self.main_layout.setContentsMargins(1, 1, 1, 1)
            self.main_layout.setSpacing(0)

            # 1. Наш кастомный Title Bar
            with b(self.main_layout, CustomTitleBar(self)) as self.title_bar: pass

            # 2. Контейнер для основного содержимого (меню, табов и т.д.)
            with b(self.main_layout, QWidget()) as self.main_content:
                self.main_content_layout = QVBoxLayout(self.main_content)
                self.main_content_layout.setContentsMargins(0, 0, 0, 0)

            # 3. Ручка для изменения размера в углу
            self.size_grip = QSizeGrip(self)
            self.main_layout.addWidget(self.size_grip, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)

    def menuBar(self) -> QMenuBar:
        # Переопределяем метод, чтобы он возвращал меню из нашего кастомного заголовка
        return self.title_bar.menu_bar

    def setWindowTitle(self, title: str) -> None:
        # Переопределяем, чтобы заголовок обновлялся в нашем виджете
        super().setWindowTitle(title)
        if hasattr(self, 'title_bar'):
            self.title_bar.update_window_title(title)

    def mousePressEvent(self, event: QMouseEvent):
        # Запоминаем позицию клика для перетаскивания
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() < self.title_bar.height():
            self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent):
        # Перетаскиваем окно
        if event.buttons() == Qt.MouseButton.LeftButton and self.old_pos:
            delta = QPoint(event.globalPosition().toPoint() - self.old_pos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.old_pos = QPoint()

``````

### 2. Обновление `AddMenu`

Теперь `AddMenu` будет наследоваться от нашего `FramelessWindow` вместо `QMainWindow`.

``````python path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/App/中Menu/Widget.py" encoding="utf-8"
# Убираем QMainWindow, т.к. FramelessWindow уже является его наследником
from BatSpec.App.中Menu.中App.Widget import AppMenu
# from BatSpec.App.中Menu.中Mode.Widget import ModeMenu

from BatSpec.QtUp.Locales import Locales
from BatSpec.QtUp.Themes import Themes
from BatSpec.QtUp.FramelessWindow import FramelessWindow

# Наследуемся от FramelessWindow и наших миксинов
class AddMenu(Locales.TranslateComponent, Locales.Trigger, Themes.Trigger, FramelessWindow):
    def __init__(self):
        # Инициализируем FramelessWindow, который создаст всю структуру
        FramelessWindow.__init__(self)

        # Инициализируем наши модульные меню
        self.app_menu = AppMenu(self)
        # self.mode_menu = ModeMenu(self)

        # Добавляем их в menuBar, который теперь находится внутри CustomTitleBar
        self.menuBar().addMenu(self.app_menu)
        # self.menuBar().addMenu(self.mode_menu)

    def onLanguageChange(self):
        # super() вызывать не нужно, т.к. миксины сами себя подключают
        pass

    def onThemeChange(self):
        pass

``````

### 3. Обновление `MainWindow`

Осталось только немного изменить `MainWindow`, чтобы он встраивал свой контент в правильное место, которое теперь предоставляет `AddMenu` (`main_content_layout`).

``````python path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/App/Widget.py" encoding="utf-8"
from PySide6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget

from BatSpec.App.中Menu.Widget import AddMenu
from BatSpec.QtUp.Locales import Locales
from BatSpec.QtUp.Themes import Themes
from BatSpec.QtUp.Builder import build_node as b

from BatSpec.App.中Project.Widget import ProjectTab
from BatSpec.App.中WinLab.Widget import WindowLabTab
# from .中Analize.Widget import RecordAnalisys

class MainWindow(AddMenu):
    def __init__(self) -> None:
        AddMenu.__init__(self)
    
        # AddMenu создал для нас контейнер self.main_content_layout.
        # Мы просто добавляем свои виджеты в него.
        # Мы больше не создаем QWidget и QVBoxLayout для всего окна.
        with b(self.main_content_layout, QTabWidget()) as self.main_tabs:
            # Подключаем модули
            self.project = ProjectTab(tab_widget=self.main_tabs)
            self.window_lab = WindowLabTab(tab_widget=self.main_tabs)
            # self.record_analisis = RecordAnalisys(tab_widget=self.main_tabs)

    def onLanguageChange(self):
        self.setWindowTitle(self.tr("BatSpec"))

    def onThemeChange(self):
        pass

``````

Теперь, когда ты запустишь приложение, у тебя будет окно без стандартной рамки, с кастомной верхней панелью, которую можно таскать, и кнопками управления. Это основа, которую можно дальше стилизовать с помощью QSS (аналог CSS) для придания кнопкам и панели уникального вида.