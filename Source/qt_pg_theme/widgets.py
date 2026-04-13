import pyqtgraph as pg
from .manager import theme_manager

class ThemedPlotWidget(pg.PlotWidget):
    """
    Drop-in замена для pg.PlotWidget.
    Автоматически меняет свои цвета (фон, оси, текст) при смене темы приложения.
    """
    def __init__(self, parent=None, background='default', plotItem=None, **kargs):
        # Инициализируем со стандартными значениями, они сразу же перезапишутся менеджером
        super().__init__(parent=parent, background=background, plotItem=plotItem, **kargs)
        # Регистрируем виджет в менеджере для получения обновлений палитры
        theme_manager.register_widget(self)