import weakref
import re
import pyqtgraph as pg
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication
from .colors import get_pg_colors_from_qpalette


class _PGThemeManager(QObject):
    """Синглтон-менеджер для отслеживания и перекраски виджетов PyQtGraph."""
    def __init__(self):
        super().__init__()
        self._widgets = weakref.WeakSet()
        self._is_connected = False
        self._is_updating = False  # Блокировка от бесконечной рекурсии (Segfault защита)

    def connect_to_app(self):
        """Подписывается на сигнал изменения палитры Qt."""
        if self._is_connected:
            return
        app = QApplication.instance()
        if app:
            app.paletteChanged.connect(self._on_palette_changed)
            self._is_connected = True
            self._on_palette_changed()

    def register_widget(self, widget: pg.PlotWidget):
        """Регистрирует график для получения динамических обновлений."""
        self._widgets.add(widget)
        self.connect_to_app()
        self._apply_theme_to_widget(widget, get_pg_colors_from_qpalette())

    def _on_palette_changed(self, *args, **kwargs):
        """Вызывается при смене темы приложения."""
        if self._is_updating:
            return
            
        colors = get_pg_colors_from_qpalette()
        if not colors:
            return
            
        self._is_updating = True
        try:
            # 1. Глобальные настройки PyQtGraph (для новых виджетов)
            pg.setConfigOption('background', colors['background'])
            pg.setConfigOption('foreground', colors['foreground'])
            
            # 2. Обновляем уже созданные виджеты
            for widget in self._widgets:
                # Проверка, что C++ объект виджета еще жив (защита PySide6)
                # if not getattr(widget, 'isVisible', lambda: False)():
                #     continue
                self._apply_theme_to_widget(widget, colors)
        finally:
            self._is_updating = False

    def _apply_theme_to_widget(self, widget: pg.PlotWidget, colors: dict):
        """Проходится по внутренностям PlotWidget и перекрашивает их."""
        bg = colors['background']
        fg = colors['foreground']
        legend_bg = colors['legend_bg']

        # Перекрашиваем основной фон
        widget.setBackground(bg)
        
        plot_item: pg.PlotItem = widget.getPlotItem()
        
        # Перекрашиваем оси
        for axis_name in ['left', 'right', 'top', 'bottom']:
            axis: pg.AxisItem = plot_item.getAxis(axis_name)
            if axis:
                axis.setPen(fg)
                axis.setTextPen(fg)
                
                # ИСПРАВЛЕНИЕ SEGFAULT: axis.grid - это число (alpha). 
                # Передаем его же, чтобы заставить сетку перерисоваться с новым setPen.
                if axis.grid:
                    axis.setGrid(axis.grid) 
        
        # Обновляем заголовок графика (очищаем от старого HTML)
        if plot_item.titleLabel.isVisible():
            title_text = str(plot_item.titleLabel.text)
            clean_text = re.sub(r'<[^>]+>', '', title_text)
            plot_item.setTitle(clean_text, color=fg.name())

        # Обновляем Легенду
        if plot_item.legend is not None:
            legend: pg.LegendItem = plot_item.legend
            
            # Фон и рамка самой плашки легенды
            legend.setBrush(legend_bg)
            legend.setPen(fg)
            legend.setLabelTextColor(fg)
            
            # Принудительно очищаем и перекрашиваем УЖЕ СУЩЕСТВУЮЩИЕ элементы
            for sample, label in legend.items:
                try:
                    label_text = str(label.text)
                    clean_label_text = re.sub(r'<[^>]+>', '', label_text)
                    label.setText(clean_label_text, color=fg.name())
                except RuntimeError:
                    # Защита от обращения к удаленному C++ QGraphicsItem
                    pass


# Глобальный инстанс
theme_manager = _PGThemeManager()

def setup_theme_integration():
    """Инициализирует глобальный перехват."""
    theme_manager.connect_to_app()