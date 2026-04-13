from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication

def get_pg_colors_from_qpalette() -> dict:
    """Извлекает текущие цвета приложения Qt и переводит их в формат для PyQtGraph."""
    app = QApplication.instance()
    if not app:
        return {}
    
    palette = app.palette()
    
    bg_color = palette.color(QPalette.ColorGroup.Normal, QPalette.ColorRole.Window)
    fg_color = palette.color(QPalette.ColorGroup.Normal, QPalette.ColorRole.WindowText)
    
    # ИСПРАВЛЕНИЕ: Явно создаем копию QColor, чтобы не повредить основной fg_color
    grid_color = QColor(fg_color)
    grid_color.setAlpha(50)

    # УЛУЧШЕНИЕ: Сделаем фон легенды на 20% прозрачным, чтобы график под ней не перекрывался наглухо
    legend_bg = QColor(bg_color)
    legend_bg.setAlpha(200)

    return {
        "background": bg_color,
        "foreground": fg_color,
        "grid": grid_color,
        "legend_bg": legend_bg,
    }