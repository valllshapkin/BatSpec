import sys
import threading
from PySide6 import QtWidgets
from .func1d_visualizer import Function1DVisualizer, update_function
from .spec2d_visualizer import Spec2DVisualizer, update_spec2d


def run_visualizer(worker_func, *args, **kwargs):
    """
    Создает окна Qt, запускает переданную функцию в фоновом потоке,
    а затем входит в главный цикл обработки событий GUI.
    """
    # 1. Инициализация приложения
    # Проверяем, нет ли уже созданного QApplication (полезно при перезапусках/дебаге)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # Импортируем окна здесь

    # 2. Создаем и показываем окна
    # Важно: сохраняем ссылки (window1, window2), чтобы сборщик мусора их не удалил
    window2 = Function1DVisualizer()
    window2.show()
    
    window1 = Spec2DVisualizer()
    window1.show()

    # 3. ЗАПУСК ВАШИХ ВЫЧИСЛЕНИЙ В ФОНЕ
    # daemon=True означает, что как только пользователь закроет окна (app.exec завершится),
    # этот фоновый поток автоматически жестко остановится.
    calc_thread = threading.Thread(
        target=worker_func,
        args=args,       # Передаем позиционные аргументы, если есть
        kwargs=kwargs,   # Передаем именованные аргументы, если есть
        daemon=True      
    )
    calc_thread.start()

    # 4. Вход в главный цикл событий Qt (Блокирующий вызов)
    sys.exit(app.exec())