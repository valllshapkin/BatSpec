import sys
from BatSpec.App.Logic import application
from BatSpec.App.Widget import MainWindow

def main():
    # Контекстный менеджер `application` заботится о создании
    # QApplication, инициализации сервисов (тем, локалей)
    # и корректном завершении через sys.exit(app.exec()).
    with application() as app:
        main_window = MainWindow()
        main_window.resize(1200, 800)
        main_window.show()
        # Выход из 'with' блока вызовет app.exec()

if __name__ == "__main__":
    main()