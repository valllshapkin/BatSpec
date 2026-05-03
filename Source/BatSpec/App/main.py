from BatSpec.App.Logic import application
from BatSpec.App.Widget import MainWindow

def main():
    with application():
        main_window = MainWindow()
        main_window.resize(1200, 800)
        main_window.show()

if __name__ == "__main__":
    main()