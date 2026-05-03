import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
from BatSpec.QtUp.Builder import build_node as b

# Инициализация глобальных инстансов-синглтонов
from BatSpec.App.中Menu.中ThemeSettings.Logic import AppThemes
from BatSpec.App.中Menu.中LocalesSettings.Logic import AppLocales, SUPPORTED_LANGUAGES

from BatSpec.App.中Menu.中ThemeSettings.Widget import ThemeSettingsPanel
from BatSpec.App.中Menu.中LocalesSettings.Widget import LocalesSettingsPanel

def main():
    app = QApplication(sys.argv)

    # Инициализируем синглтоны (поднимаем сервисы)
    AppThemes.init(debug=False)
    AppLocales.init(
        supported_languages=SUPPORTED_LANGUAGES, 
        jit_compile=False, 
        debug=False
    )

    main_window = QWidget()
    main_window.setWindowTitle("Тест панелей настроек")
    main_window.resize(400, 500)

    with b(main_window, QVBoxLayout()) as layout:
        with b(layout, ThemeSettingsPanel()): pass
        with b(layout, LocalesSettingsPanel()): pass

    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()