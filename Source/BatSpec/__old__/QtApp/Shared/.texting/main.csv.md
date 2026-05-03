Я переписал компоненты настроек локализации и тем, используя новый декларативный UI-билдер (`build_node as b`) и адаптировал их под обновленную архитектуру классов `Locales` и `Themes` из `QtUp`.

Так как новые `Locales` и `Themes` теперь являются инстанцируемыми классами (не глобальными статичными модулями), я применил паттерн **Dependency Injection** (как в твоем примере с `AppStore`). Теперь виджеты принимают инстансы `locales` и `themes` в `__init__`.

Также я встроил логику триггеров событий прямо в виджеты (через `changeEvent`), что делает их независимыми от сложных цепочек наследования миксинов.

``````py path="BatSpec/App/Shared/ThemeSettings/Widget.py" encoding="utf-8"
import qt_themes
from PySide6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QFormLayout, QGroupBox, 
    QCheckBox, QComboBox
)
from PySide6.QtCore import QEvent
from BatSpec.QtUp.Builder import build_node as b
from BatSpec.QtUp.Themes import Themes


class ThemeSettingsPanel(QWidget):
    """
    Переиспользуемая панель для управления темами.
    """
    def __init__(self, themes: Themes, parent=None):
        super().__init__(parent)
        self.themes = themes
        
        self.setup_ui()
        self._populate_comboboxes()
        self._sync_ui_to_settings()
        self._connect_signals()
        
        # Инициализируем тексты
        self.onLanguageChange()

    def setup_ui(self):
        with b(self, QVBoxLayout()) as main_layout:
            main_layout.setContentsMargins(0, 0, 0, 0)
            
            with b(main_layout, QGroupBox()) as self.group_box:
                with b(self.group_box, QFormLayout()) as self.form_layout:
                    # Создаем виджеты
                    self.cb_two_mode = QCheckBox()
                    self.combo_one = QComboBox()
                    self.combo_light = QComboBox()
                    self.combo_dark = QComboBox()
                    
                    self.label_one = QLabel()
                    self.label_light = QLabel()
                    self.label_dark = QLabel()

                    # Добавляем их в QFormLayout
                    self.form_layout.addRow(self.cb_two_mode)
                    self.form_layout.addRow(self.label_one, self.combo_one)
                    self.form_layout.addRow(self.label_light, self.combo_light)
                    self.form_layout.addRow(self.label_dark, self.combo_dark)

    def _populate_comboboxes(self):
        all_themes_dict = qt_themes.get_themes()
        
        all_themes = list(all_themes_dict.keys())
        light_themes = [k for k, v in all_themes_dict.items() if not v.is_dark_theme()]
        dark_themes = [k for k, v in all_themes_dict.items() if v.is_dark_theme()]
        
        self.combo_one.addItems(all_themes)
        self.combo_light.addItems(light_themes)
        self.combo_dark.addItems(dark_themes)

    def _sync_ui_to_settings(self):
        self.cb_two_mode.setChecked(self.themes.api.twoMode)
        self.combo_one.setCurrentText(self.themes.api.one)
        self.combo_light.setCurrentText(self.themes.api.light)
        self.combo_dark.setCurrentText(self.themes.api.dark)
        self._update_combo_states(self.themes.api.twoMode)

    def _update_combo_states(self, is_two_mode: bool):
        self.combo_one.setEnabled(not is_two_mode)
        self.combo_light.setEnabled(is_two_mode)
        self.combo_dark.setEnabled(is_two_mode)

    def _connect_signals(self):
        # --- Сигналы от UI к логике ---
        self.cb_two_mode.toggled.connect(self._on_user_toggled_mode)
        self.combo_one.currentTextChanged.connect(self._on_user_changed_one)
        self.combo_light.currentTextChanged.connect(self._on_user_changed_light)
        self.combo_dark.currentTextChanged.connect(self._on_user_changed_dark)
        
        # --- Сигналы от логики к UI ---
        self.themes.signals.one_updated.connect(self._on_setting_one_updated)
        self.themes.signals.light_updated.connect(self._on_setting_light_updated)
        self.themes.signals.dark_updated.connect(self._on_setting_dark_updated)
        self.themes.signals.twoMode_updated.connect(self._on_setting_mode_updated)

    # --- Слоты, реагирующие на действия ПОЛЬЗОВАТЕЛЯ в UI ---
    def _on_user_toggled_mode(self, checked: bool):
        self.themes.api.change_mod(checked)

    def _on_user_changed_one(self, theme_name: str):
        if not self.themes.api.twoMode:
            self.themes.api.set_one_and_apply(theme_name)

    def _on_user_changed_light(self, theme_name: str):
        if self.themes.api.twoMode:
            self.themes.api.set_light_and_apply(theme_name)

    def _on_user_changed_dark(self, theme_name: str):
        if self.themes.api.twoMode:
            self.themes.api.set_dark_and_apply(theme_name)

    # --- Слоты, реагирующие на ИЗМЕНЕНИЯ В НАСТРОЙКАХ извне ---
    def _on_setting_mode_updated(self, is_two_mode: bool):
        self.cb_two_mode.blockSignals(True)
        self.cb_two_mode.setChecked(is_two_mode)
        self.cb_two_mode.blockSignals(False)
        self._update_combo_states(is_two_mode)

    def _on_setting_one_updated(self, theme_name: str):
        self.combo_one.blockSignals(True)
        self.combo_one.setCurrentText(theme_name)
        self.combo_one.blockSignals(False)

    def _on_setting_light_updated(self, theme_name: str):
        self.combo_light.blockSignals(True)
        self.combo_light.setCurrentText(theme_name)
        self.combo_light.blockSignals(False)
        
    def _on_setting_dark_updated(self, theme_name: str):
        self.combo_dark.blockSignals(True)
        self.combo_dark.setCurrentText(theme_name)
        self.combo_dark.blockSignals(False)

    # --- Обработка смены локали / темы ---
    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.LanguageChange:
            self.onLanguageChange()
        elif event.type() == QEvent.Type.PaletteChange:
            self.onThemeChange()
        super().changeEvent(event)

    def onThemeChange(self):
        pass 
    
    def onLanguageChange(self):
        self.group_box.setTitle(self.tr("Appearance"))
        self.cb_two_mode.setText(self.tr("Synchronize with system theme"))
        self.label_one.setText(self.tr("Theme (Static):"))
        self.label_light.setText(self.tr("Light theme:"))
        self.label_dark.setText(self.tr("Dark theme:"))

``````

``````py path="BatSpec/App/Shared/LocalesSettings/Widget.py" encoding="utf-8"
from PySide6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QComboBox, QPushButton, QListWidget, QListWidgetItem
)
from PySide6.QtCore import QLocale, QTimer, QEvent
from BatSpec.QtUp.Builder import build_node as b
from BatSpec.QtUp.Locales import Locales

REVERT_TIMEOUT = 15

class LocalesSettingsPanel(QWidget):
    def __init__(self, locales: Locales, parent=None):
        super().__init__(parent)
        self.locales = locales

        self._pending_locales: list[QLocale] | None = None
        self._previous_locales: list[QLocale] = list(self.locales.api.current_locales)
        self._revert_countdown = REVERT_TIMEOUT

        self.setup_ui()

        # --- Таймер ---
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._on_timer_tick)

        # --- Инициализация и Сигналы ---
        self._populate_language_combo()
        self._sync_ui_to_settings()
        self._connect_signals()

        # Инициализируем тексты
        self.onLanguageChange()

    def setup_ui(self):
        with b(self, QVBoxLayout()) as main_layout:
            main_layout.setContentsMargins(0, 0, 0, 0)

            with b(main_layout, QGroupBox()) as self.group_box:
                with b(self.group_box, QVBoxLayout()) as group_layout:
                    
                    # 1. СТРОКА ДОБАВЛЕНИЯ
                    with b(group_layout, QHBoxLayout()) as add_row:
                        with b(add_row, QComboBox()) as self.combo_language:
                            add_row.setStretchFactor(self.combo_language, 2)
                        with b(add_row, QComboBox()) as self.combo_country:
                            add_row.setStretchFactor(self.combo_country, 2)
                        with b(add_row, QPushButton()) as self.btn_add:
                            add_row.setStretchFactor(self.btn_add, 1)

                    # 2. СПИСОК
                    with b(group_layout, QListWidget()) as self.list_widget:
                        self.list_widget.setDragDropMode(QListWidget.DragDropMode.InternalMove)

                    # 3. СТРОКА УПРАВЛЕНИЯ
                    with b(group_layout, QHBoxLayout()) as ctrl_row:
                        with b(ctrl_row, QPushButton()) as self.btn_move_up: pass
                        with b(ctrl_row, QPushButton()) as self.btn_move_down: pass
                        with b(ctrl_row, QPushButton()) as self.btn_remove: pass
                        
                        ctrl_row.addStretch() # ПРУЖИНА
                        
                        with b(ctrl_row, QPushButton()) as self.btn_apply: pass
                        with b(ctrl_row, QPushButton()) as self.btn_refresh:
                            self.btn_refresh.setFixedWidth(40)

                    # 4. ПАНЕЛЬ ПОДТВЕРЖДЕНИЯ
                    with b(group_layout, QWidget()) as self.confirm_bar:
                        with b(self.confirm_bar, QHBoxLayout()) as confirm_layout:
                            confirm_layout.setContentsMargins(0, 5, 0, 0)
                            with b(confirm_layout, QLabel()) as self.label_countdown: pass
                            
                            confirm_layout.addStretch() # ПРУЖИНА
                            
                            with b(confirm_layout, QPushButton()) as self.btn_keep: pass
                            with b(confirm_layout, QPushButton()) as self.btn_revert: pass

        # Начальное состояние (скрываем подтверждение)
        self.confirm_bar.hide()

    def _connect_signals(self):
        self.combo_language.currentIndexChanged.connect(self._on_language_selected)
        self.btn_add.clicked.connect(self._on_add_clicked)
        self.btn_remove.clicked.connect(self._on_remove_clicked)
        self.btn_move_up.clicked.connect(self._on_move_up)
        self.btn_move_down.clicked.connect(self._on_move_down)
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        self.btn_keep.clicked.connect(self._on_keep_clicked)
        self.btn_revert.clicked.connect(self._on_revert_clicked)

        self.locales.signals.locales_updated.connect(self._on_setting_locales_updated)

    # ------------------------------------------------------------------
    # ЛОГИКА
    # ------------------------------------------------------------------

    def _update_countdown_label(self):
        translated_text = self.tr("Reverting in %1 sec...")
        display_text = translated_text.replace("%1", str(self._revert_countdown))
        self.label_countdown.setText(display_text)

    def _on_timer_tick(self):
        self._revert_countdown -= 1
        if self._revert_countdown <= 0:
            self._on_revert_clicked()
        else:
            self._update_countdown_label()

    def _show_confirmation_mode(self):
        self._revert_countdown = REVERT_TIMEOUT
        self._update_countdown_label()
        
        self.btn_apply.setEnabled(False)
        self.confirm_bar.show()
        
        self.list_widget.setEnabled(False)
        self.btn_add.setEnabled(False)
        self._timer.start()

    def _hide_confirmation_mode(self):
        self._timer.stop()
        
        self.btn_apply.setEnabled(True)
        self.confirm_bar.hide()
        
        self.list_widget.setEnabled(True)
        self.btn_add.setEnabled(True)
        self._pending_locales = None

    def _populate_language_combo(self):
        self.combo_language.clear()
        seen = set()
        for loc in self.locales.api.supported_languages:
            l = loc.language()
            if l not in seen:
                seen.add(l)
                self.combo_language.addItem(QLocale.languageToString(l), l)
        self.combo_language.model().sort(0)
        self._populate_country_combo()

    def _populate_country_combo(self):
        self.combo_country.clear()
        l_enum = self.combo_language.currentData()
        if l_enum:
            for loc in self.locales.api.supported_languages:
                if loc.language() == l_enum:
                    self.combo_country.addItem(QLocale.territoryToString(loc.territory()), loc.territory())
        self.combo_country.model().sort(0)

    def _sync_ui_to_settings(self):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for loc in self.locales.api.current_locales:
            self._append_locale_item(loc)
        self.list_widget.blockSignals(False)

    def _append_locale_item(self, locale: QLocale):
        for i in range(self.list_widget.count()):
            if self.list_widget.item(i).data(256) == locale: return
        label = f"{QLocale.languageToString(locale.language())} ({QLocale.territoryToString(locale.territory())})"
        item = QListWidgetItem(label)
        item.setData(256, locale)
        self.list_widget.addItem(item)

    def _on_language_selected(self):
        self._populate_country_combo()

    def _on_add_clicked(self):
        l, t = self.combo_language.currentData(), self.combo_country.currentData()
        if l: self._append_locale_item(QLocale(l, t))

    def _on_remove_clicked(self):
        row = self.list_widget.currentRow()
        if row >= 0: self.list_widget.takeItem(row)

    def _on_move_up(self):
        row = self.list_widget.currentRow()
        if row > 0:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row - 1, item)
            self.list_widget.setCurrentRow(row - 1)

    def _on_move_down(self):
        row = self.list_widget.currentRow()
        if row < self.list_widget.count() - 1:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row + 1, item)
            self.list_widget.setCurrentRow(row + 1)

    def _on_refresh_clicked(self):
        self._hide_confirmation_mode()
        self._sync_ui_to_settings()

    def _on_apply_clicked(self):
        locs = [self.list_widget.item(i).data(256) for i in range(self.list_widget.count())]
        if not locs: return
        self._previous_locales = list(self.locales.api.current_locales)
        self._pending_locales = locs
        self.locales.api.change_language(locs)
        self._show_confirmation_mode()

    def _on_keep_clicked(self):
        self._hide_confirmation_mode()

    def _on_revert_clicked(self):
        self.locales.api.change_language(self._previous_locales)
        self._hide_confirmation_mode()
        self._sync_ui_to_settings()

    def _on_setting_locales_updated(self, locales: list):
        if self._pending_locales is not None: return
        self._sync_ui_to_settings()

    # --- Обработка смены локали / темы ---
    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.LanguageChange:
            self.onLanguageChange()
        elif event.type() == QEvent.Type.PaletteChange:
            self.onThemeChange()
        super().changeEvent(event)

    def onThemeChange(self):
        pass

    def onLanguageChange(self):
        self.group_box.setTitle(self.tr("Language Settings"))
        self.btn_add.setText(self.tr("Add"))
        self.btn_remove.setText(self.tr("Remove"))
        self.btn_move_up.setText(self.tr("Up"))
        self.btn_move_down.setText(self.tr("Down"))
        self.btn_refresh.setText("↻")
        self.btn_apply.setText(self.tr("Apply"))
        self.btn_keep.setText(self.tr("Keep"))
        self.btn_revert.setText(self.tr("Revert"))
        if self._timer.isActive():
            self._update_countdown_label()

``````

``````py path="BatSpec/App/Shared/__test__/test_settings.py" encoding="utf-8"
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PySide6.QtCore import QLocale

from BatSpec.QtUp.Themes import Themes
from BatSpec.QtUp.Locales import Locales
from BatSpec.QtUp.Builder import build_node as b

from BatSpec.App.Shared.ThemeSettings.Widget import ThemeSettingsPanel
from BatSpec.App.Shared.LocalesSettings.Widget import LocalesSettingsPanel

class MockAppStore:
    """Эмуляция Dependency Injection контейнера (Store), хранящего сервисы"""
    def __init__(self):
        # 1. Инициализируем классы
        self.themes = Themes()
        self.locales = Locales()

        # 2. Настраиваем окружение
        self.themes.Settings.SETTING_PATH = "test_themes.ini"
        self.locales.Settings.SETTING_PATH = "test_locales.ini"

        # 3. Запускаем их сервисы
        self.themes.init(debug=True)
        
        supported_langs = [
            QLocale(QLocale.Language.English, QLocale.Country.UnitedStates),
            QLocale(QLocale.Language.Russian, QLocale.Country.Russia),
            QLocale(QLocale.Language.German, QLocale.Country.Germany)
        ]
        self.locales.init(supported_languages=supported_langs, debug=False)


def main():
    app = QApplication(sys.argv)

    # Инициализация стора со всеми зависимостями
    store = MockAppStore()

    main_window = QWidget()
    main_window.setWindowTitle("Тест панелей настроек")
    main_window.resize(400, 500)

    # Собираем UI с новым билдером и прокидываем зависимости (DI)
    with b(main_window, QVBoxLayout()) as layout:
        with b(layout, ThemeSettingsPanel(themes=store.themes)): pass
        with b(layout, LocalesSettingsPanel(locales=store.locales)): pass

    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

``````