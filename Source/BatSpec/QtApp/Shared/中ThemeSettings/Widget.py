import sys
from PySide6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QFormLayout, QGroupBox, 
    QCheckBox, QComboBox
)
from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Shared.Component import Component


class ThemeSettingsPanel(*Component, QWidget):
    """
    Переиспользуемая панель для управления темами.
    """
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        
        # --- Создание UI ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.group_box = QGroupBox()
        main_layout.addWidget(self.group_box)
        
        self.form_layout = QFormLayout(self.group_box)
        
        self.cb_two_mode = QCheckBox()
        self.combo_one = QComboBox()
        self.combo_light = QComboBox()
        self.combo_dark = QComboBox()
        
        self.label_one = QLabel()
        self.label_light = QLabel()
        self.label_dark = QLabel()

        self.form_layout.addRow(self.cb_two_mode)
        self.form_layout.addRow(self.label_one, self.combo_one)
        self.form_layout.addRow(self.label_light, self.combo_light)
        self.form_layout.addRow(self.label_dark, self.combo_dark)

        # --- Заполнение данных ---
        self._populate_comboboxes()
        
        # --- Синхронизация UI при старте ---
        self._sync_ui_to_settings()
        
        # --- Подключение сигналов от UI к логике ---
        self.cb_two_mode.toggled.connect(self._on_user_toggled_mode)
        self.combo_one.currentTextChanged.connect(self._on_user_changed_one)
        self.combo_light.currentTextChanged.connect(self._on_user_changed_light)
        self.combo_dark.currentTextChanged.connect(self._on_user_changed_dark)
        
        # --- Подключение сигналов от логики к UI ---
        Themes.SETTINGS.one_updated.connect(self._on_setting_one_updated)
        Themes.SETTINGS.light_updated.connect(self._on_setting_light_updated)
        Themes.SETTINGS.dark_updated.connect(self._on_setting_dark_updated)
        Themes.SETTINGS.twoMode_updated.connect(self._on_setting_mode_updated)

    def _populate_comboboxes(self):
        all_themes = list(Themes.THEMES.keys())
        light_themes = [k for k, v in Themes.THEMES.items() if not v.is_dark_theme()]
        dark_themes = [k for k, v in Themes.THEMES.items() if v.is_dark_theme()]
        
        self.combo_one.addItems(all_themes)
        self.combo_light.addItems(light_themes)
        self.combo_dark.addItems(dark_themes)

    def _sync_ui_to_settings(self):
        self.cb_two_mode.setChecked(Themes.SETTINGS.twoMode)
        self.combo_one.setCurrentText(Themes.SETTINGS.one)
        self.combo_light.setCurrentText(Themes.SETTINGS.light)
        self.combo_dark.setCurrentText(Themes.SETTINGS.dark)
        self._update_combo_states(Themes.SETTINGS.twoMode)

    def _update_combo_states(self, is_two_mode: bool):
        self.combo_one.setEnabled(not is_two_mode)
        self.combo_light.setEnabled(is_two_mode)
        self.combo_dark.setEnabled(is_two_mode)

    # --- Слоты, реагирующие на действия ПОЛЬЗОВАТЕЛЯ в UI ---
    def _on_user_toggled_mode(self, checked: bool):
        Themes.SETTINGS.change_mod(checked)

    def _on_user_changed_one(self, theme_name: str):
        if not Themes.SETTINGS.twoMode:
            Themes.SETTINGS.set_one_and_apply(theme_name)

    def _on_user_changed_light(self, theme_name: str):
        if Themes.SETTINGS.twoMode:
            Themes.SETTINGS.set_light_and_apply(theme_name)

    def _on_user_changed_dark(self, theme_name: str):
        if Themes.SETTINGS.twoMode:
            Themes.SETTINGS.set_dark_and_apply(theme_name)

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

    def onThemeChange(self):
        pass 
    
    def onLanguageChange(self):
        self.group_box.setTitle(self.tr("Appearance"))
        self.cb_two_mode.setText(self.tr("Synchronize with system theme"))
        self.label_one.setText(self.tr("Theme (Static):"))
        self.label_light.setText(self.tr("Light theme:"))
        self.label_dark.setText(self.tr("Dark theme:"))