import qt_themes
from PySide6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QFormLayout, QGroupBox, 
    QCheckBox, QComboBox
)
from W.PySide6.QtBuilder import build_node as b
from BatSpec.App.中Menu.中ThemeSettings.Logic import AppThemes

from W.PySide6.QtLocales import Locales
from W.PySide6.QtSсheme import ComponentLifecycle

class ThemeSettingsPanel(Locales.TranslateComponent, ComponentLifecycle, QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def __init_graph__(self):
        with b(self, QVBoxLayout()) as main_layout:
            main_layout.setContentsMargins(0, 0, 0, 0)
            
            with b(main_layout, QGroupBox()) as self.group_box:
                with b(self.group_box, QFormLayout()) as self.form_layout:
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

    def __init_signal__(self):
        self.cb_two_mode.toggled.connect(self._on_user_toggled_mode)
        self.combo_one.currentTextChanged.connect(self._on_user_changed_one)
        self.combo_light.currentTextChanged.connect(self._on_user_changed_light)
        self.combo_dark.currentTextChanged.connect(self._on_user_changed_dark)
        
        AppThemes.settings.twoMode_updated.connect(self._on_setting_mode_updated)
        AppThemes.settings.one_updated.connect(self._on_setting_one_updated)
        AppThemes.settings.light_updated.connect(self._on_setting_light_updated)
        AppThemes.settings.dark_updated.connect(self._on_setting_dark_updated)

    def __init_ready__(self):
        self._populate_comboboxes()
        self._sync_ui_to_settings()

    def _populate_comboboxes(self):
        all_themes_dict = qt_themes.get_themes()
        
        all_themes = list(all_themes_dict.keys())
        light_themes = [k for k, v in all_themes_dict.items() if not v.is_dark_theme()]
        dark_themes = [k for k, v in all_themes_dict.items() if v.is_dark_theme()]
        
        self.combo_one.addItems(all_themes)
        self.combo_light.addItems(light_themes)
        self.combo_dark.addItems(dark_themes)

    def _sync_ui_to_settings(self):
        self.cb_two_mode.setChecked(AppThemes.api.twoMode)
        self.combo_one.setCurrentText(AppThemes.api.one)
        self.combo_light.setCurrentText(AppThemes.api.light)
        self.combo_dark.setCurrentText(AppThemes.api.dark)
        self._update_combo_states(AppThemes.api.twoMode)

    def _update_combo_states(self, is_two_mode: bool):
        self.combo_one.setEnabled(not is_two_mode)
        self.combo_light.setEnabled(is_two_mode)
        self.combo_dark.setEnabled(is_two_mode)

    def _on_user_toggled_mode(self, checked: bool):
        AppThemes.api.change_mod(checked)

    def _on_user_changed_one(self, theme_name: str):
        if not AppThemes.api.twoMode:
            AppThemes.api.set_one_and_apply(theme_name)

    def _on_user_changed_light(self, theme_name: str):
        if AppThemes.api.twoMode:
            AppThemes.api.set_light_and_apply(theme_name)

    def _on_user_changed_dark(self, theme_name: str):
        if AppThemes.api.twoMode:
            AppThemes.api.set_dark_and_apply(theme_name)

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

    def onLanguageChange(self):
        self.group_box.setTitle(self.tr("Appearance"))
        self.cb_two_mode.setText(self.tr("Synchronize with system theme"))
        self.label_one.setText(self.tr("Theme (Static):"))
        self.label_light.setText(self.tr("Light theme:"))
        self.label_dark.setText(self.tr("Dark theme:"))
        super().onLanguageChange()
