from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, 
    QComboBox, QLabel
)
from PySide6.QtCore import QLocale
# Указываем на ваш модуль с логикой локалей
from BatSpec.QtApp.Services.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Services.中ThemeSettings.Logic import Themes 


class LocalesSettingsPanel(Locales.TranslateComponent, Locales.Trigger, Themes.Trigger, QWidget):
    """
    Переиспользуемая панель для управления языком интерфейса.
    """
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        
        # --- Создание UI ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.group_box = QGroupBox()
        main_layout.addWidget(self.group_box)
        
        self.form_layout = QFormLayout(self.group_box)
        
        self.label_language = QLabel()
        self.combo_language = QComboBox()
        
        self.form_layout.addRow(self.label_language, self.combo_language)

        # --- Заполнение данных ---
        self._populate_combobox()
        
        # --- Синхронизация UI при старте ---
        self._sync_ui_to_settings()
        
        # --- Подключение сигналов от UI к логике ---
        self.combo_language.currentIndexChanged.connect(self._on_user_changed_language)
        
        # --- Подключение сигналов от логики к UI ---
        Locales.SETTINGS.locales_update.connect(self._on_setting_language_updated)

    def _populate_combobox(self):
        """Заполняет комбобокс поддерживаемыми языками."""
        for locale in Locales.SUPPORTED_LANGUAGES:
            # Отображаем нативное название языка (e.g., "Русский", "English", "日本語")
            # и сохраняем сам объект QLocale в данных элемента
            native_name = locale.nativeLanguageName().capitalize()
            self.combo_language.addItem(native_name, userData=locale)

    def _sync_ui_to_settings(self):
        """Устанавливает в комбобоксе текущий язык из настроек."""
        if Locales.SETTINGS.locales:
            current_locale = Locales.SETTINGS.locales[0]
            # Ищем элемент в комбобоксе по сохраненному объекту QLocale
            index = self.combo_language.findData(current_locale)
            if index != -1:
                self.combo_language.setCurrentIndex(index)

    # --- Слот, реагирующий на действия ПОЛЬЗОВАТЕЛЯ в UI ---
    def _on_user_changed_language(self, index: int):
        """Вызывается, когда пользователь выбирает новый язык в комбобоксе."""
        if index == -1:
            return
            
        selected_locale = self.combo_language.itemData(index)
        
        # Формируем список локалей: выбранная + английский как запасной
        # Это хорошая практика, если в выбранном языке не найдется перевод.
        new_locales_list = [selected_locale]
        if selected_locale.language() != QLocale.Language.English:
            new_locales_list.append(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))

        # Вызываем метод в логике для смены языка
        Locales.SETTINGS.change_language(new_locales_list)

    # --- Слот, реагирующий на ИЗМЕНЕНИЯ В НАСТРОЙКАХ извне ---
    def _on_setting_language_updated(self, locales: list):
        """Обновляет UI, если язык был изменен из другой части программы."""
        if locales:
            primary_locale = locales[0]
            index = self.combo_language.findData(primary_locale)
            if index != -1:
                # Блокируем сигналы, чтобы избежать рекурсивного вызова
                # _on_user_changed_language
                self.combo_language.blockSignals(True)
                self.combo_language.setCurrentIndex(index)
                self.combo_language.blockSignals(False)

    def onThemeChange(self):
        pass
    
    def onLanguageChange(self):
        """
        Переводит текст виджетов на этой панели.
        Вызывается автоматически благодаря наследованию от Locales.Trigger.
        """
        self.group_box.setTitle(self.tr("Language"))
        self.label_language.setText(self.tr("Interface language:"))