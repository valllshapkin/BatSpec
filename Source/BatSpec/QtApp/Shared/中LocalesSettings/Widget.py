import sys
from PySide6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QComboBox, QPushButton, QListWidget, QListWidgetItem
)
from PySide6.QtCore import QLocale, QTimer
from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Shared.Component import Component

REVERT_TIMEOUT = 15

class LocalesSettingsPanel(*Component, QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self._pending_locales: list | None = None
        self._previous_locales: list = list(Locales.SETTINGS.locales)
        self._revert_countdown = REVERT_TIMEOUT

        # --- Главный контейнер ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.group_box = QGroupBox()
        main_layout.addWidget(self.group_box)
        group_layout = QVBoxLayout(self.group_box)

        # 1. СТРОКА ДОБАВЛЕНИЯ (Теперь наверху)
        add_row = QHBoxLayout()
        self.combo_language = QComboBox()
        self.combo_country = QComboBox()
        self.btn_add = QPushButton()
        add_row.addWidget(self.combo_language, 2)
        add_row.addWidget(self.combo_country, 2)
        add_row.addWidget(self.btn_add, 1)
        group_layout.addLayout(add_row)

        # 2. СПИСОК
        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        group_layout.addWidget(self.list_widget)

        # 3. СТРОКА УПРАВЛЕНИЯ (Вверх/Вниз/Удалить + Пружина + Apply/Refresh)
        ctrl_row = QHBoxLayout()
        self.btn_move_up = QPushButton()
        self.btn_move_down = QPushButton()
        self.btn_remove = QPushButton()
        
        self.btn_apply = QPushButton() # Apply теперь здесь
        self.btn_refresh = QPushButton()
        self.btn_refresh.setFixedWidth(40)

        ctrl_row.addWidget(self.btn_move_up)
        ctrl_row.addWidget(self.btn_move_down)
        ctrl_row.addWidget(self.btn_remove)
        ctrl_row.addStretch() # ПРУЖИНА
        ctrl_row.addWidget(self.btn_apply)
        ctrl_row.addWidget(self.btn_refresh)
        group_layout.addLayout(ctrl_row)

        # 4. ПАНЕЛЬ ПОДТВЕРЖДЕНИЯ (Текст %1 + Пружина + Keep/Revert)
        self.confirm_bar = QWidget()
        confirm_layout = QHBoxLayout(self.confirm_bar)
        confirm_layout.setContentsMargins(0, 5, 0, 0)

        self.label_countdown = QLabel()
        self.btn_keep = QPushButton()   # Кнопка подтверждения "Оставить как есть"
        self.btn_revert = QPushButton() # Кнопка отмены "Вернуть"

        confirm_layout.addWidget(self.label_countdown)
        confirm_layout.addStretch() # ПРУЖИНА
        confirm_layout.addWidget(self.btn_keep)
        confirm_layout.addWidget(self.btn_revert)
        
        group_layout.addWidget(self.confirm_bar)

        # Начальное состояние (скрываем подтверждение)
        self.confirm_bar.hide()

        # --- Таймер ---
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._on_timer_tick)

        # --- Инициализация и Сигналы ---
        self._populate_language_combo()
        self._sync_ui_to_settings()

        self.combo_language.currentIndexChanged.connect(self._on_language_selected)
        self.btn_add.clicked.connect(self._on_add_clicked)
        self.btn_remove.clicked.connect(self._on_remove_clicked)
        self.btn_move_up.clicked.connect(self._on_move_up)
        self.btn_move_down.clicked.connect(self._on_move_down)
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        self.btn_keep.clicked.connect(self._on_keep_clicked)
        self.btn_revert.clicked.connect(self._on_revert_clicked)

        Locales.SETTINGS.locales_update.connect(self._on_setting_locales_updated)

    # ------------------------------------------------------------------
    # ЛОГИКА
    # ------------------------------------------------------------------

    def _update_countdown_label(self):
        # Безопасная подстановка для PySide6
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
        
        # Скрываем кнопку применить и показываем панель подтверждения
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
        for loc in Locales.SUPPORTED_LANGUAGES:
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
            for loc in Locales.SUPPORTED_LANGUAGES:
                if loc.language() == l_enum:
                    self.combo_country.addItem(QLocale.territoryToString(loc.territory()), loc.territory())
        self.combo_country.model().sort(0)

    def _sync_ui_to_settings(self):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for loc in Locales.SETTINGS.locales:
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
        self._previous_locales = list(Locales.SETTINGS.locales)
        self._pending_locales = locs
        Locales.SETTINGS.change_language(locs)
        self._show_confirmation_mode()

    def _on_keep_clicked(self):
        self._hide_confirmation_mode()

    def _on_revert_clicked(self):
        Locales.SETTINGS.change_language(self._previous_locales)
        self._hide_confirmation_mode()
        self._sync_ui_to_settings()

    def _on_setting_locales_updated(self, locales: list):
        if self._pending_locales is not None: return
        self._sync_ui_to_settings()

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