from PySide6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QComboBox, QPushButton, QListWidget, QListWidgetItem, QDialog
)
from PySide6.QtCore import QLocale, QTimer, Qt
from W.PySide6.Core.Builder import build_node as b
from BatSpec.App.中Menu.中LocalesSettings.Logic import AppLocales

from W.PySide6.App.Locales import Locales
from W.PySide6.Core.Lifecycle import ComponentLifecycle

REVERT_TIMEOUT = 15

class ConfirmDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Confirm Language"))
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self.setFixedSize(250, 100)
        
        self.countdown = REVERT_TIMEOUT
        self.result_action = "revert"

        with b(self, QVBoxLayout()) as layout:
            with b(layout, QLabel()) as self.label:
                self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            with b(layout, QHBoxLayout()) as btn_layout:
                with b(btn_layout, QPushButton(self.tr("Keep"))) as self.btn_keep:
                    self.btn_keep.clicked.connect(self.accept_keep)
                with b(btn_layout, QPushButton(self.tr("Revert"))) as self.btn_revert:
                    self.btn_revert.clicked.connect(self.accept_revert)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
        
        self.update_text()
        self.timer.start()

    def update_text(self):
        self.label.setText(self.tr("Reverting in %1 sec...").replace("%1", str(self.countdown)))

    def tick(self):
        self.countdown -= 1
        if self.countdown <= 0:
            self.accept_revert()
        else:
            self.update_text()

    def accept_keep(self):
        self.result_action = "keep"
        self.accept()

    def accept_revert(self):
        self.result_action = "revert"
        self.reject()

class LocalesSettingsPanel(Locales.TranslateComponent, ComponentLifecycle, QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def __init_graph__(self):
        with b(self, QVBoxLayout()) as main_layout:
            main_layout.setContentsMargins(0, 0, 0, 0)

            with b(main_layout, QGroupBox()) as self.group_box:
                with b(self.group_box, QVBoxLayout()) as group_layout:
                    with b(group_layout, QHBoxLayout()) as add_row:
                        with b(add_row, QComboBox()) as self.combo_language:
                            add_row.setStretchFactor(self.combo_language, 2)
                        with b(add_row, QComboBox()) as self.combo_country:
                            add_row.setStretchFactor(self.combo_country, 2)
                        with b(add_row, QPushButton()) as self.btn_add:
                            add_row.setStretchFactor(self.btn_add, 1)

                    with b(group_layout, QListWidget()) as self.list_widget:
                        self.list_widget.setDragDropMode(QListWidget.DragDropMode.InternalMove)

                    with b(group_layout, QHBoxLayout()) as ctrl_row:
                        with b(ctrl_row, QPushButton()) as self.btn_move_up: pass
                        with b(ctrl_row, QPushButton()) as self.btn_move_down: pass
                        with b(ctrl_row, QPushButton()) as self.btn_remove: pass
                        
                        ctrl_row.addStretch()
                        
                        with b(ctrl_row, QPushButton()) as self.btn_apply: pass
                        with b(ctrl_row, QPushButton()) as self.btn_refresh:
                            self.btn_refresh.setFixedWidth(40)

    def __init_signal__(self):
        self.combo_language.currentIndexChanged.connect(self._on_language_selected)
        self.btn_add.clicked.connect(self._on_add_clicked)
        self.btn_remove.clicked.connect(self._on_remove_clicked)
        self.btn_move_up.clicked.connect(self._on_move_up)
        self.btn_move_down.clicked.connect(self._on_move_down)
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        AppLocales.settings.locales_updated.connect(self._on_setting_locales_updated)

    def __init_ready__(self):
        self._populate_language_combo()
        self._sync_ui_to_settings()

    def _populate_language_combo(self):
        self.combo_language.clear()
        seen = set()
        for loc in AppLocales.api.supported_languages:
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
            for loc in AppLocales.api.supported_languages:
                if loc.language() == l_enum:
                    self.combo_country.addItem(QLocale.territoryToString(loc.territory()), loc.territory())
        self.combo_country.model().sort(0)

    def _sync_ui_to_settings(self):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for loc in AppLocales.api.current_locales:
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
        self._sync_ui_to_settings()

    def _on_apply_clicked(self):
        locs = [self.list_widget.item(i).data(256) for i in range(self.list_widget.count())]
        if not locs: return
        
        previous_locales = list(AppLocales.api.current_locales)
        AppLocales.api.change_language(locs)

        dialog = ConfirmDialog(self)
        dialog.exec()

        if dialog.result_action == "revert":
            AppLocales.api.change_language(previous_locales)
            
        self._sync_ui_to_settings()

    def _on_setting_locales_updated(self, locales: list):
        pass

    def onLanguageChange(self):
        self.group_box.setTitle(self.tr("Language Settings"))
        self.btn_add.setText(self.tr("Add"))
        self.btn_remove.setText(self.tr("Remove"))
        self.btn_move_up.setText(self.tr("Up"))
        self.btn_move_down.setText(self.tr("Down"))
        self.btn_refresh.setText("↻")
        self.btn_apply.setText(self.tr("Apply"))
        super().onLanguageChange()
