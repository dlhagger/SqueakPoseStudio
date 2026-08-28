"""Package-owned shared Qt styling for SqueakPose Studio.

This module keeps visual decisions in one place so the main application code can
focus on layout and behavior.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QComboBox, QFrame, QGraphicsDropShadowEffect, QWidget

COLORS = {
    "app_bg": "#171a1d",
    "canvas_bg": "#202429",
    "panel_bg": "#24292e",
    "panel_bg_alt": "#2b3137",
    "panel_border": "#44505a",
    "panel_border_soft": "#53616c",
    "text": "#edf2f7",
    "text_muted": "#aeb9c4",
    "text_subtle": "#84909b",
    "input_bg": "#15191d",
    "input_border": "#4f5d68",
    "button_bg": "#303842",
    "button_bg_hover": "#3a4550",
    "button_bg_pressed": "#232a31",
    "button_active": "#52606c",
    "accent": "#5e8ab4",
    "accent_hover": "#6a9ac8",
    "success": "#3d8b61",
    "warning": "#93763c",
    "danger": "#94434a",
    "terminal_bg": "#090c0f",
}


def apply_panel_shadow(
    widget: QWidget, *, blur: int = 18, y_offset: int = 2, alpha: int = 90
) -> None:
    """Apply a restrained desktop-panel shadow."""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur)
    shadow.setOffset(0, y_offset)
    shadow.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(shadow)


def combo_popup_stylesheet() -> str:
    """Stylesheet for combo-box popup list views."""
    return f"""
    QListView#ComboPopup, QAbstractItemView#ComboPopup {{
        background-color: {COLORS["input_bg"]};
        alternate-background-color: {COLORS["input_bg"]};
        color: {COLORS["text"]};
        border: 1px solid {COLORS["input_border"]};
        outline: 0;
        selection-background-color: {COLORS["accent"]};
        selection-color: #ffffff;
    }}
    QListView#ComboPopup::item, QAbstractItemView#ComboPopup::item {{
        background-color: {COLORS["input_bg"]};
        color: {COLORS["text"]};
        border: none;
        min-height: 22px;
        padding: 4px 8px;
    }}
    QListView#ComboPopup::item:hover, QAbstractItemView#ComboPopup::item:hover,
    QListView#ComboPopup::item:selected, QAbstractItemView#ComboPopup::item:selected {{
        background-color: {COLORS["accent"]};
        color: #ffffff;
    }}
    """


def style_combo_popup(view: QWidget) -> None:
    """Force combo popup list and viewport backgrounds to the dark input color."""
    parent = view.parent()
    combo = parent if isinstance(parent, QComboBox) else None
    if isinstance(combo, QComboBox) and "combobox-popup" not in combo.styleSheet():
        existing = combo.styleSheet().strip()
        popup_rule = "QComboBox { combobox-popup: 0; }"
        combo.setStyleSheet(f"{existing}\n{popup_rule}" if existing else popup_rule)

    view.setObjectName("ComboPopup")
    view.setStyleSheet(combo_popup_stylesheet())
    view.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    view.setContentsMargins(0, 0, 0, 0)
    if isinstance(view, QFrame):
        view.setFrameShape(QFrame.Shape.NoFrame)
    set_spacing = getattr(view, "setSpacing", None)
    if callable(set_spacing):
        set_spacing(0)
    set_viewport_margins = getattr(view, "setViewportMargins", None)
    if callable(set_viewport_margins):
        try:
            set_viewport_margins(0, 0, 0, 0)
        except RuntimeError:
            pass

    palette = view.palette()
    palette.setColor(QPalette.ColorRole.Base, QColor(COLORS["input_bg"]))
    palette.setColor(QPalette.ColorRole.Window, QColor(COLORS["input_bg"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLORS["text"]))
    view.setPalette(palette)
    view.setAutoFillBackground(True)

    viewport_getter = getattr(view, "viewport", None)
    if callable(viewport_getter):
        viewport = viewport_getter()
        viewport.setObjectName("ComboPopupViewport")
        viewport.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        viewport.setAutoFillBackground(True)
        viewport.setPalette(palette)
        viewport.setStyleSheet(
            f"QWidget#ComboPopupViewport {{ background-color: {COLORS['input_bg']}; }}"
        )

    container = view.window()
    if container is not view and container is not combo:
        container.setObjectName("ComboPopupContainer")
        container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        container.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        container.setWindowFlag(Qt.WindowType.NoDropShadowWindowHint, True)
        container.setAutoFillBackground(True)
        container.setContentsMargins(0, 0, 0, 0)
        container.setPalette(palette)
        if isinstance(container, QFrame):
            container.setFrameShape(QFrame.Shape.NoFrame)
        container.setStyleSheet(
            "QWidget#ComboPopupContainer, QFrame#ComboPopupContainer { "
            f"background-color: {COLORS['input_bg']}; "
            f"border: 1px solid {COLORS['input_border']}; "
            "}"
        )


class ThemedComboBox(QComboBox):
    """ComboBox that keeps macOS popup containers inside the app dark theme."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._ensure_non_native_popup()
        style_combo_popup(self.view())

    def _ensure_non_native_popup(self) -> None:
        if "combobox-popup" in self.styleSheet():
            return
        existing = self.styleSheet().strip()
        popup_rule = "QComboBox { combobox-popup: 0; }"
        self.setStyleSheet(f"{existing}\n{popup_rule}" if existing else popup_rule)

    def setView(self, item_view: QWidget) -> None:  # type: ignore[override]
        super().setView(item_view)
        self._ensure_non_native_popup()
        style_combo_popup(item_view)

    def showPopup(self) -> None:
        self._ensure_non_native_popup()
        style_combo_popup(self.view())
        super().showPopup()
        style_combo_popup(self.view())


def _font_stack(preferred_family: str, system_family: str) -> str:
    """Return a Qt stylesheet font stack without generic placeholder families."""
    ignored = {"Sans Serif", "SansSerif", "sans"}
    families: list[str] = []
    for family in (preferred_family, system_family, "Arial", "Helvetica"):
        if not family or family in ignored or family in families:
            continue
        families.append(family)
    return ", ".join(f"'{family}'" for family in families)


def app_stylesheet(preferred_family: str, system_family: str) -> str:
    """Global application stylesheet."""
    font_stack = _font_stack(preferred_family, system_family)
    return f"""
    QWidget {{
        background-color: {COLORS["app_bg"]};
        color: {COLORS["text"]};
        font-family: {font_stack};
        font-size: 11pt;
    }}
    QMainWindow, QDialog {{
        background-color: {COLORS["app_bg"]};
    }}
    QGraphicsView {{
        background-color: {COLORS["canvas_bg"]};
        border: none;
    }}
    QStatusBar {{
        background-color: #111417;
        color: {COLORS["text_muted"]};
        border-top: 1px solid #2d353c;
    }}
    QMenuBar {{
        background-color: #f1f3f5;
        color: #111417;
    }}
    QMenuBar::item:selected {{
        background-color: #dfe5ea;
    }}
    QMenu {{
        background-color: #f7f8fa;
        color: #111417;
        border: 1px solid #c8d0d8;
    }}
    QMenu::item:selected {{
        background-color: #dbe8f5;
    }}
    QPushButton {{
        background-color: {COLORS["button_bg"]};
        border: 1px solid {COLORS["input_border"]};
        border-radius: 6px;
        padding: 6px 8px;
        color: {COLORS["text"]};
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {COLORS["button_bg_hover"]};
        border-color: #687783;
    }}
    QPushButton:pressed {{
        background-color: {COLORS["button_bg_pressed"]};
    }}
    QPushButton:disabled {{
        color: #7e8892;
        background-color: #252b31;
        border-color: #3b454e;
    }}
    QPushButton[activeMode="true"] {{
        background-color: {COLORS["button_active"]};
        border-color: #7c8b98;
        color: #ffffff;
        font-weight: 700;
    }}
    QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {COLORS["input_bg"]};
        border: 1px solid {COLORS["input_border"]};
        border-radius: 6px;
        padding: 5px 8px;
        color: {COLORS["text"]};
        selection-background-color: {COLORS["accent"]};
    }}
    QComboBox {{
        combobox-popup: 0;
    }}
    QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
        color: #7e8892;
        background-color: #20262b;
        border-color: #39434c;
    }}
    QComboBox::drop-down {{
        border-left: 1px solid #3f4a53;
        width: 22px;
    }}
    QComboBox QAbstractItemView {{
        background-color: #20262b;
        alternate-background-color: #20262b;
        color: {COLORS["text"]};
        border: 1px solid {COLORS["input_border"]};
        selection-background-color: {COLORS["accent"]};
        selection-color: #ffffff;
        outline: 0;
    }}
    QComboBox QAbstractItemView::item {{
        background-color: #20262b;
        color: {COLORS["text"]};
        min-height: 22px;
        padding: 4px 8px;
    }}
    QComboBox QAbstractItemView::item:hover,
    QComboBox QAbstractItemView::item:selected {{
        background-color: {COLORS["accent"]};
        color: #ffffff;
    }}
    QLabel {{
        background: transparent;
        border: none;
        color: {COLORS["text"]};
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 0;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 0;
    }}
    QScrollBar::handle {{
        background-color: #3b454e;
        border-radius: 5px;
        min-height: 28px;
        min-width: 28px;
    }}
    QScrollBar::handle:hover {{
        background-color: #53616c;
    }}
    QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
        background: transparent;
        border: none;
        width: 0;
        height: 0;
    }}
    """


def launcher_stylesheet() -> str:
    """Stylesheet for the project launcher dialog."""
    return """
    QDialog {
        background-color: #f4f6f8;
        color: #161b20;
    }
    QLabel#LauncherTitle {
        color: #14191f;
        font-size: 16pt;
        font-weight: 800;
    }
    QLabel#LauncherSubtitle {
        color: #53606b;
        font-size: 10pt;
        line-height: 1.2;
    }
    QPushButton {
        background-color: #ffffff;
        border: 1px solid #bcc7d1;
        border-radius: 7px;
        padding: 8px 18px;
        color: #1c232a;
        font-weight: 650;
        min-height: 28px;
    }
    QPushButton:hover {
        background-color: #edf4fb;
        border-color: #8ba6bd;
    }
    QPushButton#PrimaryLauncherButton {
        background-color: #245f8f;
        border-color: #245f8f;
        color: #ffffff;
    }
    QPushButton#PrimaryLauncherButton:hover {
        background-color: #2d74ad;
    }
    """


def sidebar_stylesheet() -> str:
    """Stylesheet for permanent sidebars."""
    return f"""
    QScrollArea {{
        background-color: {COLORS["app_bg"]};
        border: none;
    }}
    QWidget#SidebarContent {{
        background-color: {COLORS["app_bg"]};
    }}
    QFrame#ToolPanel {{
        background-color: {COLORS["panel_bg"]};
        border: 1px solid {COLORS["panel_border"]};
        border-radius: 8px;
    }}
    QPushButton {{
        padding-left: 6px;
        padding-right: 6px;
    }}
    QLabel#panelTitle {{
        font-weight: 800;
        font-size: 10pt;
        color: {COLORS["text"]};
        padding-bottom: 5px;
        border-bottom: 1px solid {COLORS["panel_border"]};
    }}
    QLabel#fieldLabel {{
        font-size: 9pt;
        color: {COLORS["text_muted"]};
    }}
    QLabel#sectionLabel {{
        font-size: 8pt;
        font-weight: 800;
        color: {COLORS["text_muted"]};
        letter-spacing: 0;
        padding-top: 2px;
    }}
    QLabel#progressBadge, QLabel#brushSizeBadge {{
        font-weight: 750;
        color: {COLORS["text"]};
        background-color: {COLORS["panel_bg_alt"]};
        border: 1px solid {COLORS["panel_border_soft"]};
        border-radius: 7px;
        padding: 3px 8px;
    }}
    QLabel#samHelper {{
        font-size: 9pt;
        color: {COLORS["text"]};
        background-color: #20262b;
        border: 1px solid {COLORS["panel_border_soft"]};
        border-radius: 7px;
        padding: 7px 8px;
        line-height: 1.15;
    }}
    QPushButton[layerVisibilityPill="true"] {{
        min-height: 24px;
        padding: 3px 8px;
        background-color: {COLORS["input_bg"]};
        border: 1px solid {COLORS["input_border"]};
        border-radius: 12px;
        color: {COLORS["text_subtle"]};
        font-size: 9pt;
        font-weight: 650;
    }}
    QPushButton[layerVisibilityPill="true"]:hover {{
        background-color: {COLORS["button_bg_hover"]};
        border-color: {COLORS["accent_hover"]};
        color: {COLORS["text"]};
    }}
    QPushButton[layerVisibilityPill="true"]:checked {{
        background-color: #263b4c;
        border-color: {COLORS["accent"]};
        color: #dcefff;
    }}
    QPushButton[layerVisibilityPill="true"][activeLayer="true"],
    QPushButton[layerVisibilityPill="true"][activeLayer="true"]:disabled {{
        background-color: #35546d;
        border-color: #79a9d2;
        color: #ffffff;
        font-weight: 800;
    }}
    QPushButton#samAcceptButton:disabled {{
        color: #7d8a84;
        background-color: #253129;
        border-color: #3d5145;
    }}
    QPushButton[tone="load"] {{
        background-color: #465464;
        border-color: #718294;
    }}
    QPushButton[tone="run"] {{
        background-color: #2f5f91;
        border-color: #6f9dca;
    }}
    QPushButton[tone="accept"] {{
        background-color: #2f744f;
        border-color: #70a985;
    }}
    QPushButton[tone="clear"] {{
        background-color: #664149;
        border-color: #9a6870;
    }}
    """


def hud_stylesheet() -> str:
    """Stylesheet for small canvas HUD overlays."""
    return f"""
    QFrame {{
        background-color: rgba(28, 33, 38, 218);
        border: 1px solid rgba(104, 119, 132, 150);
        border-radius: 8px;
    }}
    QLabel {{
        background: transparent;
        border: none;
        padding: 0;
        color: {COLORS["text"]};
    }}
    QLabel#hudTitle {{
        font-weight: 800;
        font-size: 10pt;
        color: {COLORS["text"]};
    }}
    QLabel#zoomValue {{
        font-weight: 800;
        font-size: 11pt;
        color: {COLORS["text"]};
    }}
    QLabel#layerEditing {{
        font-weight: 800;
        font-size: 9pt;
        color: #ffffff;
    }}
    QLabel#layerReference {{
        font-size: 8pt;
        color: #a9cce7;
    }}
    """


def train_dialog_stylesheet() -> str:
    """Stylesheet for the training dialog."""
    return f"""
    QDialog {{
        background-color: {COLORS["app_bg"]};
    }}
    QFrame#TrainSettingsPanel, QFrame#TrainOutputPanel {{
        background-color: {COLORS["panel_bg"]};
        border: 1px solid {COLORS["panel_border"]};
        border-radius: 8px;
    }}
    QLabel {{
        color: {COLORS["text"]};
        background: transparent;
        border: none;
    }}
    QLabel#TrainPanelTitle {{
        color: {COLORS["text"]};
        font-size: 12pt;
        font-weight: 800;
        padding: 0;
    }}
    QLabel#TrainStatusLabel {{
        background-color: {COLORS["panel_bg_alt"]};
        border: 1px solid {COLORS["panel_border_soft"]};
        border-radius: 10px;
        color: {COLORS["text"]};
        font-size: 9pt;
        padding: 3px 10px;
    }}
    QLabel#TrainStatusLabel[tone="running"] {{
        background-color: #214f63;
        border-color: #3f879c;
        color: #dff8ff;
    }}
    QLabel#TrainStatusLabel[tone="complete"] {{
        background-color: #214f3a;
        border-color: #3d8b61;
        color: #e4fff1;
    }}
    QLabel#TrainStatusLabel[tone="failed"] {{
        background-color: #5a2528;
        border-color: #94434a;
        color: #ffe4e8;
    }}
    QLabel#TrainStatusLabel[tone="canceled"] {{
        background-color: #5a4a25;
        border-color: #93763c;
        color: #fff5d8;
    }}
    QLabel#TrainFormLabel {{
        color: {COLORS["text_muted"]};
        font-weight: 650;
    }}
    QLabel#TrainHintLabel {{
        color: {COLORS["text_muted"]};
        font-size: 9pt;
    }}
    QLabel#TrainPhaseLabel {{
        color: #bfe8ff;
        font-size: 11pt;
        font-weight: 750;
    }}
    QLabel#TrainEpochLabel, QLabel#TrainEtaLabel {{
        color: {COLORS["text_muted"]};
        font-weight: 650;
        padding-left: 12px;
    }}
    QTabWidget#TrainOutputTabs::pane {{
        background-color: {COLORS["panel_bg_alt"]};
        border: 1px solid {COLORS["panel_border_soft"]};
        border-radius: 6px;
        top: -1px;
    }}
    QTabWidget#TrainOutputTabs QTabBar::tab {{
        background-color: {COLORS["button_bg_pressed"]};
        border: 1px solid {COLORS["panel_border_soft"]};
        color: {COLORS["text_muted"]};
        min-width: 90px;
        padding: 6px 14px;
    }}
    QTabWidget#TrainOutputTabs QTabBar::tab:selected {{
        background-color: {COLORS["panel_bg_alt"]};
        border-bottom-color: {COLORS["panel_bg_alt"]};
        color: {COLORS["text"]};
        font-weight: 700;
    }}
    QProgressBar#TrainOverallProgress, QProgressBar#TrainEpochProgress {{
        background-color: {COLORS["terminal_bg"]};
        border: 1px solid {COLORS["panel_border_soft"]};
        border-radius: 6px;
        color: #eef7fb;
        min-height: 20px;
        text-align: center;
    }}
    QProgressBar#TrainOverallProgress::chunk {{
        background-color: #2b8dbd;
        border-radius: 5px;
    }}
    QProgressBar#TrainEpochProgress::chunk {{
        background-color: #3d9b70;
        border-radius: 5px;
    }}
    QFrame#TrainMetricCard {{
        background-color: {COLORS["terminal_bg"]};
        border: 1px solid {COLORS["panel_border_soft"]};
        border-radius: 7px;
    }}
    QLabel#TrainMetricCaption {{
        color: {COLORS["text_muted"]};
        font-size: 9pt;
    }}
    QLabel#TrainMetricValue {{
        color: #f4fbff;
        font-size: 15pt;
        font-weight: 800;
    }}
    QLabel#TrainLossDetail {{
        color: {COLORS["text_muted"]};
        font-size: 9pt;
        padding: 0 3px 2px 3px;
    }}
    QLabel#TrainHistoryTitle {{
        color: {COLORS["text"]};
        font-weight: 750;
        padding-top: 2px;
    }}
    QTableWidget#TrainHistoryTable {{
        background-color: {COLORS["terminal_bg"]};
        alternate-background-color: {COLORS["terminal_bg"]};
        border: 1px solid {COLORS["panel_border_soft"]};
        color: {COLORS["text"]};
        gridline-color: #303941;
        selection-background-color: {COLORS["accent"]};
    }}
    QTableWidget#TrainHistoryTable QHeaderView::section {{
        background-color: {COLORS["button_bg_pressed"]};
        border: none;
        border-right: 1px solid {COLORS["panel_border_soft"]};
        border-bottom: 1px solid {COLORS["panel_border_soft"]};
        color: {COLORS["text_muted"]};
        font-weight: 700;
        padding: 5px;
    }}
    QPlainTextEdit#TrainLogView {{
        background-color: {COLORS["terminal_bg"]};
        color: #d8dee4;
        border: 1px solid #303941;
        border-radius: 6px;
        padding: 8px;
        selection-background-color: {COLORS["accent"]};
    }}
    """


def analysis_dialog_stylesheet() -> str:
    """Stylesheet for the analysis workflow dialog."""
    return f"""
    QDialog {{
        background-color: {COLORS["app_bg"]};
    }}
    QFrame#AnalysisPanel {{
        background-color: {COLORS["panel_bg"]};
        border: 1px solid {COLORS["panel_border"]};
        border-radius: 8px;
    }}
    QFrame#AnalysisSubPanel {{
        background-color: {COLORS["panel_bg_alt"]};
        border: 1px solid {COLORS["panel_border_soft"]};
        border-radius: 6px;
    }}
    QScrollArea#AnalysisLeftScroll {{
        background: transparent;
        border: none;
    }}
    QScrollArea#AnalysisLeftScroll QWidget {{
        background: transparent;
    }}
    QLabel {{
        color: {COLORS["text"]};
        background: transparent;
        border: none;
    }}
    QLabel#AnalysisPanelTitle {{
        color: {COLORS["text"]};
        font-size: 12pt;
        font-weight: 800;
        padding: 0;
    }}
    QLabel#AnalysisStatusLabel {{
        background-color: {COLORS["panel_bg_alt"]};
        border: 1px solid {COLORS["panel_border_soft"]};
        border-radius: 10px;
        color: {COLORS["text"]};
        font-size: 9pt;
        padding: 3px 10px;
    }}
    QLabel#AnalysisHintLabel {{
        color: {COLORS["text_muted"]};
        font-size: 9pt;
    }}
    QLabel#AnalysisInputDetail {{
        color: #b9c8d4;
        font-size: 9pt;
        padding: 2px 1px;
    }}
    QLabel#AnalysisValuePill {{
        background-color: {COLORS["input_bg"]};
        border: 1px solid {COLORS["input_border"]};
        border-radius: 6px;
        color: {COLORS["text"]};
        min-height: 26px;
        padding: 3px 8px;
    }}
    QWidget#AnalysisFrameView {{
        background-color: #0f151a;
        border: 1px solid {COLORS["panel_border_soft"]};
        border-radius: 6px;
    }}
    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
        background-color: {COLORS["input_bg"]};
        border: 1px solid {COLORS["input_border"]};
        border-radius: 6px;
        color: {COLORS["text"]};
        min-height: 26px;
        padding: 3px 8px;
        selection-background-color: {COLORS["accent"]};
    }}
    QPushButton {{
        background-color: {COLORS["button_bg"]};
        border: 1px solid {COLORS["panel_border_soft"]};
        border-radius: 6px;
        color: {COLORS["text"]};
        font-weight: 650;
        min-height: 28px;
        padding: 4px 10px;
    }}
    QPushButton:hover {{
        background-color: {COLORS["button_bg_hover"]};
        border-color: {COLORS["accent_hover"]};
    }}
    QPushButton:checked {{
        background-color: {COLORS["accent"]};
        border-color: {COLORS["accent"]};
        color: #ffffff;
    }}
    QPushButton:disabled {{
        color: {COLORS["text_subtle"]};
        background-color: {COLORS["button_bg_pressed"]};
        border-color: {COLORS["panel_border_soft"]};
    }}
    QCheckBox {{
        background: transparent;
        color: {COLORS["text"]};
        spacing: 6px;
    }}
    QCheckBox::indicator {{
        background-color: {COLORS["input_bg"]};
        border: 1px solid {COLORS["input_border"]};
        border-radius: 4px;
        height: 15px;
        width: 15px;
    }}
    QCheckBox::indicator:hover {{
        border-color: {COLORS["accent_hover"]};
    }}
    QCheckBox::indicator:checked {{
        background-color: {COLORS["accent"]};
        border-color: {COLORS["accent"]};
    }}
    QCheckBox::indicator:disabled {{
        background-color: {COLORS["button_bg_pressed"]};
        border-color: {COLORS["panel_border_soft"]};
    }}
    QListWidget#AnalysisRoiList {{
        background-color: {COLORS["terminal_bg"]};
        border: 1px solid #303941;
        border-radius: 6px;
        color: {COLORS["text"]};
        padding: 4px;
        selection-background-color: {COLORS["accent"]};
    }}
    QListWidget#VideoLibraryList {{
        background-color: {COLORS["terminal_bg"]};
        border: 1px solid #303941;
        border-radius: 6px;
        color: {COLORS["text"]};
        selection-background-color: {COLORS["accent"]};
        selection-color: #ffffff;
    }}
    QListWidget#VideoLibraryList::item {{
        padding: 3px 5px;
    }}
    QLabel#VideoLibrarySummary {{
        color: {COLORS["text_muted"]};
        font-weight: 600;
    }}
    QProgressBar {{
        background-color: {COLORS["terminal_bg"]};
        border: 1px solid #303941;
        border-radius: 5px;
        color: {COLORS["text"]};
        max-height: 12px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background-color: {COLORS["accent"]};
        border-radius: 4px;
    }}
    QPlainTextEdit#AnalysisSummaryView {{
        background-color: {COLORS["terminal_bg"]};
        color: #d8dee4;
        border: 1px solid #303941;
        border-radius: 6px;
        padding: 8px;
        selection-background-color: {COLORS["accent"]};
    }}
    QPlainTextEdit#AnalysisLogView {{
        background-color: {COLORS["terminal_bg"]};
        color: #d8dee4;
        border: 1px solid #303941;
        border-radius: 6px;
        padding: 8px;
        selection-background-color: {COLORS["accent"]};
    }}
    """


def inference_dialog_stylesheet() -> str:
    """Stylesheet for the project-aware inference video picker."""
    return f"""
    QLabel#InferencePickerTitle {{
        color: {COLORS["text"]};
        font-size: 17pt;
        font-weight: 800;
    }}
    QLabel#InferencePickerSubtitle {{
        color: {COLORS["text_muted"]};
        font-size: 10pt;
    }}
    QFrame#InferenceTrackingDefaults {{
        background-color: #20272d;
        border: 1px solid {COLORS["panel_border"]};
        border-radius: 7px;
    }}
    QLabel#InferenceTrackingTitle {{
        color: {COLORS["text"]};
        font-weight: 750;
    }}
    QLabel#InferenceTrackingHint,
    QLabel#InferenceSequentialHint {{
        color: {COLORS["text_muted"]};
        font-size: 9pt;
    }}
    QWidget#InferenceDefaultAnimals,
    QComboBox#InferenceDefaultTracker {{
        min-height: 30px;
    }}
    QPushButton#InferenceApplyDefaults {{
        min-height: 30px;
        min-width: 135px;
    }}
    QTableWidget#InferenceVideoTable {{
        background-color: {COLORS["terminal_bg"]};
        alternate-background-color: #11161a;
        border: 1px solid {COLORS["panel_border"]};
        border-radius: 7px;
        color: {COLORS["text"]};
    }}
    QTableWidget#InferenceVideoTable::item {{
        padding: 9px 8px;
        border-bottom: 1px solid #252d33;
    }}
    QTableWidget#InferenceVideoTable::item:hover {{
        background-color: #1f2c36;
        color: {COLORS["text"]};
    }}
    QTableWidget#InferenceVideoTable QHeaderView::section {{
        background-color: {COLORS["panel_bg_alt"]};
        color: {COLORS["text_muted"]};
        border: none;
        border-bottom: 1px solid {COLORS["panel_border"]};
        min-height: 34px;
        padding: 7px 8px;
        font-weight: 700;
    }}
    QTableWidget#InferenceVideoTable::indicator {{
        background-color: #11161a;
        border: 1px solid #687783;
        border-radius: 4px;
        height: 17px;
        width: 17px;
    }}
    QTableWidget#InferenceVideoTable::indicator:hover {{
        border-color: {COLORS["accent_hover"]};
    }}
    QTableWidget#InferenceVideoTable::indicator:checked {{
        background-color: {COLORS["accent"]};
        border-color: #8eb6db;
    }}
    QWidget#InferenceDefaultAnimals,
    QWidget#InferenceRowAnimals,
    QComboBox#InferenceRowTracker {{
        background-color: #171d22;
        border: 1px solid #53616c;
        border-radius: 6px;
        min-height: 30px;
    }}
    QWidget#InferenceRowAnimals,
    QComboBox#InferenceRowTracker {{
        margin: 5px 8px;
    }}
    QComboBox#InferenceRowTracker {{
        padding-left: 9px;
        padding-right: 8px;
    }}
    QWidget#InferenceDefaultAnimals:hover,
    QWidget#InferenceRowAnimals:hover,
    QComboBox#InferenceRowTracker:hover {{
        border-color: {COLORS["accent_hover"]};
    }}
    QLabel#AnimalCountValue {{
        color: {COLORS["text"]};
        font-weight: 700;
        min-width: 18px;
    }}
    QToolButton#AnimalCountButton {{
        background: transparent;
        border: none;
        border-radius: 4px;
        color: #c7e5ff;
        font-size: 13pt;
        font-weight: 700;
        min-height: 22px;
        min-width: 22px;
        max-height: 22px;
        max-width: 22px;
        padding: 0;
    }}
    QToolButton#AnimalCountButton:hover {{
        background-color: #304352;
        color: #ffffff;
    }}
    QToolButton#AnimalCountButton:disabled {{
        color: #56616b;
        background: transparent;
    }}
    QFrame#InferencePickerFooter {{
        background-color: {COLORS["panel_bg"]};
        border: 1px solid {COLORS["panel_border"]};
        border-radius: 7px;
    }}
    QLabel#InferenceSelectionSummary {{
        color: #c7e5ff;
        font-weight: 700;
    }}
    QPushButton#InferenceBrowseButton {{
        border-color: {COLORS["accent"]};
    }}
    QPushButton#InferenceRunButton {{
        background-color: #2f5f91;
        border-color: #6f9dca;
        min-width: 125px;
    }}
    """
