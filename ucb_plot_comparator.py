#!/usr/bin/env python3
"""
UCB plot/image comparator.

A small cross-platform desktop app for comparing two images from a nested
Synthetic UCB / code-comparison plot directory. It scans a parent image folder,
parses image filenames into a tuple-like identity, and gives independent
selectors for the left and right image.

Typical use:

    python ucb_plot_comparator.py ./SyntheticUCBs/Result/Code_comparison_plots

Dependencies:

    pip install PySide6

Optional PDF preview support:

    pip install pymupdf
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from collections import defaultdict
import re
import sys

try:
    import fitz  # PyMuPDF; optional, only needed for PDF preview.
except Exception:  # pragma: no cover - optional dependency
    fitz = None

from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import QPixmap, QImage, QAction, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


# -----------------------------------------------------------------------------
# Configuration: edit this block if your naming conventions evolve.
# -----------------------------------------------------------------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".pdf"}

# Known code names. Long names should appear before shorter names if there is
# possible overlap; the parser also sorts by length internally.
KNOWN_CODES = (
    "ComBinE",
    "COMPAS",
    "COSMIC",
    "METISSE",
    "BPASS",
    "SEVN",
    "SeBa",
    "BSE",
)

# These are not part of the clean one-to-one image/data hierarchy. They can still
# be included using the checkbox in the app.
AUXILIARY_DIR_NAMES = {"SAVE", "Old_versions", "Pilot runs", "Pilot_runs"}

NO_CODE = "(no code)"
END_OF_FOLDER_TUPLE = "__END_OF_FOLDER_TUPLE__"

# More specific subtype prefixes should appear before broader ones.
IMAGE_SUBTYPE_PREFIXES = [
    "@DWD points channels",
    "@WDMS points channels",
    "@ZAMS points channels",
    "@DWD formation",
    "@DWD points",
    "@WDMS formation",
    "@WDMS points",
    "@ZAMS points",
    "@ZAMS",
    "GW Galaxy all",
    "GW Galaxy only",
    "Galaxy all",
    "Galaxy only",
    "M1-M2 normalised",
    "Mass-radius",
    "R1-R2",
]

CHANNEL_RE = re.compile(r"\b(CE_?1|SMT_?1|DCCE|STABLE|other)\b", re.IGNORECASE)
SOURCE_COUNT_RE = re.compile(
    r"\b(?P<n>\d[\d,]*)\s+(?P<lisa>LISA\s+)?sources\b",
    re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# Parsing utilities
# -----------------------------------------------------------------------------

def clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def normalise_channel(channel: str | None) -> str | None:
    if channel is None:
        return None

    c = channel.strip()
    c_upper = c.upper().replace("_", "")

    if c_upper == "CE1":
        return "CE_1"
    if c_upper == "SMT1":
        return "SMT_1"
    if c.upper() == "DCCE":
        return "DCCE"
    if c.upper() == "STABLE":
        return "STABLE"
    if c.lower() == "other":
        return "other"

    return c


def split_image_code_and_descriptor(
    stem: str,
    allowed_codes: Iterable[str] = KNOWN_CODES,
) -> tuple[str | None, str]:
    """
    Extract the code name and the remaining image descriptor.

    Handles both common forms:

        COMPAS @DWD formation CE_1
        COMPAS GW Galaxy only 24,131 LISA sources

    and less standard forms:

        M1-M2 normalised COMPAS
        Mass-radius COMPAS revised
        R1-R2 ComBinE <14 Gyr
    """
    stem = clean_spaces(stem)
    codes = sorted(set(allowed_codes), key=len, reverse=True)

    # Preferred case: the filename starts with the code.
    for code in codes:
        if stem == code:
            return code, ""
        if stem.startswith(code + " "):
            return code, clean_spaces(stem[len(code) :])

    # Fallback: the code appears as a standalone token later in the name.
    for code in codes:
        pattern = re.compile(rf"(?<!\S){re.escape(code)}(?!\S)")
        match = pattern.search(stem)
        if match:
            descriptor = stem[: match.start()] + " " + stem[match.end() :]
            return code, clean_spaces(descriptor)

    return None, stem


def parse_image_descriptor(descriptor: str) -> tuple[str, str, str | None, int | None, bool]:
    """
    Return

        subtype, detail, channel, source_count, source_count_is_lisa

    Examples
    --------
    '@DWD formation CE_1'
        -> subtype='@DWD formation', detail='', channel='CE_1'

    'GW Galaxy only 24,131 LISA sources CE_1'
        -> subtype='GW Galaxy only', detail='', channel='CE_1',
           source_count=24131, source_count_is_lisa=True

    'Mass-radius revised'
        -> subtype='Mass-radius', detail='revised'
    """
    raw = clean_spaces(descriptor)

    source_count = None
    source_count_is_lisa = False
    source_match = SOURCE_COUNT_RE.search(raw)
    if source_match:
        source_count = int(source_match.group("n").replace(",", ""))
        source_count_is_lisa = source_match.group("lisa") is not None

    channel_matches = list(CHANNEL_RE.finditer(raw))
    channel = normalise_channel(channel_matches[-1].group(1)) if channel_matches else None

    # Remove less-important metadata before identifying the canonical subtype.
    cleaned = SOURCE_COUNT_RE.sub(" ", raw)
    cleaned = CHANNEL_RE.sub(" ", cleaned)
    cleaned = clean_spaces(cleaned)

    subtype = cleaned if cleaned else "(base)"
    detail = ""

    for prefix in IMAGE_SUBTYPE_PREFIXES:
        if cleaned == prefix:
            subtype = prefix
            detail = ""
            break
        if cleaned.startswith(prefix + " "):
            subtype = prefix
            detail = clean_spaces(cleaned[len(prefix) :])
            break

    return subtype, detail, channel, source_count, source_count_is_lisa


# -----------------------------------------------------------------------------
# Data model and scanner
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PlotImage:
    folder_tuple: tuple[str, ...]
    code: str
    subtype: str
    detail: str
    channel: str | None
    source_count: int | None
    source_count_is_lisa: bool
    raw_descriptor: str
    ext: str
    path: Path

    @property
    def image_tuple(self) -> tuple[str, ...]:
        """The main logical image tuple: (*folder_tuple, code, subtype)."""
        return (*self.folder_tuple, self.code, self.subtype)

    @property
    def variant_label(self) -> str:
        parts: list[str] = []

        if self.detail:
            parts.append(self.detail)

        if self.channel:
            parts.append(self.channel)

        if self.source_count is not None:
            label = f"{self.source_count:,} "
            if self.source_count_is_lisa:
                label += "LISA "
            label += "sources"
            parts.append(label)

        if not parts:
            parts.append("base")

        return " | ".join(parts)

    @property
    def full_variant_label(self) -> str:
        return f"{self.variant_label}  —  {self.path.name}"


def is_auxiliary_record(path: Path, root: Path, code: str | None) -> bool:
    rel_parent_parts = path.relative_to(root).parent.parts

    if len(rel_parent_parts) == 0:
        return True

    if any(part in AUXILIARY_DIR_NAMES for part in rel_parent_parts):
        return True

    if code is None:
        return True

    return False


def scan_plot_images(
    root: str | Path,
    *,
    include_auxiliary: bool = False,
    known_codes: Iterable[str] = KNOWN_CODES,
) -> list[PlotImage]:
    """
    Recursively scan the parent plot folder and return parsed image records.

    If include_auxiliary=False, skip top-level summary plots, Pilot runs, SAVE,
    Old_versions, and images whose filename cannot be associated with a known
    code.
    """
    root = Path(root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    records: list[PlotImage] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        folder_tuple = path.relative_to(root).parent.parts
        code, raw_descriptor = split_image_code_and_descriptor(path.stem, known_codes)
        auxiliary = is_auxiliary_record(path, root, code)

        if auxiliary and not include_auxiliary:
            continue

        if code is None:
            code = NO_CODE
            subtype = clean_spaces(path.stem)
            detail = ""
            channel = None
            source_count = None
            source_count_is_lisa = False
        else:
            subtype, detail, channel, source_count, source_count_is_lisa = parse_image_descriptor(
                raw_descriptor
            )

        records.append(
            PlotImage(
                folder_tuple=folder_tuple,
                code=code,
                subtype=subtype,
                detail=detail,
                channel=channel,
                source_count=source_count,
                source_count_is_lisa=source_count_is_lisa,
                raw_descriptor=raw_descriptor,
                ext=path.suffix.lower().lstrip("."),
                path=path,
            )
        )

    return records


# -----------------------------------------------------------------------------
# Image loading
# -----------------------------------------------------------------------------

def load_pixmap(path: Path) -> tuple[QPixmap | None, str | None]:
    """Load a raster image, or the first page of a PDF if PyMuPDF is installed."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        if fitz is None:
            return None, "PDF preview requires optional dependency: pip install pymupdf"

        try:
            doc = fitz.open(str(path))
            if len(doc) == 0:
                return None, "PDF has no pages."
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image.copy())
            doc.close()
            return pixmap, None
        except Exception as exc:  # pragma: no cover - depends on local file
            return None, f"Could not render PDF: {exc}"

    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        return None, f"Could not load image: {path.name}"

    return pixmap, None


# -----------------------------------------------------------------------------
# GUI widgets
# -----------------------------------------------------------------------------

class ImageView(QScrollArea):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setAlignment(Qt.AlignCenter)

        self.label = QLabel("No image selected")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setWidget(self.label)
        self.setWidgetResizable(True)

        self._path: Path | None = None
        self._pixmap: QPixmap | None = None
        self._fit_to_window = True

    def set_fit_to_window(self, fit: bool) -> None:
        self._fit_to_window = fit
        self.setWidgetResizable(fit)
        self._update_display()

    def set_path(self, path: Path | None) -> None:
        self._path = path
        self._pixmap = None

        if path is None:
            self.label.setText("No image selected")
            self.label.setPixmap(QPixmap())
            return

        pixmap, error = load_pixmap(path)
        if error is not None or pixmap is None:
            self.label.setPixmap(QPixmap())
            self.label.setText(f"{path.name}\n\n{error}")
            return

        self._pixmap = pixmap
        self._update_display()

    def resizeEvent(self, event):  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        if self._fit_to_window:
            self._update_display()

    def _update_display(self) -> None:
        if self._pixmap is None:
            return

        if self._fit_to_window:
            target_size = self.viewport().size()
            if target_size.width() <= 1 or target_size.height() <= 1:
                return
            dpr = self.devicePixelRatio()
            from PySide6.QtCore import QSize
            physical_size = QSize(
                int(target_size.width() * dpr),
                int(target_size.height() * dpr),
            )
            scaled = self._pixmap.scaled(
                physical_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            scaled.setDevicePixelRatio(dpr)
            self.label.setPixmap(scaled)
            self.label.resize(target_size)
        else:
            self.label.setPixmap(self._pixmap)
            self.label.resize(self._pixmap.size())


class SelectionPanel(QWidget):
    record_selected = Signal(object)  # emits PlotImage or None

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.title = title
        self.records: list[PlotImage] = []
        self.max_folder_depth = 0
        self.folder_combos: list[QComboBox] = []
        self.record_by_path: dict[str, PlotImage] = {}
        self._updating = False

        self.title_label = QLabel(f"<b>{title}</b>")

        self.selector_layout = QGridLayout()
        self.selector_layout.setColumnStretch(1, 1)

        self.code_combo = QComboBox()
        self.subtype_combo = QComboBox()
        self.variant_combo = QComboBox()

        self.code_combo.currentIndexChanged.connect(self.refresh_after_fixed_selectors)
        self.subtype_combo.currentIndexChanged.connect(self.refresh_after_fixed_selectors)
        self.variant_combo.currentIndexChanged.connect(self.emit_current_record)

        self.path_label = QLabel("No file selected")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.path_label.setWordWrap(True)

        self.open_button = QPushButton("Open")
        self.copy_button = QPushButton("Copy path")
        self.open_button.clicked.connect(self.open_selected)
        self.copy_button.clicked.connect(self.copy_selected_path)

        button_row = QHBoxLayout()
        button_row.addWidget(self.open_button)
        button_row.addWidget(self.copy_button)
        button_row.addStretch(1)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.title_label)
        main_layout.addLayout(self.selector_layout)
        main_layout.addLayout(button_row)
        main_layout.addWidget(self.path_label)

    def set_records(self, records: list[PlotImage]) -> None:
        self.records = list(records)
        self.record_by_path = {str(r.path): r for r in self.records}
        self.max_folder_depth = max((len(r.folder_tuple) for r in self.records), default=0)

        self._rebuild_folder_combos()
        self.refresh_all_selectors()

    def _rebuild_folder_combos(self) -> None:
        # Clear old folder selector rows and fixed selector rows.
        while self.selector_layout.count():
            item = self.selector_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        self.folder_combos = []

        row = 0
        for level in range(self.max_folder_depth):
            label = QLabel(f"Tuple level {level}")
            combo = QComboBox()
            combo.currentIndexChanged.connect(self.refresh_all_selectors)
            self.folder_combos.append(combo)
            self.selector_layout.addWidget(label, row, 0)
            self.selector_layout.addWidget(combo, row, 1)
            row += 1

        self.selector_layout.addWidget(QLabel("Code"), row, 0)
        self.selector_layout.addWidget(self.code_combo, row, 1)
        row += 1

        self.selector_layout.addWidget(QLabel("Image subtype"), row, 0)
        self.selector_layout.addWidget(self.subtype_combo, row, 1)
        row += 1

        self.selector_layout.addWidget(QLabel("Variant / file"), row, 0)
        self.selector_layout.addWidget(self.variant_combo, row, 1)

    @staticmethod
    def _set_combo_items(
        combo: QComboBox,
        items: list[object],
        display,
    ) -> None:
        old_data = combo.currentData()
        combo.blockSignals(True)
        combo.clear()

        for item in items:
            combo.addItem(display(item), item)

        new_index = 0
        for i, item in enumerate(items):
            if item == old_data:
                new_index = i
                break

        if items:
            combo.setCurrentIndex(new_index)

        combo.blockSignals(False)

    def _folder_display(self, item: object, level: int) -> str:
        if item == END_OF_FOLDER_TUPLE:
            return "(root)" if level == 0 else "(none)"
        return str(item)

    def _selected_folder_constraint(self) -> tuple[tuple[str, ...], bool]:
        """
        Return (prefix, exact).

        exact=True means the user selected '(none)' at the next level, or the
        selected folder has reached maximum depth.
        """
        prefix: list[str] = []
        exact = False

        for combo in self.folder_combos:
            data = combo.currentData()
            if data is None:
                break
            if data == END_OF_FOLDER_TUPLE:
                exact = True
                break
            prefix.append(str(data))

        if len(prefix) == self.max_folder_depth:
            exact = True

        return tuple(prefix), exact

    def _records_matching_folder_constraint(self) -> list[PlotImage]:
        prefix, exact = self._selected_folder_constraint()

        if exact:
            return [r for r in self.records if r.folder_tuple == prefix]

        return [r for r in self.records if r.folder_tuple[: len(prefix)] == prefix]

    def refresh_all_selectors(self) -> None:
        if self._updating:
            return

        self._updating = True

        # Rebuild folder-level options sequentially, preserving valid selections.
        for level, combo in enumerate(self.folder_combos):
            prefix: list[str] = []
            previous_exact = False

            for previous_combo in self.folder_combos[:level]:
                data = previous_combo.currentData()
                if data is None:
                    break
                if data == END_OF_FOLDER_TUPLE:
                    previous_exact = True
                    break
                prefix.append(str(data))

            if previous_exact:
                items: list[object] = [END_OF_FOLDER_TUPLE]
            else:
                options: set[object] = set()
                prefix_tuple = tuple(prefix)

                for record in self.records:
                    folder_tuple = record.folder_tuple
                    if folder_tuple[: len(prefix_tuple)] != prefix_tuple:
                        continue

                    if len(folder_tuple) == level:
                        options.add(END_OF_FOLDER_TUPLE)
                    elif len(folder_tuple) > level:
                        options.add(folder_tuple[level])

                def sort_key(x: object) -> tuple[int, str]:
                    if x == END_OF_FOLDER_TUPLE:
                        return (1, "")
                    return (0, str(x).lower())

                items = sorted(options, key=sort_key)
                if not items:
                    items = [END_OF_FOLDER_TUPLE]

            self._set_combo_items(combo, items, lambda item, level=level: self._folder_display(item, level))
            combo.setEnabled(not (len(items) == 1 and items[0] == END_OF_FOLDER_TUPLE and level > 0))

        self._updating = False
        self.refresh_after_fixed_selectors()

    def refresh_after_fixed_selectors(self) -> None:
        if self._updating:
            return

        self._updating = True

        folder_records = self._records_matching_folder_constraint()

        # Code selector.
        code_items = sorted({r.code for r in folder_records}, key=str.lower)
        self._set_combo_items(self.code_combo, code_items, str)
        selected_code = self.code_combo.currentData()

        code_records = [r for r in folder_records if r.code == selected_code] if selected_code else []

        # Subtype selector.
        subtype_items = sorted({r.subtype for r in code_records}, key=str.lower)
        self._set_combo_items(self.subtype_combo, subtype_items, str)
        selected_subtype = self.subtype_combo.currentData()

        subtype_records = [
            r for r in code_records if selected_subtype is not None and r.subtype == selected_subtype
        ]

        # Variant/file selector.
        subtype_records = sorted(
            subtype_records,
            key=lambda r: (
                r.channel or "",
                r.source_count if r.source_count is not None else -1,
                r.detail,
                r.path.name.lower(),
            ),
        )
        self._set_combo_items(
            self.variant_combo,
            [str(r.path) for r in subtype_records],
            lambda p: self.record_by_path[p].full_variant_label,
        )

        self._updating = False
        self.emit_current_record()

    def selected_record(self) -> PlotImage | None:
        path_key = self.variant_combo.currentData()
        if path_key is None:
            return None
        return self.record_by_path.get(str(path_key))

    def emit_current_record(self) -> None:
        if self._updating:
            return

        record = self.selected_record()
        if record is None:
            self.path_label.setText("No file selected")
            self.record_selected.emit(None)
            return

        self.path_label.setText(str(record.path))
        self.record_selected.emit(record)

    def open_selected(self) -> None:
        record = self.selected_record()
        if record is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(record.path)))

    def copy_selected_path(self) -> None:
        record = self.selected_record()
        if record is None:
            return
        QApplication.clipboard().setText(str(record.path))

    def set_selection_from_record(self, record: PlotImage) -> None:
        """Set this panel to a specific record, if it exists in this collection."""
        if str(record.path) not in self.record_by_path:
            return

        self._updating = True

        # Set folder levels.
        for level, combo in enumerate(self.folder_combos):
            desired = record.folder_tuple[level] if level < len(record.folder_tuple) else END_OF_FOLDER_TUPLE
            idx = combo.findData(desired)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        self._updating = False
        self.refresh_after_fixed_selectors()

        idx = self.code_combo.findData(record.code)
        if idx >= 0:
            self.code_combo.setCurrentIndex(idx)
        idx = self.subtype_combo.findData(record.subtype)
        if idx >= 0:
            self.subtype_combo.setCurrentIndex(idx)
        idx = self.variant_combo.findData(str(record.path))
        if idx >= 0:
            self.variant_combo.setCurrentIndex(idx)


class SideWidget(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.selector = SelectionPanel(title)
        self.view = ImageView()

        layout = QVBoxLayout(self)
        layout.addWidget(self.selector)
        layout.addWidget(self.view, stretch=1)

        self.selector.record_selected.connect(self._record_selected)

    def set_records(self, records: list[PlotImage]) -> None:
        self.selector.set_records(records)

    def set_fit_to_window(self, fit: bool) -> None:
        self.view.set_fit_to_window(fit)

    def _record_selected(self, record: PlotImage | None) -> None:
        self.view.set_path(record.path if record is not None else None)


class MainWindow(QMainWindow):
    def __init__(self, initial_root: Path | None = None):
        super().__init__()
        self.setWindowTitle("UCB plot comparator")
        self.resize(1500, 900)

        self.records: list[PlotImage] = []

        self.root_edit = QLineEdit()
        self.root_edit.setPlaceholderText("Choose the parent image folder, e.g. .../Code_comparison_plots")
        if initial_root is not None:
            self.root_edit.setText(str(initial_root.expanduser()))

        self.browse_button = QPushButton("Browse…")
        self.rescan_button = QPushButton("Scan")
        self.include_aux_checkbox = QCheckBox("Include auxiliary/summary folders")
        self.fit_checkbox = QCheckBox("Fit images to pane")
        self.fit_checkbox.setChecked(True)

        self.browse_button.clicked.connect(self.browse_for_folder)
        self.rescan_button.clicked.connect(self.scan_current_folder)
        self.include_aux_checkbox.stateChanged.connect(self.scan_current_folder)
        self.fit_checkbox.stateChanged.connect(self.update_fit_mode)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Image parent folder:"))
        top_row.addWidget(self.root_edit, stretch=1)
        top_row.addWidget(self.browse_button)
        top_row.addWidget(self.rescan_button)
        top_row.addWidget(self.include_aux_checkbox)
        top_row.addWidget(self.fit_checkbox)

        self.left = SideWidget("Image 1")
        self.right = SideWidget("Image 2")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.left)
        splitter.addWidget(self.right)
        splitter.setSizes([750, 750])

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.addLayout(top_row)
        central_layout.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self._create_menu()

        if initial_root is not None:
            self.scan_current_folder()

    def _create_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        open_action = QAction("Choose image folder…", self)
        open_action.triggered.connect(self.browse_for_folder)
        file_menu.addAction(open_action)

        rescan_action = QAction("Rescan", self)
        rescan_action.triggered.connect(self.scan_current_folder)
        file_menu.addAction(rescan_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def browse_for_folder(self) -> None:
        start_dir = self.root_edit.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Choose parent image folder", start_dir)
        if folder:
            self.root_edit.setText(folder)
            self.scan_current_folder()

    def scan_current_folder(self) -> None:
        root_text = self.root_edit.text().strip()
        if not root_text:
            return

        root = Path(root_text).expanduser()
        include_aux = self.include_aux_checkbox.isChecked()

        try:
            records = scan_plot_images(root, include_auxiliary=include_aux)
        except Exception as exc:
            QMessageBox.critical(self, "Could not scan folder", str(exc))
            return

        self.records = records
        self.left.set_records(records)
        self.right.set_records(records)
        self.update_fit_mode()

        folder_tuples = {r.folder_tuple for r in records}
        image_tuples = {r.image_tuple for r in records}
        self.status.showMessage(
            f"Loaded {len(records):,} images from {len(folder_tuples):,} folders "
            f"and {len(image_tuples):,} image tuples."
        )

        if not records:
            QMessageBox.information(
                self,
                "No images found",
                "No matching images were found. Try enabling 'Include auxiliary/summary folders', "
                "or check that the chosen folder is the plot parent folder.",
            )

    def update_fit_mode(self) -> None:
        fit = self.fit_checkbox.isChecked()
        self.left.set_fit_to_window(fit)
        self.right.set_fit_to_window(fit)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> int:
    app = QApplication(sys.argv)

    initial_root = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    window = MainWindow(initial_root=initial_root)
    window.show()

    if initial_root is None:
        window.status.showMessage("Choose the parent image folder to begin.")

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
