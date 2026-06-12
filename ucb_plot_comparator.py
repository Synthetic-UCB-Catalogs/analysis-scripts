#!/usr/bin/env python3
"""
UCB Plot Comparator v6
======================

A small cross-platform PySide6 desktop application for interactively comparing
plot images arranged in tuple-like folders, for example

    initial_condition_variations/fiducial/COMPAS @DWD formation.png
    mass_transfer_variations/common_envelope/alpha_lambda_02/BSE Galaxy only ...png

Each image is parsed into

    (*folder_tuple, code, subtype, variant)

Main features
-------------
- Choose the parent image folder from the GUI or command line.
- Compare Image 1 and Image 2 side by side.
- Each selector has left/right stepping buttons for every tuple element.
- Each side has Previous/Next buttons to quickly toggle through images.
- Five display modes: plot-to-plot, all codes, all figure types for one run,
  all deepest-level variations for one code/subtype, and all evolutionary stages.
- Retina/HiDPI-aware scaling to avoid blurred/smeared images on macOS.
- Compact two-column selector layout with click-to-focus highlighting.
- Optional first-page PDF rendering if PyMuPDF is installed.

Run
---
    python ucb_plot_comparator.py

or

    python ucb_plot_comparator.py ./SyntheticUCBs/Result/Code_comparison_plots

Dependencies
------------
    pip install PySide6

Optional PDF preview support:
    pip install pymupdf
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSettings, QSignalBlocker, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QCursor,
    QDesktopServices,
    QImage,
    QKeySequence,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

try:
    import fitz  # PyMuPDF, optional
except Exception:  # pragma: no cover - optional dependency
    fitz = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

KNOWN_CODES: tuple[str, ...] = (
    "ComBinE",
    "COMPAS",
    "COSMIC",
    "METISSE",
    "BPASS",
    "SEVN",
    "SeBa",
    "BSE",
)

RASTER_IMAGE_EXTS: set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PDF_EXTS: set[str] = {".pdf"}
IMAGE_EXTS: set[str] = RASTER_IMAGE_EXTS | PDF_EXTS

IGNORE_DIR_NAMES: set[str] = {
    "SAVE",
    "Old_versions",
    "Pilot runs",
    "__pycache__",
}

IMAGE_SUBTYPE_PREFIXES: list[str] = [
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
# Parsing helpers
# -----------------------------------------------------------------------------

def clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalise_channel(channel: str | None) -> str | None:
    if channel is None:
        return None

    raw = channel.strip()
    compact_upper = raw.upper().replace("_", "")

    if compact_upper == "CE1":
        return "CE_1"
    if compact_upper == "SMT1":
        return "SMT_1"
    if raw.upper() == "DCCE":
        return "DCCE"
    if raw.upper() == "STABLE":
        return "STABLE"
    if raw.lower() == "other":
        return "other"

    return raw


def stage_family_for_subtype(subtype: str) -> tuple[str, ...]:
    """Return the evolutionary-stage family containing subtype, if known."""
    for family in EVOLUTION_STAGE_FAMILIES:
        if subtype in family:
            return family
    return (subtype,)


def split_image_code_and_descriptor(
    stem: str,
    allowed_codes: Iterable[str] = KNOWN_CODES,
) -> tuple[str | None, str]:
    """
    Extract code and descriptor from a filename stem.

    Examples
    --------
    COMPAS @DWD formation CE_1
        -> code='COMPAS', descriptor='@DWD formation CE_1'

    M1-M2 normalised ComBinE <14 Gyr
        -> code='ComBinE', descriptor='M1-M2 normalised <14 Gyr'
    """
    stem = clean_spaces(stem)
    codes = sorted(set(allowed_codes), key=len, reverse=True)

    for code in codes:
        if stem == code:
            return code, ""
        if stem.startswith(code + " "):
            return code, clean_spaces(stem[len(code) :])

    for code in codes:
        pattern = re.compile(rf"(?<!\S){re.escape(code)}(?!\S)")
        match = pattern.search(stem)
        if match:
            descriptor = stem[: match.start()] + " " + stem[match.end() :]
            return code, clean_spaces(descriptor)

    return None, stem


def parse_image_descriptor(descriptor: str) -> tuple[str, str, str | None, int | None, bool]:
    """
    Return subtype, detail, channel, source_count, source_count_is_lisa.
    """
    raw = clean_spaces(descriptor)

    source_count: int | None = None
    source_count_is_lisa = False

    source_match = SOURCE_COUNT_RE.search(raw)
    if source_match:
        source_count = int(source_match.group("n").replace(",", ""))
        source_count_is_lisa = source_match.group("lisa") is not None

    channel_matches = list(CHANNEL_RE.finditer(raw))
    channel: str | None = None
    if channel_matches:
        channel = normalise_channel(channel_matches[-1].group(1))

    cleaned = SOURCE_COUNT_RE.sub(" ", raw)
    cleaned = CHANNEL_RE.sub(" ", cleaned)
    cleaned = clean_spaces(cleaned)

    subtype = cleaned or "<unknown subtype>"
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


# Evolutionary-stage grouping used by mode 5.
# The idea is: hold the figure kind fixed and vary only the stage token.
STAGE_SUBTYPE_GROUPS: dict[str, tuple[str, ...]] = {
    "formation": ("@ZAMS", "@WDMS formation", "@DWD formation"),
    "points": ("@ZAMS points", "@WDMS points", "@DWD points"),
    "points channels": ("@ZAMS points channels", "@WDMS points channels", "@DWD points channels"),
}

STAGE_SUBTYPE_TO_GROUP: dict[str, str] = {
    subtype: group
    for group, subtypes in STAGE_SUBTYPE_GROUPS.items()
    for subtype in subtypes
}


def stage_related_subtypes(subtype: str) -> tuple[str, ...]:
    """Return subtypes that differ only by evolutionary stage."""
    group = STAGE_SUBTYPE_TO_GROUP.get(subtype)
    if group is None:
        return (subtype,)
    return STAGE_SUBTYPE_GROUPS[group]


def stage_group_label(subtype: str) -> str:
    group = STAGE_SUBTYPE_TO_GROUP.get(subtype)
    return group if group is not None else subtype


# -----------------------------------------------------------------------------
# Image index
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageRecord:
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
        return (*self.folder_tuple, self.code, self.subtype)

    def exact_variant_label(self) -> str:
        parts: list[str] = []

        if self.channel:
            parts.append(self.channel)

        if self.source_count is not None:
            label = f"{self.source_count:,} "
            if self.source_count_is_lisa:
                label += "LISA "
            label += "sources"
            parts.append(label)

        if self.detail:
            parts.append(self.detail)

        if not parts:
            parts.append("plain")

        return " | ".join(parts) + f" — {self.path.name}"

    def variant_group_label(self) -> str:
        """
        Coarser label used in all-codes mode.

        Source counts are intentionally not included here because they differ
        by code and would prevent useful all-code grouping.
        """
        parts: list[str] = []
        if self.channel:
            parts.append(self.channel)
        if self.detail:
            parts.append(self.detail)
        return " | ".join(parts) if parts else "plain"

    def title_label(self) -> str:
        bits = [self.code, self.subtype]
        variant = self.variant_group_label()
        if variant != "plain":
            bits.append(variant)
        if self.source_count is not None:
            count = f"{self.source_count:,}"
            if self.source_count_is_lisa:
                count += " LISA"
            bits.append(count)
        return " | ".join(bits)


class ImageIndex:
    def __init__(self, records: list[ImageRecord], root: Path):
        self.root = root
        self.records = sorted(
            records,
            key=lambda r: (
                r.folder_tuple,
                r.code,
                r.subtype,
                r.variant_group_label(),
                r.source_count or -1,
                r.path.name,
            ),
        )

        self._by_folder: dict[tuple[str, ...], list[ImageRecord]] = defaultdict(list)
        self._by_folder_code: dict[tuple[tuple[str, ...], str], list[ImageRecord]] = defaultdict(list)
        self._by_folder_code_subtype: dict[
            tuple[tuple[str, ...], str, str], list[ImageRecord]
        ] = defaultdict(list)
        self._by_folder_subtype: dict[tuple[tuple[str, ...], str], list[ImageRecord]] = defaultdict(list)

        for record in self.records:
            self._by_folder[record.folder_tuple].append(record)
            self._by_folder_code[(record.folder_tuple, record.code)].append(record)
            self._by_folder_code_subtype[(record.folder_tuple, record.code, record.subtype)].append(record)
            self._by_folder_subtype[(record.folder_tuple, record.subtype)].append(record)

        self.folder_tuples = sorted(self._by_folder)
        self.folder_tuple_set = set(self.folder_tuples)
        self.max_folder_depth = max((len(t) for t in self.folder_tuples), default=0)

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        include_auxiliary: bool = False,
        allowed_codes: Iterable[str] = KNOWN_CODES,
        image_exts: Iterable[str] = IMAGE_EXTS,
    ) -> "ImageIndex":
        root = Path(root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Folder does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Not a folder: {root}")

        image_exts_clean = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in image_exts
        }
        records: list[ImageRecord] = []

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in image_exts_clean:
                continue

            rel = path.relative_to(root)

            if not include_auxiliary:
                if len(rel.parent.parts) == 0:
                    continue
                if any(part in IGNORE_DIR_NAMES for part in rel.parts):
                    continue

            code, raw_descriptor = split_image_code_and_descriptor(path.stem, allowed_codes)
            if code is None:
                continue

            subtype, detail, channel, source_count, source_count_is_lisa = parse_image_descriptor(
                raw_descriptor
            )

            records.append(
                ImageRecord(
                    folder_tuple=rel.parent.parts,
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

        return cls(records, root)

    def next_level_values(self, prefix: tuple[str, ...]) -> list[str]:
        n = len(prefix)
        return sorted(
            {
                folder_tuple[n]
                for folder_tuple in self.folder_tuples
                if len(folder_tuple) > n and folder_tuple[:n] == prefix
            }
        )

    def is_folder_tuple(self, folder_tuple: tuple[str, ...]) -> bool:
        return folder_tuple in self.folder_tuple_set

    def codes(self, folder_tuple: tuple[str, ...]) -> list[str]:
        return sorted({r.code for r in self._by_folder.get(folder_tuple, [])})

    def subtypes(self, folder_tuple: tuple[str, ...], code: str) -> list[str]:
        return sorted({r.subtype for r in self._by_folder_code.get((folder_tuple, code), [])})

    def subtypes_for_folder(self, folder_tuple: tuple[str, ...]) -> list[str]:
        return sorted({r.subtype for r in self._by_folder.get(folder_tuple, [])})

    def records_for(self, folder_tuple: tuple[str, ...], code: str, subtype: str) -> list[ImageRecord]:
        return sorted(
            self._by_folder_code_subtype.get((folder_tuple, code, subtype), []),
            key=lambda r: (r.variant_group_label(), r.source_count or -1, r.path.name),
        )

    def records_for_folder_subtype(
        self,
        folder_tuple: tuple[str, ...],
        subtype: str,
        *,
        variant_group: str | None = None,
    ) -> list[ImageRecord]:
        records = self._by_folder_subtype.get((folder_tuple, subtype), [])
        if variant_group and variant_group != "<all variants>":
            records = [r for r in records if r.variant_group_label() == variant_group]
        return sorted(
            records,
            key=lambda r: (r.code, r.variant_group_label(), r.source_count or -1, r.path.name),
        )

    def variant_groups_for_folder_subtype(self, folder_tuple: tuple[str, ...], subtype: str) -> list[str]:
        return sorted(
            {r.variant_group_label() for r in self._by_folder_subtype.get((folder_tuple, subtype), [])}
        )

    def records_for_folder_code(
        self,
        folder_tuple: tuple[str, ...],
        code: str,
        *,
        variant_group: str | None = None,
    ) -> list[ImageRecord]:
        """All figure subtypes/variants for one folder tuple and one code."""
        records = self._by_folder_code.get((folder_tuple, code), [])
        if variant_group and variant_group != "<all variants>":
            records = [r for r in records if r.variant_group_label() == variant_group]
        return sorted(
            records,
            key=lambda r: (r.subtype, r.variant_group_label(), r.source_count or -1, r.path.name),
        )

    def variant_groups_for_folder_code(self, folder_tuple: tuple[str, ...], code: str) -> list[str]:
        """Variant groups available for one folder tuple and one code."""
        return sorted(
            {r.variant_group_label() for r in self._by_folder_code.get((folder_tuple, code), [])}
        )

    def sibling_variation_folder_tuples(self, folder_tuple: tuple[str, ...]) -> list[tuple[str, ...]]:
        """
        Return folders that differ only in the deepest tuple element.

        Example:
            ("mass_transfer_variations", "common_envelope", "alpha_lambda_02")

        returns all third-level folders under

            ("mass_transfer_variations", "common_envelope")

        with the same tuple depth. This implements the "all variations" mode.
        """
        if not folder_tuple:
            return []
        prefix = folder_tuple[:-1]
        depth = len(folder_tuple)
        return sorted(
            t for t in self.folder_tuples
            if len(t) == depth and t[:-1] == prefix
        )

    def codes_for_variation_siblings(self, folder_tuple: tuple[str, ...]) -> list[str]:
        folders = self.sibling_variation_folder_tuples(folder_tuple)
        return sorted({r.code for f in folders for r in self._by_folder.get(f, [])})

    def subtypes_for_variation_siblings(
        self,
        folder_tuple: tuple[str, ...],
        code: str,
    ) -> list[str]:
        folders = self.sibling_variation_folder_tuples(folder_tuple)
        return sorted(
            {
                r.subtype
                for f in folders
                for r in self._by_folder_code.get((f, code), [])
            }
        )

    def records_for_variation_siblings(
        self,
        folder_tuple: tuple[str, ...],
        code: str,
        subtype: str,
        *,
        variant_group: str | None = None,
    ) -> list[ImageRecord]:
        """All matching records across sibling deepest-level variations."""
        folders = self.sibling_variation_folder_tuples(folder_tuple)
        records: list[ImageRecord] = []
        for f in folders:
            records.extend(self._by_folder_code_subtype.get((f, code, subtype), []))
        if variant_group and variant_group != "<all variants>":
            records = [r for r in records if r.variant_group_label() == variant_group]
        return sorted(
            records,
            key=lambda r: (r.folder_tuple, r.variant_group_label(), r.source_count or -1, r.path.name),
        )

    def variant_groups_for_variation_siblings(
        self,
        folder_tuple: tuple[str, ...],
        code: str,
        subtype: str,
    ) -> list[str]:
        folders = self.sibling_variation_folder_tuples(folder_tuple)
        return sorted(
            {
                r.variant_group_label()
                for f in folders
                for r in self._by_folder_code_subtype.get((f, code, subtype), [])
            }
        )

    # ------------------------------------------------------------------
    # Prefix-based variation helpers for mode 4.
    #
    # In mode 4 the user chooses everything *above* the deepest variation
    # level. For example:
    #
    #     initial_condition_variations / [all variations]
    #
    # or
    #
    #     mass_transfer_variations / common_envelope / [all variations]
    # ------------------------------------------------------------------

    def variation_parent_prefixes(self) -> set[tuple[str, ...]]:
        counts: dict[tuple[str, ...], int] = defaultdict(int)
        for folder_tuple in self.folder_tuples:
            if folder_tuple:
                counts[folder_tuple[:-1]] += 1
        return {prefix for prefix, count in counts.items() if count >= 2}

    def is_variation_parent_prefix(self, prefix: tuple[str, ...]) -> bool:
        return prefix in self.variation_parent_prefixes()

    def variation_folder_tuples_from_prefix(self, prefix: tuple[str, ...]) -> list[tuple[str, ...]]:
        depth = len(prefix) + 1
        return sorted(
            t for t in self.folder_tuples
            if len(t) == depth and t[:-1] == prefix
        )

    def codes_for_variation_prefix(self, prefix: tuple[str, ...]) -> list[str]:
        folders = self.variation_folder_tuples_from_prefix(prefix)
        return sorted({r.code for f in folders for r in self._by_folder.get(f, [])})

    def subtypes_for_variation_prefix(self, prefix: tuple[str, ...], code: str) -> list[str]:
        folders = self.variation_folder_tuples_from_prefix(prefix)
        return sorted(
            {
                r.subtype
                for f in folders
                for r in self._by_folder_code.get((f, code), [])
            }
        )

    def records_for_variation_prefix(
        self,
        prefix: tuple[str, ...],
        code: str,
        subtype: str,
        *,
        variant_group: str | None = None,
    ) -> list[ImageRecord]:
        folders = self.variation_folder_tuples_from_prefix(prefix)
        records: list[ImageRecord] = []
        for f in folders:
            records.extend(self._by_folder_code_subtype.get((f, code, subtype), []))
        if variant_group and variant_group != "<all variants>":
            records = [r for r in records if r.variant_group_label() == variant_group]
        return sorted(
            records,
            key=lambda r: (r.folder_tuple, r.variant_group_label(), r.source_count or -1, r.path.name),
        )

    def variant_groups_for_variation_prefix(
        self,
        prefix: tuple[str, ...],
        code: str,
        subtype: str,
    ) -> list[str]:
        folders = self.variation_folder_tuples_from_prefix(prefix)
        return sorted(
            {
                r.variant_group_label()
                for f in folders
                for r in self._by_folder_code_subtype.get((f, code, subtype), [])
            }
        )

    # ------------------------------------------------------------------
    # Evolutionary-stage helpers for mode 5.
    # ------------------------------------------------------------------

    def records_for_stage_family(
        self,
        folder_tuple: tuple[str, ...],
        code: str,
        subtype: str,
        *,
        variant_group: str | None = None,
    ) -> list[ImageRecord]:
        records: list[ImageRecord] = []
        for stage_subtype in stage_related_subtypes(subtype):
            records.extend(self._by_folder_code_subtype.get((folder_tuple, code, stage_subtype), []))
        if variant_group and variant_group != "<all variants>":
            records = [r for r in records if r.variant_group_label() == variant_group]
        return sorted(
            records,
            key=lambda r: (stage_related_subtypes(subtype).index(r.subtype) if r.subtype in stage_related_subtypes(subtype) else 999,
                           r.variant_group_label(), r.source_count or -1, r.path.name),
        )

    def variant_groups_for_stage_family(
        self,
        folder_tuple: tuple[str, ...],
        code: str,
        subtype: str,
    ) -> list[str]:
        records: list[ImageRecord] = []
        for stage_subtype in stage_related_subtypes(subtype):
            records.extend(self._by_folder_code_subtype.get((folder_tuple, code, stage_subtype), []))
        return sorted({r.variant_group_label() for r in records})


# -----------------------------------------------------------------------------
# Image loading and display
# -----------------------------------------------------------------------------

def pixmap_from_path(path: Path) -> QPixmap:
    """Load a raster image or, if possible, render page 1 of a PDF."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        if fitz is None:
            return QPixmap()
        try:
            doc = fitz.open(str(path))
            page = doc.load_page(0)
            zoom = 200 / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            fmt = QImage.Format.Format_RGB888
            image = QImage(pix.samples, pix.width, pix.height, pix.stride, fmt).copy()
            doc.close()
            return QPixmap.fromImage(image)
        except Exception:
            return QPixmap()

    return QPixmap(str(path))


class ImageDisplay(QWidget):
    """Single/grid image display with Retina-aware scaling and Ctrl-hover magnifier."""

    magnifierMoved = Signal(float, float, int)
    magnifierHidden = Signal()

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._records: list[ImageRecord] = []
        self._tiles: list[tuple[QLabel, QPixmap, ImageRecord]] = []
        self._label_to_tile: dict[QLabel, int] = {}
        self._pixmap_cache: dict[Path, QPixmap] = {}
        self._emitting_magnifier = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel(title)
        self.fit_checkbox = QCheckBox("Fit")
        self.fit_checkbox.setChecked(True)
        self.hidpi_checkbox = QCheckBox("Retina/HiDPI sharp")
        self.hidpi_checkbox.setChecked(True)
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(20, 250)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setEnabled(False)
        self.zoom_label = QLabel("100%")
        self.open_first_button = QPushButton("Open first")
        self.open_folder_button = QPushButton("Open folder")
        self.open_first_button.setEnabled(False)
        self.open_folder_button.setEnabled(False)

        controls.addWidget(self.title_label)
        controls.addStretch(1)
        controls.addWidget(self.fit_checkbox)
        controls.addWidget(self.hidpi_checkbox)
        controls.addWidget(QLabel("Zoom"))
        controls.addWidget(self.zoom_slider)
        controls.addWidget(self.zoom_label)
        controls.addWidget(self.open_first_button)
        controls.addWidget(self.open_folder_button)
        outer.addLayout(controls)

        self.path_line = QLineEdit()
        self.path_line.setReadOnly(True)
        self.path_line.setPlaceholderText("No image selected")
        outer.addWidget(self.path_line)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.content = QWidget()
        self.grid = QGridLayout(self.content)
        self.grid.setContentsMargins(8, 8, 8, 8)
        self.grid.setSpacing(10)
        self.scroll_area.setWidget(self.content)
        self.scroll_area.viewport().setMouseTracking(True)
        outer.addWidget(self.scroll_area, stretch=1)

        # Floating Ctrl-hover lens. It is a child of the viewport, so it overlays
        # the image area and does not steal layout space.
        self.lens_label = QLabel(self.scroll_area.viewport())
        self.lens_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lens_label.setFrameShape(QFrame.Shape.Box)
        self.lens_label.setStyleSheet(
            "background: rgba(255, 255, 255, 235); border: 2px solid #444;"
        )
        self.lens_label.resize(360, 260)
        self.lens_label.move(12, 12)
        self.lens_label.hide()

        self.fit_checkbox.toggled.connect(self._fit_toggled)
        self.hidpi_checkbox.toggled.connect(lambda _checked: self.schedule_scaled_pixmap_update())
        self.zoom_slider.valueChanged.connect(self._zoom_changed)
        self.open_first_button.clicked.connect(self.open_first)
        self.open_folder_button.clicked.connect(self.open_folder)

    def set_records(self, records: list[ImageRecord]) -> None:
        self._records = list(records)
        self._clear_grid()
        self._tiles.clear()
        self._label_to_tile.clear()
        self.lens_label.hide()

        self.open_first_button.setEnabled(bool(records))
        self.open_folder_button.setEnabled(bool(records))

        if not records:
            self.path_line.clear()
            msg = QLabel("No image selected")
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            msg.setFrameShape(QFrame.Shape.Box)
            msg.setMinimumSize(200, 200)
            self.grid.addWidget(msg, 0, 0)
            return

        if len(records) == 1:
            self.path_line.setText(str(records[0].path))
        else:
            folder = records[0].folder_tuple
            subtype = records[0].subtype
            self.path_line.setText(f"{len(records)} images | {'/'.join(folder)} | {subtype}")

        for i, record in enumerate(records):
            tile = QWidget()
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(0, 0, 0, 0)
            tile_layout.setSpacing(3)

            show_folder_context = len({r.folder_tuple for r in records}) > 1
            title = QLabel(self._record_title(record, show_folder_context))
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setWordWrap(True)
            title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

            image_label = QLabel("Loading…")
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setFrameShape(QFrame.Shape.Box)
            image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            image_label.setMinimumSize(80, 80)
            image_label.setMouseTracking(True)
            image_label.installEventFilter(self)

            pixmap = self._pixmap_cache.get(record.path)
            if pixmap is None:
                pixmap = pixmap_from_path(record.path)
                self._pixmap_cache[record.path] = pixmap

            if pixmap.isNull():
                image_label.setText(f"Could not load:\n{record.path.name}\n\nPDF preview needs: pip install pymupdf")
            else:
                image_label.setText("")

            tile_layout.addWidget(title, stretch=0)
            tile_layout.addWidget(image_label, stretch=1)

            row, col = self._grid_position(i, len(records))
            self.grid.addWidget(tile, row, col)
            self._label_to_tile[image_label] = i
            self._tiles.append((image_label, pixmap, record))

        self.content.updateGeometry()
        self.scroll_area.viewport().update()
        self.schedule_scaled_pixmap_update()

    def _record_title(self, record: ImageRecord, show_folder_context: bool) -> str:
        title = record.title_label()
        if not show_folder_context:
            return title

        # In all-variations mode, the key context is usually the deepest
        # folder component; include the full tuple as a tooltip-like second line
        # when that is clearer than only the leaf name.
        folder_leaf = record.folder_tuple[-1] if record.folder_tuple else "<root>"
        return f"{folder_leaf} | {title}"

    def _clear_grid(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _grid_columns(self, n: int) -> int:
        if n <= 1:
            return 1
        width = max(1, self.scroll_area.viewport().width())
        if width >= 1000:
            return 3
        if width >= 540:
            return 2
        return 1

    def _grid_position(self, index: int, n: int) -> tuple[int, int]:
        cols = self._grid_columns(n)
        return index // cols, index % cols

    def _device_pixel_ratio(self) -> float:
        if not self.hidpi_checkbox.isChecked():
            return 1.0

        window = self.window().windowHandle() if self.window() is not None else None
        screen = window.screen() if window is not None else QApplication.primaryScreen()
        if screen is None:
            return 1.0
        return max(1.0, float(screen.devicePixelRatio()))

    def _scale_pixmap(self, pixmap: QPixmap, logical_target: QSize) -> QPixmap:
        if pixmap.isNull():
            return pixmap

        dpr = self._device_pixel_ratio()
        target_physical = QSize(
            max(1, int(logical_target.width() * dpr)),
            max(1, int(logical_target.height() * dpr)),
        )
        scaled = pixmap.scaled(
            target_physical,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if self.hidpi_checkbox.isChecked():
            scaled.setDevicePixelRatio(dpr)
        return scaled

    def _tile_target_size(self, n: int) -> QSize:
        if n <= 0:
            return QSize(100, 100)

        viewport = self.scroll_area.viewport().size()
        if self.fit_checkbox.isChecked():
            if n == 1:
                return QSize(max(80, viewport.width() - 30), max(80, viewport.height() - 30))

            cols = self._grid_columns(n)
            width = max(160, int((viewport.width() - 30 - 12 * (cols - 1)) / cols))
            height = max(140, int(width * 0.78))
            return QSize(width, height)

        zoom = self.zoom_slider.value() / 100.0
        return QSize(max(80, int(700 * zoom)), max(80, int(520 * zoom)))

    def schedule_scaled_pixmap_update(self) -> None:
        """
        Refit images after Qt has assigned final widget sizes.

        On first window opening, especially on macOS/Retina displays and when
        launched with an initial folder argument, the first set_records() call can
        occur before the scroll-area viewport has its true size.  A few delayed
        passes make the initial display match what the user gets after nudging a
        selector left/right.
        """
        for delay_ms in (0, 30, 120, 300):
            QTimer.singleShot(delay_ms, self.update_scaled_pixmaps)

    def update_scaled_pixmaps(self) -> None:
        if not self._tiles:
            return

        # If Qt has not finished laying out the scroll area, do not commit a
        # bad initial scale.  Try again just after the event loop settles.
        vp = self.scroll_area.viewport().size()
        if vp.width() < 50 or vp.height() < 50:
            QTimer.singleShot(30, self.update_scaled_pixmaps)
            return

        n = len(self._tiles)
        target = self._tile_target_size(n)
        # For a single image, Fit should resize the content to the viewport.
        # For multi-image modes, keeping the content widget non-resizable prevents
        # Qt from vertically compressing many thumbnails into the viewport with no
        # useful scrollbar.
        self.scroll_area.setWidgetResizable(self.fit_checkbox.isChecked() and n == 1)

        # If the number of columns changes on resize, rebuild the layout positions.
        for i in reversed(range(self.grid.count())):
            item = self.grid.itemAt(i)
            widget = item.widget()
            if widget is not None:
                self.grid.removeWidget(widget)

        for i, (label, pixmap, _record) in enumerate(self._tiles):
            tile = label.parentWidget()
            row, col = self._grid_position(i, n)
            self.grid.addWidget(tile, row, col)

            if pixmap.isNull():
                continue
            scaled = self._scale_pixmap(pixmap, target)
            label.setPixmap(scaled)
            dpr = max(1.0, float(scaled.devicePixelRatio()))
            logical_size = QSize(
                max(1, int(scaled.width() / dpr)),
                max(1, int(scaled.height() / dpr)),
            )
            label.setFixedSize(logical_size)

        if n > 1 or not self.fit_checkbox.isChecked():
            self.content.adjustSize()
            self.content.setMinimumSize(self.grid.sizeHint())
        else:
            self.content.setMinimumSize(1, 1)

    def _relative_position_in_label(self, label: QLabel, pos: QPoint) -> tuple[float, float] | None:
        shown = label.pixmap()
        if shown is None or shown.isNull():
            return None

        dpr = max(1.0, float(shown.devicePixelRatio()))
        shown_w = shown.width() / dpr
        shown_h = shown.height() / dpr
        x0 = (label.width() - shown_w) / 2
        y0 = (label.height() - shown_h) / 2
        x = pos.x() - x0
        y = pos.y() - y0
        if x < 0 or y < 0 or x > shown_w or y > shown_h:
            return None
        return (max(0.0, min(1.0, x / shown_w)), max(0.0, min(1.0, y / shown_h)))

    def eventFilter(self, watched: QObject, event) -> bool:  # noqa: N802 - Qt method name
        if isinstance(watched, QLabel) and watched in self._label_to_tile:
            if event.type() == QEvent.Type.MouseMove:
                if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier:
                    watched.setCursor(Qt.CursorShape.CrossCursor)
                    pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
                    rel = self._relative_position_in_label(watched, pos)
                    if rel is not None:
                        tile_index = self._label_to_tile[watched]
                        self.show_magnifier(rel[0], rel[1], tile_index)
                        self.magnifierMoved.emit(rel[0], rel[1], tile_index)
                    return False
                watched.unsetCursor()
                self.hide_magnifier(emit=True)

            elif event.type() == QEvent.Type.Leave:
                watched.unsetCursor()
                self.hide_magnifier(emit=True)

            elif event.type() == QEvent.Type.Wheel:
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    delta = event.angleDelta().y()
                    if delta:
                        self.fit_checkbox.setChecked(False)
                        step = 10 if delta > 0 else -10
                        self.zoom_slider.setValue(
                            max(self.zoom_slider.minimum(), min(self.zoom_slider.maximum(), self.zoom_slider.value() + step))
                        )
                    event.accept()
                    return True

        return super().eventFilter(watched, event)

    def show_magnifier(self, rel_x: float, rel_y: float, tile_index: int = 0) -> None:
        if not self._tiles:
            self.hide_magnifier(emit=False)
            return

        tile_index = max(0, min(tile_index, len(self._tiles) - 1))
        _label, pixmap, record = self._tiles[tile_index]
        if pixmap.isNull():
            self.hide_magnifier(emit=False)
            return

        lens_w, lens_h = self.lens_label.width(), self.lens_label.height()
        aspect = lens_h / max(1, lens_w)
        crop_w = min(pixmap.width(), max(80, pixmap.width() // 4))
        crop_h = min(pixmap.height(), max(60, int(crop_w * aspect)))
        crop_w = min(crop_w, int(crop_h / aspect)) if aspect > 0 else crop_w

        cx = int(rel_x * pixmap.width())
        cy = int(rel_y * pixmap.height())
        x = max(0, min(pixmap.width() - crop_w, cx - crop_w // 2))
        y = max(0, min(pixmap.height() - crop_h, cy - crop_h // 2))

        crop = pixmap.copy(QRect(x, y, crop_w, crop_h))
        scaled = self._scale_pixmap(crop, QSize(lens_w - 8, lens_h - 8))
        self.lens_label.setPixmap(scaled)
        self.lens_label.setToolTip(f"Ctrl-hover lens: {record.path.name}")

        # Keep the lens visible but not too intrusive.
        vp = self.scroll_area.viewport()
        self.lens_label.move(max(4, vp.width() - lens_w - 16), 12)
        self.lens_label.show()
        self.lens_label.raise_()

    def hide_magnifier(self, *, emit: bool = False) -> None:
        if self.lens_label.isVisible():
            self.lens_label.hide()
            if emit:
                self.magnifierHidden.emit()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt method name
        super().resizeEvent(event)
        if self.fit_checkbox.isChecked():
            self.schedule_scaled_pixmap_update()

    def _fit_toggled(self, checked: bool) -> None:
        self.zoom_slider.setEnabled(not checked)
        # update_scaled_pixmaps decides whether a fitted multi-image grid should
        # still scroll instead of being compressed.
        self.schedule_scaled_pixmap_update()

    def _zoom_changed(self, value: int) -> None:
        self.zoom_label.setText(f"{value}%")
        if not self.fit_checkbox.isChecked():
            self.update_scaled_pixmaps()

    def open_first(self) -> None:
        if not self._records:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._records[0].path)))

    def open_folder(self) -> None:
        if not self._records:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._records[0].path.parent)))


# -----------------------------------------------------------------------------
# Selector panel
# -----------------------------------------------------------------------------

class ClickableSelectorLabel(QLabel):
    """Small QLabel that can be clicked to activate its selector."""

    clicked = Signal()

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.setToolTip("Click to make this selector respond to Left/Right keys")

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt method name
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class SteppableComboBox(QComboBox):
    """QComboBox where Left/Right step through options.

    Up/Down are intentionally ignored so only one keyboard convention changes
    selections. Clicking/focusing the combo emits selectorActivated so the panel
    can show which selector is currently controlled by Left/Right.
    """

    selectorActivated = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt method name
        key = event.key()
        if key == Qt.Key.Key_Left:
            self._step(-1)
            event.accept()
            return
        if key == Qt.Key.Key_Right:
            self._step(+1)
            event.accept()
            return
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            # Explicitly do nothing: no accidental selector changes with Up/Down.
            event.accept()
            return
        super().keyPressEvent(event)

    def focusInEvent(self, event) -> None:  # noqa: N802 - Qt method name
        self.selectorActivated.emit(self)
        super().focusInEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt method name
        self.selectorActivated.emit(self)
        super().mousePressEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt method name
        # Avoid accidental changes while scrolling the image grid; use explicit
        # Left/Right keys or the visible ◀/▶ buttons instead.
        event.ignore()

    def _step(self, delta: int) -> None:
        if not self.isEnabled() or self.count() <= 1:
            return
        self.setCurrentIndex((self.currentIndex() + delta) % self.count())


class ImageSelectorPanel(QWidget):
    STOP = "<STOP>"
    ALL_VARIANTS = "<all variants>"

    MODE_SINGLE = "single"
    MODE_ALL_CODES = "all_codes"
    MODE_ALL_TYPES = "all_types"
    MODE_ALL_VARIATIONS = "all_variations"
    MODE_ALL_STAGES = "all_stages"

    MODE_ITEMS: list[tuple[str, str]] = [
        ("1) Plot-to-plot", MODE_SINGLE),
        ("2) All codes for current selection", MODE_ALL_CODES),
        ("3) All figure types for this run", MODE_ALL_TYPES),
        ("4) All variations for this code/type", MODE_ALL_VARIATIONS),
        ("5) All evolutionary stages for this figure", MODE_ALL_STAGES),
    ]

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.index: ImageIndex | None = None
        self._updating = False
        self._active_combo: QComboBox | None = None

        self.level_labels: list[ClickableSelectorLabel] = []
        self.level_combos: list[SteppableComboBox] = []
        self.selector_labels: dict[QComboBox, ClickableSelectorLabel] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(3)

        self.prev_button = QPushButton("◀ Prev")
        self.next_button = QPushButton("Next ▶")
        self.prev_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.next_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.nav_widget = QWidget()
        nav_layout = QHBoxLayout(self.nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(3)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)

        self.mode_combo = SteppableComboBox()
        for label, value in self.MODE_ITEMS:
            self.mode_combo.addItem(label, value)
        self.mode_combo.setMinimumWidth(220)
        self.mode_combo.setToolTip(
            "Choose whether this side shows one image, all codes, all figure types, "
            "all deepest-level variations, or all evolutionary stages."
        )

        self.selector_box = QGroupBox(title)
        self.selector_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.selector_layout = QGridLayout(self.selector_box)
        self.selector_layout.setContentsMargins(5, 5, 5, 5)
        self.selector_layout.setHorizontalSpacing(4)
        self.selector_layout.setVerticalSpacing(2)
        self.selector_box.setMaximumHeight(132)
        self.selector_box.setStyleSheet(
            "QComboBox { min-height: 20px; max-height: 22px; } "
            "QComboBox[activeSelector=\"true\"] { border: 2px solid #3478f6; background: #eef4ff; } "
            "QToolButton { min-width: 17px; max-width: 17px; min-height: 17px; max-height: 17px; padding: 0px; } "
            "QPushButton { min-height: 20px; max-height: 23px; padding-left: 6px; padding-right: 6px; } "
            "QLabel { margin: 0px; } "
            "QLabel[activeSelector=\"true\"] { background: #dceaff; border-radius: 3px; font-weight: bold; }"
        )
        outer.addWidget(self.selector_box, stretch=0)

        self.code_combo = SteppableComboBox()
        self.subtype_combo = SteppableComboBox()
        self.variant_combo = SteppableComboBox()

        for combo in (self.mode_combo, self.code_combo, self.subtype_combo, self.variant_combo):
            combo.selectorActivated.connect(self.set_active_combo)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.code_combo.currentIndexChanged.connect(self._on_code_changed)
        self.subtype_combo.currentIndexChanged.connect(self._on_subtype_changed)
        self.variant_combo.currentIndexChanged.connect(self._on_variant_changed)
        self.prev_button.clicked.connect(lambda _checked=False: self.step_image(-1))
        self.next_button.clicked.connect(lambda _checked=False: self.step_image(+1))

        self.display = ImageDisplay(title="Preview")
        outer.addWidget(self.display, stretch=12)

        self.set_enabled(False)

    def mode(self) -> str:
        data = self.mode_combo.currentData()
        return str(data) if data is not None else self.MODE_SINGLE

    def set_enabled(self, enabled: bool) -> None:
        self.prev_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)
        self.mode_combo.setEnabled(enabled)
        self.selector_box.setEnabled(enabled)
        self.display.setEnabled(enabled)

    def clear_selector_layout(self) -> None:
        persistent = {
            self.mode_combo,
            self.code_combo,
            self.subtype_combo,
            self.variant_combo,
            self.prev_button,
            self.next_button,
            self.nav_widget,
        }
        while self.selector_layout.count():
            item = self.selector_layout.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue
            if widget in persistent:
                widget.setParent(self)
            else:
                widget.deleteLater()
        self.level_labels.clear()
        self.level_combos.clear()
        self.selector_labels.clear()
        self._active_combo = None

    def set_index(self, index: ImageIndex) -> None:
        self.index = index
        self.set_enabled(True)
        self.clear_selector_layout()

        # Dynamic folder selectors. The visual order below is:
        #   left column:  Mode, F0, F1, F2, ...
        #   right column: Code, Type, Var, then Previous/Next.
        for level in range(index.max_folder_depth):
            combo = SteppableComboBox()
            combo.selectorActivated.connect(self.set_active_combo)
            combo.currentIndexChanged.connect(
                lambda _idx, level=level: self._on_folder_level_changed(level)
            )
            label = ClickableSelectorLabel(f"F{level}")
            label.setToolTip(f"Folder tuple level {level}; click to activate; Left/Right to step")
            self.level_labels.append(label)
            self.level_combos.append(combo)

        left_items: list[tuple[str, QComboBox, str]] = [
            ("Mode", self.mode_combo, "Display mode"),
        ]
        left_items.extend(
            (f"F{level}", combo, f"Folder tuple level {level}")
            for level, combo in enumerate(self.level_combos)
        )

        right_items: list[tuple[str, QComboBox, str]] = [
            ("Code", self.code_combo, "Population-synthesis code"),
            ("Type", self.subtype_combo, "Figure subtype"),
            ("Var", self.variant_combo, "Variant/file group"),
        ]

        for row, (text, combo, tip) in enumerate(left_items):
            self._add_selector_item(row, 0, text, combo, tip)
        for row, (text, combo, tip) in enumerate(right_items):
            self._add_selector_item(row, 1, text, combo, tip)

        # Put Previous/Next in the grid instead of a separate header row.
        self._add_navigation_item(row=len(right_items), side=1)

        n_rows = max(len(left_items), len(right_items) + 1)
        self.selector_box.setMaximumHeight(28 + 25 * n_rows)

        self._refresh_folder_levels()
        self._refresh_codes()
        self.set_active_combo(self.mode_combo, set_focus=False)

    def _repolish(self, widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def set_active_combo(self, combo: object, set_focus: bool = False) -> None:
        """Make one selector visibly active for Left/Right keyboard stepping."""
        if not isinstance(combo, QComboBox):
            return
        self._active_combo = combo
        for candidate, label in self.selector_labels.items():
            active = candidate is combo
            candidate.setProperty("activeSelector", active)
            label.setProperty("activeSelector", active)
            self._repolish(candidate)
            self._repolish(label)
        if set_focus:
            combo.setFocus(Qt.FocusReason.MouseFocusReason)

    def _add_selector_item(
        self,
        row: int,
        side: int,
        label_text: str,
        combo: QComboBox,
        tooltip: str = "",
    ) -> None:
        """Add one compact labelled selector plus left/right step buttons."""
        group_col = 0 if side == 0 else 4

        label = ClickableSelectorLabel(label_text)
        label.setMinimumWidth(34)
        label.setToolTip(tooltip or "Click to activate this selector; use Left/Right to step.")
        label.clicked.connect(lambda c=combo: self.set_active_combo(c, set_focus=True))

        buttons = QWidget()
        buttons_layout = QHBoxLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(1)

        prev = QToolButton()
        next_ = QToolButton()
        prev.setText("◀")
        next_.setText("▶")
        prev.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        next_.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        prev.setToolTip("Previous option; Left key also works after activating the selector")
        next_.setToolTip("Next option; Right key also works after activating the selector")
        prev.clicked.connect(lambda _checked=False, c=combo: self.step_combo(c, -1))
        next_.clicked.connect(lambda _checked=False, c=combo: self.step_combo(c, +1))
        buttons_layout.addWidget(prev)
        buttons_layout.addWidget(next_)

        combo.setMinimumWidth(132 if side == 0 else 150)
        combo.setToolTip(tooltip)
        self.selector_layout.addWidget(label, row, group_col + 0)
        self.selector_layout.addWidget(combo, row, group_col + 1)
        self.selector_layout.addWidget(buttons, row, group_col + 2)
        self.selector_layout.setColumnStretch(group_col + 1, 1)
        self.selector_labels[combo] = label

    def _add_navigation_item(self, row: int, side: int) -> None:
        """Put Previous/Next in the selector grid, instead of a sparse header row."""
        group_col = 0 if side == 0 else 4
        label = QLabel("Image")
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        label.setToolTip("Step through the currently displayed image choice")

        self.selector_layout.addWidget(label, row, group_col + 0)
        self.selector_layout.addWidget(self.nav_widget, row, group_col + 1, 1, 2)

    def step_combo(self, combo: QComboBox, delta: int) -> None:
        self.set_active_combo(combo, set_focus=True)
        if not combo.isEnabled() or combo.count() <= 1:
            return
        new_index = (combo.currentIndex() + delta) % combo.count()
        combo.setCurrentIndex(new_index)

    def step_image(self, delta: int) -> None:
        """Quick stepping: variant first, then subtype, then code."""
        if self.variant_combo.isEnabled() and self.variant_combo.count() > 1:
            self.step_combo(self.variant_combo, delta)
            return
        if self.subtype_combo.isEnabled() and self.subtype_combo.count() > 1:
            self.step_combo(self.subtype_combo, delta)
            return
        if self.code_combo.isEnabled() and self.code_combo.count() > 1:
            self.step_combo(self.code_combo, delta)
            return

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt method name
        key = event.key()
        if key == Qt.Key.Key_Left:
            if self._active_combo is not None:
                self.step_combo(self._active_combo, -1)
                event.accept()
                return
        elif key == Qt.Key.Key_Right:
            if self._active_combo is not None:
                self.step_combo(self._active_combo, +1)
                event.accept()
                return
        elif key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            # Do nothing. Up/Down should not alter selectors in this version.
            event.accept()
            return
        super().keyPressEvent(event)

    def _set_combo_items(
        self,
        combo: QComboBox,
        items: list[tuple[str, object]],
        preferred_data: object | None = None,
    ) -> None:
        with QSignalBlocker(combo):
            combo.clear()
            for text, data in items:
                combo.addItem(text, data)

            if not items:
                combo.setEnabled(False)
                return

            combo.setEnabled(True)
            if preferred_data is not None:
                for i in range(combo.count()):
                    if combo.itemData(i) == preferred_data:
                        combo.setCurrentIndex(i)
                        return
            combo.setCurrentIndex(0)

    def _current_level_values(self) -> list[object | None]:
        return [combo.currentData() for combo in self.level_combos]

    def selected_folder_tuple(self) -> tuple[str, ...] | None:
        if self.index is None:
            return None

        prefix: list[str] = []
        for combo in self.level_combos:
            if not combo.isEnabled():
                break
            value = combo.currentData()
            if value == self.STOP or value is None:
                break
            prefix.append(str(value))

        folder_tuple = tuple(prefix)
        if self.index.is_folder_tuple(folder_tuple):
            return folder_tuple
        return None

    def selected_variation_prefix(self) -> tuple[str, ...] | None:
        """Prefix above the deepest variation level, used in mode 4."""
        if self.index is None:
            return None

        prefix: list[str] = []
        for combo in self.level_combos:
            if not combo.isEnabled():
                break
            value = combo.currentData()
            if value == self.STOP or value is None:
                break
            prefix.append(str(value))

        prefix_tuple = tuple(prefix)
        if self.index.is_variation_parent_prefix(prefix_tuple):
            return prefix_tuple
        return None

    def selected_context_folder_or_prefix(self) -> tuple[str, ...] | None:
        if self.mode() == self.MODE_ALL_VARIATIONS:
            return self.selected_variation_prefix()
        return self.selected_folder_tuple()

    def _refresh_folder_levels(self) -> None:
        if self.index is None:
            return

        old_values = self._current_level_values()
        prefix: tuple[str, ...] = ()
        mode = self.mode()
        self._updating = True
        try:
            stop_after: int | None = None

            for level, combo in enumerate(self.level_combos):
                # In mode 4, once the current prefix is the parent of a set
                # of deepest-level variations, do not ask the user to choose
                # the leaf. Show an explicit disabled placeholder instead.
                if mode == self.MODE_ALL_VARIATIONS and self.index.is_variation_parent_prefix(prefix):
                    self._set_combo_items(combo, [("— all variations —", self.STOP)])
                    combo.setEnabled(False)
                    stop_after = level
                    break

                next_values = self.index.next_level_values(prefix)
                can_stop = bool(prefix) and self.index.is_folder_tuple(prefix)

                items: list[tuple[str, object]] = []
                if can_stop and mode != self.MODE_ALL_VARIATIONS:
                    items.append(("— use this folder —", self.STOP))
                items.extend((value, value) for value in next_values)

                old_value = old_values[level] if level < len(old_values) else None
                valid_values = [data for _text, data in items]

                if old_value in valid_values:
                    preferred = old_value
                elif next_values:
                    preferred = next_values[0]
                elif can_stop and mode != self.MODE_ALL_VARIATIONS:
                    preferred = self.STOP
                else:
                    preferred = None

                self._set_combo_items(combo, items, preferred)

                if not items:
                    combo.setEnabled(False)
                    stop_after = level
                    break

                selected = combo.currentData()
                if selected == self.STOP or selected is None:
                    stop_after = level
                    break

                prefix = (*prefix, str(selected))

            if stop_after is not None:
                for level in range(stop_after + 1, len(self.level_combos)):
                    combo = self.level_combos[level]
                    self._set_combo_items(combo, [("—", None)])
                    combo.setEnabled(False)
        finally:
            self._updating = False

    def _refresh_codes(self) -> None:
        if self.index is None:
            return

        context = self.selected_context_folder_or_prefix()
        old_code = self.code_combo.currentData()
        mode = self.mode()

        if context is None:
            self._set_combo_items(self.code_combo, [])
            self._refresh_subtypes()
            return

        if mode == self.MODE_ALL_VARIATIONS:
            codes = self.index.codes_for_variation_prefix(context)
        else:
            codes = self.index.codes(context)

        self._set_combo_items(self.code_combo, [(code, code) for code in codes], old_code)
        self.code_combo.setEnabled(mode != self.MODE_ALL_CODES and bool(codes))
        self._refresh_subtypes()

    def _refresh_subtypes(self) -> None:
        if self.index is None:
            return

        context = self.selected_context_folder_or_prefix()
        old_subtype = self.subtype_combo.currentData()
        mode = self.mode()

        if context is None:
            self._set_combo_items(self.subtype_combo, [])
            self._refresh_variants()
            return

        if mode == self.MODE_ALL_TYPES:
            self._set_combo_items(self.subtype_combo, [("— all figure types —", None)])
            self.subtype_combo.setEnabled(False)
            self._refresh_variants()
            return

        if mode == self.MODE_ALL_CODES:
            # context is a concrete folder tuple here.
            subtypes = self.index.subtypes_for_folder(context)
        else:
            code = self.code_combo.currentData()
            if code is None:
                self._set_combo_items(self.subtype_combo, [])
                self._refresh_variants()
                return
            if mode == self.MODE_ALL_VARIATIONS:
                subtypes = self.index.subtypes_for_variation_prefix(context, str(code))
            else:
                subtypes = self.index.subtypes(context, str(code))

        self._set_combo_items(self.subtype_combo, [(s, s) for s in subtypes], old_subtype)
        self.subtype_combo.setEnabled(mode != self.MODE_ALL_TYPES and bool(subtypes))
        self._refresh_variants()

    def _refresh_variants(self) -> None:
        if self.index is None:
            return

        context = self.selected_context_folder_or_prefix()
        mode = self.mode()
        old_variant = self.variant_combo.currentData()

        if context is None:
            self._set_combo_items(self.variant_combo, [])
            self.display.set_records([])
            return

        items: list[tuple[str, object]] = []

        if mode == self.MODE_SINGLE:
            code = self.code_combo.currentData()
            subtype = self.subtype_combo.currentData()
            if code is None or subtype is None:
                self._set_combo_items(self.variant_combo, [])
                self.display.set_records([])
                return
            records = self.index.records_for(context, str(code), str(subtype))
            items = [(record.exact_variant_label(), str(record.path)) for record in records]

        elif mode == self.MODE_ALL_CODES:
            subtype = self.subtype_combo.currentData()
            if subtype is None:
                self._set_combo_items(self.variant_combo, [])
                self.display.set_records([])
                return
            groups = self.index.variant_groups_for_folder_subtype(context, str(subtype))
            if groups:
                items = [("All variants", self.ALL_VARIANTS)]
                items.extend((group, group) for group in groups)

        elif mode == self.MODE_ALL_TYPES:
            code = self.code_combo.currentData()
            if code is None:
                self._set_combo_items(self.variant_combo, [])
                self.display.set_records([])
                return
            groups = self.index.variant_groups_for_folder_code(context, str(code))
            if groups:
                items = [("All variants", self.ALL_VARIANTS)]
                items.extend((group, group) for group in groups)

        elif mode == self.MODE_ALL_VARIATIONS:
            code = self.code_combo.currentData()
            subtype = self.subtype_combo.currentData()
            if code is None or subtype is None:
                self._set_combo_items(self.variant_combo, [])
                self.display.set_records([])
                return
            groups = self.index.variant_groups_for_variation_prefix(
                context,
                str(code),
                str(subtype),
            )
            if groups:
                items = [("All variants", self.ALL_VARIANTS)]
                items.extend((group, group) for group in groups)

        elif mode == self.MODE_ALL_STAGES:
            code = self.code_combo.currentData()
            subtype = self.subtype_combo.currentData()
            if code is None or subtype is None:
                self._set_combo_items(self.variant_combo, [])
                self.display.set_records([])
                return
            groups = self.index.variant_groups_for_stage_family(context, str(code), str(subtype))
            if groups:
                items = [("All variants", self.ALL_VARIANTS)]
                items.extend((group, group) for group in groups)

        self._set_combo_items(self.variant_combo, items, old_variant)
        self._update_display()

    def _variant_group_from_combo(self) -> str | None:
        variant = self.variant_combo.currentData()
        if variant is None or variant == self.ALL_VARIANTS:
            return None
        return str(variant)

    def _update_display(self) -> None:
        if self.index is None:
            self.display.set_records([])
            return

        context = self.selected_context_folder_or_prefix()
        mode = self.mode()

        if context is None:
            self.display.set_records([])
            return

        if mode == self.MODE_SINGLE:
            path_text = self.variant_combo.currentData()
            if path_text is None:
                self.display.set_records([])
                return
            records = [r for r in self.index.records if str(r.path) == str(path_text)]
            self.display.set_records(records)
            return

        variant_group = self._variant_group_from_combo()

        if mode == self.MODE_ALL_CODES:
            subtype = self.subtype_combo.currentData()
            if subtype is None:
                self.display.set_records([])
                return
            records = self.index.records_for_folder_subtype(
                context,
                str(subtype),
                variant_group=variant_group,
            )
            self.display.set_records(records)
            return

        if mode == self.MODE_ALL_TYPES:
            code = self.code_combo.currentData()
            if code is None:
                self.display.set_records([])
                return
            records = self.index.records_for_folder_code(
                context,
                str(code),
                variant_group=variant_group,
            )
            self.display.set_records(records)
            return

        if mode == self.MODE_ALL_VARIATIONS:
            code = self.code_combo.currentData()
            subtype = self.subtype_combo.currentData()
            if code is None or subtype is None:
                self.display.set_records([])
                return
            records = self.index.records_for_variation_prefix(
                context,
                str(code),
                str(subtype),
                variant_group=variant_group,
            )
            self.display.set_records(records)
            return

        if mode == self.MODE_ALL_STAGES:
            code = self.code_combo.currentData()
            subtype = self.subtype_combo.currentData()
            if code is None or subtype is None:
                self.display.set_records([])
                return
            records = self.index.records_for_stage_family(
                context,
                str(code),
                str(subtype),
                variant_group=variant_group,
            )
            self.display.set_records(records)
            return

        self.display.set_records([])

    def _on_folder_level_changed(self, _level: int) -> None:
        if self._updating:
            return
        self._refresh_folder_levels()
        self._refresh_codes()

    def _on_mode_changed(self, _index: int | None = None) -> None:
        if self._updating:
            return
        self._refresh_folder_levels()
        self._refresh_codes()

    def _on_code_changed(self, _index: int | None = None) -> None:
        if self._updating:
            return
        self._refresh_subtypes()

    def _on_subtype_changed(self, _index: int | None = None) -> None:
        if self._updating:
            return
        self._refresh_variants()

    def _on_variant_changed(self, _index: int | None = None) -> None:
        if self._updating:
            return
        self._update_display()


# -----------------------------------------------------------------------------
# Main window
# -----------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, initial_folder: str | None = None):
        super().__init__()
        self.setWindowTitle("UCB Plot Comparator")
        self.resize(1650, 950)
        self.settings = QSettings("SyntheticUCBs", "UCBPlotComparator")
        self.index: ImageIndex | None = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        top = QHBoxLayout()
        self.folder_line = QLineEdit()
        self.folder_line.setPlaceholderText(
            "Select image parent folder, e.g. ./SyntheticUCBs/Result/Code_comparison_plots"
        )
        self.browse_button = QPushButton("Browse…")
        self.scan_button = QPushButton("Scan")
        self.include_aux_checkbox = QCheckBox("Include auxiliary/old folders")
        self.status_label = QLabel("No folder scanned")
        self.status_label.setMinimumWidth(320)

        top.addWidget(QLabel("Image parent folder"))
        top.addWidget(self.folder_line, stretch=1)
        top.addWidget(self.browse_button)
        top.addWidget(self.scan_button)
        top.addWidget(self.include_aux_checkbox)
        top.addWidget(self.status_label)
        main_layout.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.left_panel = ImageSelectorPanel("Image 1")
        self.right_panel = ImageSelectorPanel("Image 2")

        # Synchronized Ctrl-hover magnifier: moving over either side shows
        # the same relative crop on the other side where possible.
        self.left_panel.display.magnifierMoved.connect(
            lambda x, y, i: self.right_panel.display.show_magnifier(x, y, i)
        )
        self.right_panel.display.magnifierMoved.connect(
            lambda x, y, i: self.left_panel.display.show_magnifier(x, y, i)
        )
        self.left_panel.display.magnifierHidden.connect(
            lambda: self.right_panel.display.hide_magnifier(emit=False)
        )
        self.right_panel.display.magnifierHidden.connect(
            lambda: self.left_panel.display.hide_magnifier(emit=False)
        )

        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setSizes([825, 825])
        main_layout.addWidget(splitter, stretch=1)

        self.browse_button.clicked.connect(self.browse_folder)
        self.scan_button.clicked.connect(self.scan_folder)
        self.folder_line.returnPressed.connect(self.scan_folder)

        self._build_menu()

        folder = initial_folder or str(self.settings.value("last_folder", ""))
        if folder:
            self.folder_line.setText(folder)
            if initial_folder:
                # Defer the first scan until after the main window has been
                # shown and Qt has real widget geometry. This prevents the
                # initial images from being fitted to a stale pre-layout size.
                QTimer.singleShot(0, self.scan_folder)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802 - Qt method name
        if event.key() == Qt.Key.Key_Control:
            self.left_panel.display.hide_magnifier(emit=False)
            self.right_panel.display.hide_magnifier(emit=False)
        super().keyReleaseEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt method name
        super().showEvent(event)
        self.schedule_initial_refit()

    def schedule_initial_refit(self) -> None:
        """Force final fitted scaling after the main window becomes visible."""
        for delay_ms in (0, 80, 220, 450):
            QTimer.singleShot(delay_ms, self.refit_displays)

    def refit_displays(self) -> None:
        self.left_panel.display.schedule_scaled_pixmap_update()
        self.right_panel.display.schedule_scaled_pixmap_update()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        open_action = QAction("Open image parent folder…", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.browse_folder)
        file_menu.addAction(open_action)

        scan_action = QAction("Rescan", self)
        scan_action.setShortcut(QKeySequence("Ctrl+R"))
        scan_action.triggered.connect(self.scan_folder)
        file_menu.addAction(scan_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(lambda _checked=False: self.close())
        file_menu.addAction(quit_action)

    def browse_folder(self, _checked: bool = False) -> None:
        start = self.folder_line.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select image parent folder", start)
        if folder:
            self.folder_line.setText(folder)
            self.scan_folder()

    def scan_folder(self, _checked: bool = False) -> None:
        folder_text = self.folder_line.text().strip()
        if not folder_text:
            QMessageBox.warning(self, "No folder", "Please select an image parent folder first.")
            return

        try:
            index = ImageIndex.from_root(
                folder_text,
                include_auxiliary=self.include_aux_checkbox.isChecked(),
            )
        except Exception as exc:  # noqa: BLE001 - user-facing error dialog
            QMessageBox.critical(self, "Could not scan folder", str(exc))
            return

        if not index.records:
            QMessageBox.warning(
                self,
                "No images found",
                "No matching images were found.\n\n"
                "The app looks for image files whose filenames contain one of the known "
                "code names: COMPAS, BSE, COSMIC, METISSE, SEVN, SeBa, ComBinE, BPASS.",
            )
            return

        self.index = index
        self.left_panel.set_index(index)
        self.right_panel.set_index(index)
        self.settings.setValue("last_folder", str(index.root))

        n_images = len(index.records)
        n_folders = len(index.folder_tuples)
        n_tuples = len({r.image_tuple for r in index.records})
        self.status_label.setText(
            f"{n_images:,} images; {n_tuples:,} image tuples; {n_folders:,} folders"
        )
        self.schedule_initial_refit()


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    app.setApplicationName("UCB Plot Comparator")
    app.setOrganizationName("SyntheticUCBs")

    initial_folder = argv[1] if len(argv) > 1 else None
    window = MainWindow(initial_folder=initial_folder)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
