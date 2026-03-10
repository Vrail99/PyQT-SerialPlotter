"""
Widget module loader and selection dialog for user-provided QWidget plugins.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QWidget,
)


class WidgetModuleLoader:
    """Scans user_widgets and instantiates selected QWidget subclasses."""

    def __init__(self, widgets_dir: Path) -> None:
        self.widgets_dir = widgets_dir
        self._module_cache: Dict[str, ModuleType] = {}

    def discover_modules(self) -> List[str]:
        if not self.widgets_dir.exists():
            return []

        modules = [
            ".".join(path.relative_to(self.widgets_dir).with_suffix("").parts)
            for path in self.widgets_dir.rglob("*.py")
            if path.name != "__init__.py" and not path.name.startswith("_")
        ]
        return sorted(modules)

    def discover_widget_classes(self, module_name: str) -> List[str]:
        module = self._load_module(module_name)
        classes: List[str] = []

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if issubclass(obj, QWidget):
                classes.append(name)

        return sorted(classes)

    def instantiate(self, module_name: str, class_name: str, parent: Optional[QWidget] = None) -> QWidget:
        module = self._load_module(module_name)
        widget_cls = getattr(module, class_name, None)

        if not isinstance(widget_cls, type) or not issubclass(widget_cls, QWidget):
            raise TypeError(f"{class_name} is not a QWidget subclass")

        return widget_cls(parent=parent)

    def _load_module(self, module_name: str) -> ModuleType:
        if module_name in self._module_cache:
            return self._module_cache[module_name]

        module_path = self.widgets_dir / Path(*module_name.split(".")).with_suffix(".py")
        if not module_path.exists():
            raise FileNotFoundError(f"Module file not found: {module_path}")

        spec = importlib.util.spec_from_file_location(f"user_widgets.{module_name}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec for {module_name}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module_cache[module_name] = module
        return module


class WidgetLoaderDialog(QDialog):
    """Dropdown-based widget picker backed by WidgetModuleLoader."""

    def __init__(self, loader: WidgetModuleLoader, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.loader = loader

        self.setWindowTitle("Load External Widget")
        self.setModal(True)

        self.module_combo = QComboBox(self)
        self.class_combo = QComboBox(self)

        form = QFormLayout(self)
        form.addRow(QLabel("Module:"), self.module_combo)
        form.addRow(QLabel("Widget class:"), self.class_combo)

        buttons = (
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box = QDialogButtonBox(buttons, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        form.addRow(self.button_box)

        self.module_combo.currentTextChanged.connect(self._populate_classes)

        self._populate_modules()

    def _populate_modules(self) -> None:
        modules = self.loader.discover_modules()
        self.module_combo.clear()
        self.module_combo.addItems(modules)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(modules))

        if not modules:
            self.class_combo.clear()
            self.class_combo.addItem("No modules found")
            return

        self._populate_classes(self.module_combo.currentText())

    def _populate_classes(self, module_name: str) -> None:
        self.class_combo.clear()
        if not module_name:
            self.class_combo.addItem("No classes found")
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
            return

        try:
            class_names = self.loader.discover_widget_classes(module_name)
        except Exception:
            class_names = []

        if not class_names:
            self.class_combo.addItem("No QWidget classes found")
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
            return

        self.class_combo.addItems(class_names)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)

    def selected(self) -> Tuple[Optional[str], Optional[str]]:
        module_name = self.module_combo.currentText().strip()
        class_name = self.class_combo.currentText().strip()

        if not module_name or module_name.startswith("No "):
            return None, None
        if not class_name or class_name.startswith("No "):
            return None, None

        return module_name, class_name

    @classmethod
    def prompt_selection(
        cls,
        loader: WidgetModuleLoader,
        parent: Optional[QWidget] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        dialog = cls(loader, parent)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None, None
        return dialog.selected()

    @staticmethod
    def show_empty_hint(parent: Optional[QWidget] = None) -> None:
        QMessageBox.information(
            parent,
            "No widgets available",
            "No modules were found in the user_widgets directory.",
        )
