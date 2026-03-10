"""
Centralised error handling and logging.
"""

import logging
from typing import Callable
from enum import Enum
from PySide6.QtWidgets import QMessageBox


class ErrorSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorHandler:
    """Centralised error handler supporting logging, UI dialogs, and callbacks."""

    def __init__(self, parent_widget=None, enable_logging: bool = True):
        self.parent_widget = parent_widget
        self.enable_logging = enable_logging
        self.logger = logging.getLogger("SerialPlotter")
        self.error_callbacks = []

        if enable_logging:
            self._setup_logging()

    def _setup_logging(self) -> None:
        if not self.logger.handlers:
            handler = logging.FileHandler("serialplotter.log")
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    def register_callback(self, callback: Callable) -> None:
        self.error_callbacks.append(callback)

    def handle(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR,
               show_dialog: bool = True) -> None:
        if self.enable_logging:
            getattr(self.logger, severity.value)(message)

        if show_dialog and self.parent_widget:
            self._show_dialog(message, severity)

        for cb in self.error_callbacks:
            cb(message, severity)

    def _show_dialog(self, message: str, severity: ErrorSeverity) -> None:
        if severity == ErrorSeverity.INFO:
            QMessageBox.information(self.parent_widget, "Information", message)
        elif severity == ErrorSeverity.WARNING:
            QMessageBox.warning(self.parent_widget, "Warning", message)
        else:
            title = "Critical Error" if severity == ErrorSeverity.CRITICAL else "Error"
            QMessageBox.critical(self.parent_widget, title, message)

    def info(self, message: str, show_dialog: bool = False) -> None:
        self.handle(message, ErrorSeverity.INFO, show_dialog)

    def warning(self, message: str, show_dialog: bool = False) -> None:
        self.handle(message, ErrorSeverity.WARNING, show_dialog)

    def error(self, message: str, show_dialog: bool = True) -> None:
        self.handle(message, ErrorSeverity.ERROR, show_dialog)

    def critical(self, message: str, show_dialog: bool = True) -> None:
        self.handle(message, ErrorSeverity.CRITICAL, show_dialog)
