"""
Centralized error handling and logging.

Provides consistent error handling across the application.
"""

import logging
from typing import Optional, Callable
from enum import Enum
from PySide6.QtWidgets import QMessageBox


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorHandler:
    """
    Centralized error handler for consistent error management.
    
    Supports multiple output channels: logging, UI dialogs, and callbacks.
    """
    
    def __init__(self, parent_widget=None, enable_logging: bool = True):
        """
        Initialize error handler.
        
        Args:
            parent_widget: Parent Qt widget for message dialogs
            enable_logging: Whether to log errors to file/console
        """
        self.parent_widget = parent_widget
        self.enable_logging = enable_logging
        self.logger = logging.getLogger("SerialPlotter")
        
        # Setup logging if enabled
        if enable_logging:
            self._setup_logging()
        
        # Callbacks for custom error handling
        self.error_callbacks = []
    
    def _setup_logging(self):
        """Configure logging."""
        if not self.logger.handlers:
            handler = logging.FileHandler("serialplotter.log")
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
    
    def register_callback(self, callback: Callable):
        """Register callback for all errors."""
        self.error_callbacks.append(callback)
    
    def handle(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, 
               show_dialog: bool = True) -> None:
        """
        Handle an error.
        
        Args:
            message: Error message
            severity: Error severity level
            show_dialog: Whether to show UI dialog
        """
        # Log the error
        if self.enable_logging:
            if severity == ErrorSeverity.INFO:
                self.logger.info(message)
            elif severity == ErrorSeverity.WARNING:
                self.logger.warning(message)
            elif severity == ErrorSeverity.ERROR:
                self.logger.error(message)
            elif severity == ErrorSeverity.CRITICAL:
                self.logger.critical(message)
        
        # Show dialog if requested and widget available
        if show_dialog and self.parent_widget:
            self._show_dialog(message, severity)
        
        # Call registered callbacks
        for callback in self.error_callbacks:
            callback(message, severity)
    
    def _show_dialog(self, message: str, severity: ErrorSeverity) -> None:
        """Show error dialog."""
        if severity == ErrorSeverity.INFO:
            QMessageBox.information(self.parent_widget, "Information", message)
        elif severity == ErrorSeverity.WARNING:
            QMessageBox.warning(self.parent_widget, "Warning", message)
        elif severity == ErrorSeverity.ERROR:
            QMessageBox.critical(self.parent_widget, "Error", message)
        elif severity == ErrorSeverity.CRITICAL:
            QMessageBox.critical(self.parent_widget, "Critical Error", message)
    
    def info(self, message: str, show_dialog: bool = False) -> None:
        """Log info message."""
        self.handle(message, ErrorSeverity.INFO, show_dialog)
    
    def warning(self, message: str, show_dialog: bool = True) -> None:
        """Log warning message and show dialog."""
        self.handle(message, ErrorSeverity.WARNING, show_dialog)
    
    def error(self, message: str, show_dialog: bool = True) -> None:
        """Log error message and show dialog."""
        self.handle(message, ErrorSeverity.ERROR, show_dialog)
    
    def critical(self, message: str, show_dialog: bool = True) -> None:
        """Log critical error and show dialog."""
        self.handle(message, ErrorSeverity.CRITICAL, show_dialog)
