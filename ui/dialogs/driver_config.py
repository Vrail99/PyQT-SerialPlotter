"""
Driver configuration dialog: an interactive window for hardware driver commands.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QTextEdit, QGridLayout, QMessageBox, QLineEdit,
)
from PySide6.QtCore import Qt


class DriverConfigDialog(QDialog):
    """Modal dialog for inspecting driver state and sending commands."""

    def __init__(self, driver_manager, parent=None) -> None:
        super().__init__(parent)
        self.driver_manager = driver_manager
        self.current_driver = driver_manager.get_current_driver()

        self.setWindowTitle("Hardware Driver Configuration")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self._setup_ui()
        self._update_device_info()

    # ─── UI construction ──────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        layout.addWidget(self._create_device_info_group())
        layout.addWidget(self._create_quick_commands_group())
        layout.addWidget(self._create_custom_command_group())
        layout.addWidget(self._create_command_history_group())

        close_layout = QHBoxLayout()
        close_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_layout.addWidget(close_btn)
        layout.addLayout(close_layout)

        self.setLayout(layout)

    def _create_device_info_group(self) -> QGroupBox:
        group = QGroupBox("Device Information")
        layout = QGridLayout()
        for row, label in enumerate(["Profile:", "Driver:", "Port:", "Baudrate:", "Status:"]):
            layout.addWidget(QLabel(label), row, 0)
        self.profile_label  = QLabel("Not connected")
        self.driver_label   = QLabel("-")
        self.port_label     = QLabel("-")
        self.baudrate_label = QLabel("-")
        self.status_label   = QLabel("Disconnected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        for row, widget in enumerate([self.profile_label, self.driver_label,
                                       self.port_label, self.baudrate_label, self.status_label]):
            layout.addWidget(widget, row, 1)
        refresh_btn = QPushButton("Refresh Info")
        refresh_btn.clicked.connect(self._update_device_info)
        layout.addWidget(refresh_btn, 5, 0, 1, 2)
        group.setLayout(layout)
        return group

    def _create_quick_commands_group(self) -> QGroupBox:
        group = QGroupBox("Quick Commands")
        layout = QGridLayout()
        self.command_buttons = {}

        if self.current_driver and self.current_driver.profile.commands:
            row = col = 0
            for cmd_name, cmd_string in self.current_driver.profile.commands.items():
                btn = QPushButton(cmd_name.replace("_", " ").title())
                btn.setToolTip(f"Send: {cmd_string}")
                btn.clicked.connect(
                    lambda checked, n=cmd_name, c=cmd_string: self._send_quick_command(n, c)
                )
                layout.addWidget(btn, row, col)
                self.command_buttons[cmd_name] = btn
                col += 1
                if col >= 3:
                    col = 0
                    row += 1
        else:
            lbl = QLabel("No commands defined in profile")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl, 0, 0, 1, 3)

        group.setLayout(layout)
        return group

    def _create_custom_command_group(self) -> QGroupBox:
        group = QGroupBox("Custom Command")
        layout = QHBoxLayout()
        self.custom_command_input = QLineEdit()
        self.custom_command_input.setPlaceholderText("Enter custom command...")
        self.custom_command_input.returnPressed.connect(self._send_custom_command)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._send_custom_command)
        layout.addWidget(self.custom_command_input)
        layout.addWidget(send_btn)
        group.setLayout(layout)
        return group

    def _create_command_history_group(self) -> QGroupBox:
        group = QGroupBox("Command History")
        layout = QVBoxLayout()
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setMaximumHeight(200)
        clear_layout = QHBoxLayout()
        clear_layout.addStretch()
        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(self.history_text.clear)
        clear_layout.addWidget(clear_btn)
        layout.addWidget(self.history_text)
        layout.addLayout(clear_layout)
        group.setLayout(layout)
        return group

    # ─── Device info ──────────────────────────────────────────────────────

    def _update_device_info(self) -> None:
        self.current_driver = self.driver_manager.get_current_driver()
        connected = bool(self.current_driver and self.current_driver.is_connected)

        if not self.current_driver:
            self.profile_label.setText("Not connected")
            self.driver_label.setText("-")
            self.port_label.setText("-")
            self.baudrate_label.setText("-")
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            info = self.current_driver.get_device_info()
            self.profile_label.setText(self.current_driver.profile.name)
            self.driver_label.setText(info.get("driver", "-"))
            self.port_label.setText(info.get("port", "-"))
            self.baudrate_label.setText(str(self.current_driver.profile.baudrate))
            if connected:
                self.status_label.setText("Connected")
                self.status_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.status_label.setText("Disconnected")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")

        for btn in self.command_buttons.values():
            btn.setEnabled(connected)

    def refresh(self) -> None:
        self._update_device_info()

    # ─── Command helpers ──────────────────────────────────────────────────

    def _send_quick_command(self, cmd_name: str, cmd_string: str) -> None:
        if not self.current_driver or not self.current_driver.is_connected:
            QMessageBox.warning(self, "Error", "Device not connected")
            return
        try:
            self._log_command(f"{cmd_name}: {cmd_string}")
            response = self.current_driver.write_command(cmd_string)
            self._log_response(response or "(No response)")
        except Exception as e:
            self._log_error(str(e))
            QMessageBox.warning(self, "Error", f"Failed to send command: {e}")

    def _send_custom_command(self) -> None:
        if not self.current_driver or not self.current_driver.is_connected:
            QMessageBox.warning(self, "Error", "Device not connected")
            return
        command = self.custom_command_input.text().strip()
        if not command:
            return
        try:
            self._log_command(f"Custom: {command}")
            response = self.current_driver.write_command(command)
            self._log_response(response or "(No response)")
            self.custom_command_input.clear()
        except Exception as e:
            self._log_error(str(e))
            QMessageBox.warning(self, "Error", f"Failed to send command: {e}")

    def _log_command(self, text: str) -> None:
        self.history_text.append(f"<b style='color: blue;'>→ {text}</b>")
        self._scroll_history()

    def _log_response(self, text: str) -> None:
        self.history_text.append(f"<span style='color: green;'>← {text}</span>")
        self._scroll_history()

    def _log_error(self, text: str) -> None:
        self.history_text.append(f"<b style='color: red;'>✗ ERROR: {text}</b>")
        self._scroll_history()

    def _scroll_history(self) -> None:
        sb = self.history_text.verticalScrollBar()
        sb.setValue(sb.maximum())
