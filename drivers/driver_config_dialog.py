"""
Driver Configuration Dialog - Separate window for hardware driver interaction.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QTextEdit, QGridLayout, QMessageBox, QLineEdit
)
from PySide6.QtCore import Qt
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DriverConfigDialog(QDialog):
    """Configuration dialog for hardware driver interaction."""
    
    def __init__(self, driver_manager, parent=None):
        super().__init__(parent)
        
        self.driver_manager = driver_manager
        self.current_driver = driver_manager.get_current_driver()
        
        self.setWindowTitle("Hardware Driver Configuration")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        self._setup_ui()
        self._update_device_info()
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout()
        
        layout.addWidget(self._create_device_info_group())
        layout.addWidget(self._create_quick_commands_group())
        layout.addWidget(self._create_custom_command_group())
        layout.addWidget(self._create_command_history_group())
        
        # Close Button
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_layout.addWidget(close_button)
        layout.addLayout(close_layout)
        
        self.setLayout(layout)
    
    def _create_device_info_group(self) -> QGroupBox:
        """Create device information display."""
        group = QGroupBox("Device Information")
        layout = QGridLayout()
        
        layout.addWidget(QLabel("Profile:"), 0, 0)
        self.profile_label = QLabel("Not connected")
        layout.addWidget(self.profile_label, 0, 1)
        
        layout.addWidget(QLabel("Driver:"), 1, 0)
        self.driver_label = QLabel("-")
        layout.addWidget(self.driver_label, 1, 1)
        
        layout.addWidget(QLabel("Port:"), 2, 0)
        self.port_label = QLabel("-")
        layout.addWidget(self.port_label, 2, 1)
        
        layout.addWidget(QLabel("Baudrate:"), 3, 0)
        self.baudrate_label = QLabel("-")
        layout.addWidget(self.baudrate_label, 3, 1)
        
        layout.addWidget(QLabel("Status:"), 4, 0)
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(self.status_label, 4, 1)
        
        refresh_button = QPushButton("Refresh Info")
        refresh_button.clicked.connect(self._update_device_info)
        layout.addWidget(refresh_button, 5, 0, 1, 2)
        
        group.setLayout(layout)
        return group
    
    def _create_quick_commands_group(self) -> QGroupBox:
        """Create quick command buttons from profile."""
        group = QGroupBox("Quick Commands")
        layout = QGridLayout()
        
        self.command_buttons = {}
        
        if self.current_driver and self.current_driver.profile.commands:
            commands = self.current_driver.profile.commands
            
            # Create buttons in a grid (3 columns)
            row, col = 0, 0
            for cmd_name, cmd_string in commands.items():
                button = QPushButton(cmd_name.replace('_', ' ').title())
                button.setToolTip(f"Send: {cmd_string}")
                button.clicked.connect(
                    lambda checked, cmd=cmd_string, name=cmd_name: 
                    self._send_quick_command(name, cmd)
                )
                
                layout.addWidget(button, row, col)
                self.command_buttons[cmd_name] = button
                
                col += 1
                if col >= 3:
                    col = 0
                    row += 1
        else:
            label = QLabel("No commands defined in profile")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label, 0, 0, 1, 3)
        
        group.setLayout(layout)
        return group
    
    def _create_custom_command_group(self) -> QGroupBox:
        """Create custom command input."""
        group = QGroupBox("Custom Command")
        layout = QHBoxLayout()
        
        self.custom_command_input = QLineEdit()
        self.custom_command_input.setPlaceholderText("Enter custom command...")
        self.custom_command_input.returnPressed.connect(self._send_custom_command)
        
        send_button = QPushButton("Send")
        send_button.clicked.connect(self._send_custom_command)
        
        layout.addWidget(self.custom_command_input)
        layout.addWidget(send_button)
        
        group.setLayout(layout)
        return group
    
    def _create_command_history_group(self) -> QGroupBox:
        """Create command history display."""
        group = QGroupBox("Command History")
        layout = QVBoxLayout()
        
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setMaximumHeight(200)
        
        clear_layout = QHBoxLayout()
        clear_layout.addStretch()
        clear_button = QPushButton("Clear History")
        clear_button.clicked.connect(self._clear_history)
        clear_layout.addWidget(clear_button)
        
        layout.addWidget(self.history_text)
        layout.addLayout(clear_layout)
        
        group.setLayout(layout)
        return group
    
    def _update_device_info(self):
        """Update device information display."""
        self.current_driver = self.driver_manager.get_current_driver()
        
        if not self.current_driver:
            self.profile_label.setText("Not connected")
            self.driver_label.setText("-")
            self.port_label.setText("-")
            self.baudrate_label.setText("-")
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            
            for button in self.command_buttons.values():
                button.setEnabled(False)
            return
        
        info = self.current_driver.get_device_info()
        
        self.profile_label.setText(self.current_driver.profile.name)
        self.driver_label.setText(info.get('driver', '-'))
        self.port_label.setText(info.get('port', '-'))
        self.baudrate_label.setText(str(self.current_driver.profile.baudrate))
        
        if self.current_driver.is_connected:
            self.status_label.setText("Connected")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            for button in self.command_buttons.values():
                button.setEnabled(True)
        else:
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            for button in self.command_buttons.values():
                button.setEnabled(False)
    
    def _send_quick_command(self, cmd_name: str, cmd_string: str):
        """Send a predefined quick command."""
        if not self.current_driver or not self.current_driver.is_connected:
            QMessageBox.warning(self, "Error", "Device not connected")
            return
        
        try:
            self._log_command(f"{cmd_name}: {cmd_string}")
            response = self.current_driver.write_command(cmd_string)
            
            if response:
                self._log_response(response)
            else:
                self._log_response("(No response)")
                
        except Exception as e:
            self._log_error(str(e))
            QMessageBox.warning(self, "Error", f"Failed to send command: {e}")
    
    def _send_custom_command(self):
        """Send custom command from input field."""
        if not self.current_driver or not self.current_driver.is_connected:
            QMessageBox.warning(self, "Error", "Device not connected")
            return
        
        command = self.custom_command_input.text().strip()
        if not command:
            return
        
        try:
            self._log_command(f"Custom: {command}")
            response = self.current_driver.write_command(command)
            
            if response:
                self._log_response(response)
            else:
                self._log_response("(No response)")
            
            self.custom_command_input.clear()
            
        except Exception as e:
            self._log_error(str(e))
            QMessageBox.warning(self, "Error", f"Failed to send command: {e}")
    
    def _log_command(self, command: str):
        """Log command to history."""
        self.history_text.append(f"<b style='color: blue;'>→ {command}</b>")
        self.history_text.verticalScrollBar().setValue(
            self.history_text.verticalScrollBar().maximum()
        )
    
    def _log_response(self, response: str):
        """Log response to history."""
        self.history_text.append(f"<span style='color: green;'>← {response}</span>")
        self.history_text.verticalScrollBar().setValue(
            self.history_text.verticalScrollBar().maximum()
        )
    
    def _log_error(self, error: str):
        """Log error to history."""
        self.history_text.append(f"<b style='color: red;'>✗ ERROR: {error}</b>")
        self.history_text.verticalScrollBar().setValue(
            self.history_text.verticalScrollBar().maximum()
        )
    
    def _clear_history(self):
        """Clear command history."""
        self.history_text.clear()
    
    def refresh(self):
        """Refresh the dialog when driver changes."""
        self._update_device_info()