"""
UI construction for SerialPlotter.

Builds all widgets and layouts, emitting signals for user actions
so SerialPlotter can stay decoupled from UI details.
"""

from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget, QPushButton,
    QComboBox, QLineEdit, QLabel
)
from PySide6.QtCore import QObject, Signal as pyqtSignal
from PySide6.QtGui import QIntValidator
from config import Constants


class UIBuilder(QObject):
    """
    Builds the SerialPlotter UI and exposes user actions as signals.

    All widget creation is contained here. SerialPlotter connects to
    the signals and delegates to the appropriate manager.
    """

    # User action signals
    profileSelected = pyqtSignal(str)
    portSelected = pyqtSignal(str)
    baudrateChanged = pyqtSignal(int)
    baudrateDisplay = pyqtSignal(str)   # for syncing baudrate from profile
    reconnectRequested = pyqtSignal()
    commandSendRequested = pyqtSignal(str)
    plotLengthChanged = pyqtSignal(int)
    clearPlotsRequested = pyqtSignal()
    startStopToggled = pyqtSignal()
    saveCsvRequested = pyqtSignal()
    driverConfigRequested = pyqtSignal()

    def __init__(self, driver_manager, connection_manager, config, max_plot_length):
        super().__init__()
        self.driver_manager = driver_manager
        self.connection_manager = connection_manager
        self.config = config
        self.max_plot_length = max_plot_length

        # Exposed widget references
        self.profile_dropdown = None
        self.port_dropdown = None
        self.connection_status_button = None
        self.baudrate_dropdown = None
        self.command_input = None
        self.plot_length_input = None
        self.start_stop_button = None

        connection_manager.connectionLost.connect(self._resetPort)
        connection_manager.connectionEstablished.connect(self._set_port_selection)

    def build(self, plot_layout, param_tree) -> QHBoxLayout:
        """Build and return the complete main layout."""
        plot_control = QVBoxLayout()
        plot_control.addWidget(plot_layout)
        plot_control.addLayout(self._build_serial_controls())
        plot_control.addLayout(self._build_plot_length_control())
        plot_control.addLayout(self._build_command_panel())
        plot_control.addLayout(self._build_action_buttons())

        main_layout = QHBoxLayout()
        main_layout.addWidget(param_tree, stretch=1)
        main_layout.addLayout(plot_control, stretch=4)
        return main_layout

    def _build_serial_controls(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        # Profile dropdown
        self.profile_dropdown = QComboBox()
        profile_names = self.driver_manager.get_profile_names()
        self.profile_dropdown.addItems(profile_names if profile_names else ["No profiles found"])
        self.profile_dropdown.currentTextChanged.connect(self._on_profile_selected)

        # Port dropdown
        self.port_dropdown = QComboBox()
        self.port_dropdown.enterEvent = lambda e: self._refresh_ports()
        self._refresh_ports()
        self.port_dropdown.currentIndexChanged.connect(
            lambda: self.portSelected.emit(self.port_dropdown.currentText().split('-')[0])
        )

        # Connection status button
        self.connection_status_button = QPushButton("●")
        self.connection_status_button.setFixedSize(35, 25)
        self.connection_status_button.clicked.connect(self.reconnectRequested.emit)
        self.set_connection_status(False)

        # Baudrate dropdown
        self.baudrate_dropdown = QComboBox()
        self.baudrate_dropdown.addItems(["921600", "115200", "9600"])
        self.baudrate_dropdown.setCurrentText(str(Constants.DEFAULT_BAUDRATE))
        self.baudrate_dropdown.setToolTip("Baudrate")
        self.baudrate_dropdown.currentTextChanged.connect(
            lambda v: self.baudrateChanged.emit(int(v)) if v.isdigit() else None
        )

        # Driver config button
        config_btn = QPushButton("Driver Config")
        config_btn.setToolTip("Open driver configuration and command interface")
        config_btn.clicked.connect(self.driverConfigRequested.emit)

        for label_text, widget in [
            ("Profile:", self.profile_dropdown),
            (None, config_btn),
            ("Port:", self.port_dropdown),
            (None, self.connection_status_button),
            ("Baud:", self.baudrate_dropdown),
        ]:
            if label_text:
                layout.addWidget(QLabel(label_text))
            layout.addWidget(widget)

        return layout

    def _build_plot_length_control(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.plot_length_input = QLineEdit(str(self.max_plot_length))
        self.plot_length_input.setValidator(QIntValidator(1, 100000))
        self.plot_length_input.editingFinished.connect(
            lambda: self.plotLengthChanged.emit(int(self.plot_length_input.text()))
        )
        layout.addWidget(QLabel("Max Plot Length:"))
        layout.addWidget(self.plot_length_input)
        return layout

    def _build_command_panel(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.command_input = QLineEdit("VAL?")
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(lambda: self.commandSendRequested.emit(self.command_input.text()))
        clear_btn = QPushButton("Clear Plot Data")
        clear_btn.clicked.connect(self.clearPlotsRequested.emit)
        layout.addWidget(self.command_input)
        layout.addWidget(send_btn)
        layout.addWidget(clear_btn)
        return layout

    def _build_action_buttons(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        self.start_stop_button = QPushButton("Start DAQ & Plot")
        self.start_stop_button.clicked.connect(self.startStopToggled.emit)
        save_btn = QPushButton("Save to CSV")
        save_btn.clicked.connect(self.saveCsvRequested.emit)
        layout.addWidget(self.start_stop_button)
        layout.addWidget(save_btn)
        return layout

    def _on_profile_selected(self, name: str) -> None:
        if name != "No profiles found":
            self.profileSelected.emit(name)

    def _refresh_ports(self) -> None:
        ports = self.connection_manager.get_available_ports()
        if len(ports) != self.port_dropdown.count():
            self.port_dropdown.blockSignals(True)
            self.port_dropdown.clear()
            self.port_dropdown.addItems([f"{p[0]}-({p[1]})" for p in ports])
            self.port_dropdown.blockSignals(False)

    def _set_port_selection(self, port: str) -> None:
        """Set the port dropdown to show the specified port."""
        for i in range(self.port_dropdown.count()):
            if self.port_dropdown.itemText(i).startswith(port):
                self.port_dropdown.blockSignals(True)
                self.port_dropdown.setCurrentIndex(i)
                self.port_dropdown.blockSignals(False)
                break
    
    def _resetPort(self) -> None:
        self.port_dropdown.blockSignals(True)
        self.port_dropdown.setCurrentIndex(0)
        self.port_dropdown.blockSignals(False)

    # ─── Public control methods ────────────────────────────────────────────

    def set_connection_status(self, connected: bool) -> None:
        """Update connection indicator style."""
        if connected:
            style = ("QPushButton { background-color: #00cc00; color: white; font-size: 16px;"
                     " font-weight: bold; border-radius: 3px; }"
                     "QPushButton:hover { background-color: #00ff00; }")
            tip = "Connected - Click to reconnect"
        else:
            style = ("QPushButton { background-color: #cc0000; color: white; font-size: 16px;"
                     " font-weight: bold; border-radius: 3px; }"
                     "QPushButton:hover { background-color: #ff0000; }")
            tip = "Disconnected - Click to reconnect"
        self.connection_status_button.setStyleSheet(style)
        self.connection_status_button.setToolTip(tip)

    def set_start_stop_text(self, text: str) -> None:
        self.start_stop_button.setText(text)

    def set_baudrate_display(self, baudrate: int) -> None:
        self.baudrate_dropdown.setCurrentText(str(baudrate))

    def get_current_port(self) -> str:
        return self.port_dropdown.currentText().split('-')[0]

    def update_ports(self, ports: list) -> None:
        self._refresh_ports()

    def initialize_first_profile(self) -> None:
        """Trigger initial profile signal after all connections are wired."""
        name = self.profile_dropdown.currentText()
        if name and name != "No profiles found":
            self.profileSelected.emit(name)
