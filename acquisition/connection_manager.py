"""
Connection manager: driver lifecycle, port selection, and connection state.
"""

from typing import Optional, Tuple

from PySide6.QtCore import QObject, Signal as pyqtSignal

from hardware.drivers.serial_device import SerialDevice
from hardware.manager import HardwareDriverManager
from core.error_handler import ErrorHandler
from core.models import ConnectionInfo


class ConnectionManager(QObject):
    """
    Manages hardware driver connections and port lifecycle.

    Emits signals for connection state changes so the rest of the
    application can remain decoupled from serial details.
    """

    connectionEstablished = pyqtSignal(str)   # port name
    connectionLost = pyqtSignal()
    portListUpdated = pyqtSignal(list)
    profileChanged = pyqtSignal(str)
    profileBaudrateChanged = pyqtSignal(int)

    def __init__(self, driver_manager: HardwareDriverManager,
                 error_handler: ErrorHandler) -> None:
        super().__init__()
        self.driver_manager = driver_manager
        self.error_handler = error_handler
        self.current_driver = None
        self.fallback_serial_device = SerialDevice(baudrate=115200, timeout=1)

    # ─── Profile ──────────────────────────────────────────────────────────

    def set_profile(self, profile_name: str) -> bool:
        try:
            if self.is_connected():
                self.disconnect()
            self.current_driver = self.driver_manager.create_driver(profile_name)
            if not self.current_driver:
                self.error_handler.error(f"Failed to create driver for {profile_name}")
                return False
            self.driver_manager.set_current_driver(self.current_driver)
            self.profileChanged.emit(profile_name)
            if hasattr(self.current_driver, "profile"):
                self.profileBaudrateChanged.emit(self.current_driver.profile.baudrate)
            self.error_handler.info(f"Selected profile: {profile_name}")
            return True
        except Exception as e:
            self.error_handler.error(f"Error setting profile: {e}")
            return False

    # ─── Port enumeration ─────────────────────────────────────────────────

    def get_available_ports(self) -> list:
        try:
            ports = (
                self.current_driver.list_available_ports()
                if self.current_driver
                else self.fallback_serial_device.list_devices()
            )
            ports.insert(0, (" ", "Disconnect"))
            return ports
        except Exception as e:
            self.error_handler.warning(f"Error listing ports: {e}")
            return [(" ", "Disconnect")]

    # ─── Connection lifecycle ─────────────────────────────────────────────

    def connect(self, port: str) -> Tuple[bool, Optional[str]]:
        if port == " " or not port:
            return self.disconnect()
        if not self.current_driver:
            return False, "No hardware profile selected"
        try:
            if not self.current_driver.connect(port):
                self.error_handler.error(f"Failed to connect to {port}")
                return False, "Failed to connect to device"
            if not self.current_driver.initialize():
                self.error_handler.error(f"Failed to initialise driver for {port}")
                self.current_driver.disconnect()
                return False, "Failed to initialise device"
            self.connectionEstablished.emit(port)
            return True, f"Connected to {port}"
        except Exception as e:
            error_msg = f"{self.current_driver.profile.name}: {e}"
            self.error_handler.error(error_msg)
            if self.current_driver:
                self.current_driver.disconnect()
            self.connectionLost.emit()
            return False, error_msg

    def disconnect(self) -> Tuple[bool, Optional[str]]:
        if self.current_driver and self.current_driver.is_connected:
            self.current_driver.disconnect()
        self.connectionLost.emit()
        self.error_handler.info("Disconnected from device")
        return True, None

    def reconnect(self, port: str) -> Tuple[bool, Optional[str]]:
        if not port or port == " ":
            return False, "No port specified"
        self.disconnect()
        return self.connect(port)

    # ─── Data I/O ─────────────────────────────────────────────────────────

    def set_baudrate(self, baudrate: int) -> bool:
        try:
            if self.current_driver:
                self.current_driver.set_baudrate(baudrate)
            else:
                self.fallback_serial_device.setBaudrate(baudrate)
            return True
        except Exception as e:
            self.error_handler.warning(f"Error setting baudrate: {e}")
            return False

    def send_command(self, command: str) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Not connected to device"
        try:
            response = self.current_driver.write_command(command)
            return True, response
        except Exception as e:
            error_msg = f"Command error: {e}"
            self.error_handler.error(error_msg)
            return False, error_msg

    def is_connected(self) -> bool:
        return bool(self.current_driver and self.current_driver.is_connected)

    def is_data_available(self) -> bool:
        return bool(self.current_driver and self.current_driver.is_data_available())

    def read_sample(self):
        if not self.current_driver:
            return None
        try:
            return self.current_driver.read_sample()
        except TimeoutError as e:
            self.error_handler.warning(str(e))
        except ConnectionError as e:
            self.error_handler.warning(str(e))
        except Exception as e:
            self.error_handler.warning(f"Error reading sample: {e}")
        return None

    def flush(self) -> None:
        if self.current_driver:
            self.current_driver.flush()

    def get_connection_info(self) -> ConnectionInfo:
        return ConnectionInfo(
            port=getattr(self.current_driver, "port", None) if self.current_driver else None,
            profile_name=(
                self.current_driver.profile.name
                if self.current_driver and hasattr(self.current_driver, "profile")
                else None
            ),
            is_connected=self.is_connected(),
        )
