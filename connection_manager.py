"""
Connection management for hardware drivers.

Handles driver lifecycle, port selection, and connection state.
"""

from typing import Optional, Tuple
from PySide6.QtCore import QObject, Signal as pyqtSignal
from drivers.serialdevice import SerialDevice
from hardware_driver_manager import HardwareDriverManager
from error_handler import ErrorHandler
from acquisition_state import ConnectionInfo


class ConnectionManager(QObject):
    """
    Manages hardware driver connections and port lifecycle.
    
    Emits signals for connection state changes.
    """
    
    # Signals
    connectionEstablished = pyqtSignal(str)  # port name
    connectionLost = pyqtSignal()
    portListUpdated = pyqtSignal(list)  # list of available ports
    profileChanged = pyqtSignal(str)  # profile name
    profileBaudrateChanged = pyqtSignal(int)  # baudrate from new profile
    
    def __init__(self, driver_manager: HardwareDriverManager, 
                 error_handler: ErrorHandler):
        """
        Initialize connection manager.
        
        Args:
            driver_manager: Hardware driver manager
            error_handler: Error handler for logging
        """
        super().__init__()
        self.driver_manager = driver_manager
        self.error_handler = error_handler
        self.current_driver = None
        self.fallback_serial_device = SerialDevice(
            baudrate=115200, 
            timeout=1
        )
    
    def set_profile(self, profile_name: str) -> bool:
        """
        Select and initialize a hardware profile.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            True if successful
        """
        try:
            # Disconnect current driver if connected
            if self.is_connected():
                self.disconnect()
            
            # Create new driver
            self.current_driver = self.driver_manager.create_driver(profile_name)
            if not self.current_driver:
                self.error_handler.error(f"Failed to create driver for {profile_name}")
                return False
            
            self.driver_manager.set_current_driver(self.current_driver)
            self.profileChanged.emit(profile_name)
            if hasattr(self.current_driver, 'profile') and hasattr(self.current_driver.profile, 'baudrate'):
                self.profileBaudrateChanged.emit(self.current_driver.profile.baudrate)
            self.error_handler.info(f"Selected profile: {profile_name}")
            return True
            
        except Exception as e:
            self.error_handler.error(f"Error setting profile: {e}")
            return False
    
    def get_available_ports(self) -> list:
        """
        Get list of available serial ports.
        
        Returns:
            List of (port, description) tuples
        """
        try:
            if self.current_driver:
                ports = self.current_driver.list_available_ports()
            else:
                ports = self.fallback_serial_device.list_devices()
            
            # Add disconnect option
            ports.insert(0, (" ", "Disconnect"))
            # self.portListUpdated.emit(ports)
            return ports
            
        except Exception as e:
            self.error_handler.warning(f"Error listing ports: {e}")
            return [(" ", "Disconnect")]
    
    def connect(self, port: str) -> Tuple[bool, Optional[str]]:
        """
        Connect to a serial port.
        
        Args:
            port: Port name to connect to
            
        Returns:
            Tuple of (success, error_message)
        """
        if port == " " or not port:
            return self.disconnect()
        
        if not self.current_driver:
            return False, "No hardware profile selected"
        
        try:
            # Attempt connection
            success = self.current_driver.connect(port)
            if not success:
                self.error_handler.error(f"Failed to connect to {port}")
                return False, "Failed to connect to device"
            
            # Initialize driver
            if not self.current_driver.initialize():
                self.error_handler.error(f"Failed to initialize driver for {port}")
                self.current_driver.disconnect()
                return False, "Failed to initialize device"
            
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
        """
        Disconnect from current device.
        
        Returns:
            Tuple of (success, error_message)
        """
        if self.current_driver and self.current_driver.is_connected:
            self.current_driver.disconnect()

        self.connectionLost.emit()
        self.error_handler.info("Disconnected from device")
        return True, None
    
    def reconnect(self, port: str) -> Tuple[bool, Optional[str]]:
        """
        Reconnect to a port.
        
        Args:
            port: Port to reconnect to
            
        Returns:
            Tuple of (success, error_message)
        """
        if not port or port == " ":
            return False, "No port specified"
        
        # Disconnect first
        self.disconnect()
        
        # Reconnect
        return self.connect(port)
    
    def set_baudrate(self, baudrate: int) -> bool:
        """
        Set serial baudrate.
        
        Args:
            baudrate: Baudrate to set
            
        Returns:
            True if successful
        """
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
        """
        Send command to connected device.
        
        Args:
            command: Command string to send
            
        Returns:
            Tuple of (success, response)
        """
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
        """Check if currently connected."""
        if self.current_driver:
            return self.current_driver.is_connected
        return False
    
    def is_data_available(self) -> bool:
        """Check if data is available from device."""
        if self.current_driver:
            return self.current_driver.is_data_available()
        return False
    
    def read_sample(self):
        """Read one sample from device."""
        if self.current_driver:
            try:
                return self.current_driver.read_sample()
            except TimeoutError as e:
                self.error_handler.warning(f"{e}")
            except ConnectionError as e:
                self.error_handler.warning(f"{e}")
            except Exception as e:
                self.error_handler.warning(f"Other Error: {e}")
        return None
    
    def flush(self) -> None:
        """Flush device buffers."""
        if self.current_driver:
            self.current_driver.flush()
    
    def get_connection_info(self) -> ConnectionInfo:
        """Get current connection information."""
        return ConnectionInfo(
            port=getattr(self.current_driver, 'port', None) if self.current_driver else None,
            profile_name=getattr(self.current_driver, 'profile', {}).get('name') if self.current_driver else None,
            is_connected=self.is_connected()
        )
