"""
Hardware driver abstraction for supporting multiple device types.

This module provides the abstract base class that all hardware drivers must implement,
along with the HardwareProfile dataclass for configuration management.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class HardwareProfile:
    """
    Hardware configuration profile.
    
    This dataclass holds all configuration parameters needed for a specific
    hardware device, including communication settings, command definitions,
    and data format specifications.
    """
    name: str
    driver_class: str
    baudrate: int = 115200
    timeout: float = 1.0
    terminator: str = "\n"
    commands: Dict[str, str] = field(default_factory=dict)
    data_format: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default data format if not provided."""
        if not self.data_format:
            self.data_format = {
                'type': 'csv',
                'separator': ',',
                'scale_factors': []
            }


class HardwareDriver(ABC):
    """
    Abstract base class for hardware communication drivers.
    
    All hardware drivers must implement these methods to be compatible
    with the SerialPlotter application. This provides a consistent interface
    for different hardware types with varying communication protocols.
    """
    
    def __init__(self, profile: HardwareProfile):
        """
        Initialize driver with hardware profile.
        
        Args:
            profile: HardwareProfile with device-specific configuration
        """
        self.profile = profile
        self.is_connected = False
    
    @abstractmethod
    def connect(self, port: str) -> bool:
        """
        Connect to hardware on specified port.
        
        Args:
            port: Serial port identifier (e.g., 'COM3', '/dev/ttyUSB0')
            
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from hardware and cleanup resources."""
        pass
    
    @abstractmethod
    def read_sample(self) -> Optional[List[float]]:
        """
        Read one data sample from hardware.
        
        This method should read and parse one complete data sample,
        returning a list of channel values. The parsing should be
        handled by the driver based on the hardware's protocol.
        
        Returns:
            List of channel values as floats, or None if read failed
            
        Example:
            [1.23, 4.56, 7.89]  # 3 channels of data
        """
        pass
    
    @abstractmethod
    def write_command(self, command: str) -> Optional[str]:
        """
        Send command to hardware and get response.
        
        Args:
            command: Command string to send (without terminator)
            
        Returns:
            Response string, or None if command failed
        """
        pass
    
    @abstractmethod
    def is_data_available(self) -> bool:
        """
        Check if data is available to read.
        
        Returns:
            True if data can be read immediately, False otherwise
        """
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """Flush input/output buffers."""
        pass
    
    # Optional methods with default implementations
    
    def initialize(self) -> bool:
        """
        Initialize hardware after connection, safely
        
        Override this to send initialization commands defined in the profile.
        The default implementation sends the 'initialize' command if defined.
        
        Returns:
            True if initialization successful
        """
        
        init_cmd = self.profile.commands.get('initialize', '')
        return_on_init = self.profile.commands.get('return_on_init', '?')
        if init_cmd:
            response = self.write_command(init_cmd)
            if response != return_on_init:
                return False
            
        return True
        # except TimeoutError as e:
        #     raise TimeoutError(f"Initialization timeout: {e}")
            # return False, f"Initialization timeout: {e}"

    def start_streaming(self) -> bool:
        """
        Start continuous data streaming mode.
        
        Override if your hardware needs a command to start streaming.
        The default implementation sends the 'start_streaming' command if defined.
        
        Returns:
            True if streaming started successfully
        """
        start_cmd = self.profile.commands.get('start_streaming', '')
        try:
            if start_cmd:
                response = self.write_command(start_cmd)
                return response is not None
            return True
        except TimeoutError as e:
            raise TimeoutError(f"HardwareDriver: Error starting streaming: {e}")
    
        return False
    
    def stop_streaming(self) -> bool:
        """
        Stop continuous data streaming mode.
        
        Override if your hardware needs a command to stop streaming.
        The default implementation sends the 'stop_streaming' command if defined.
        
        Returns:
            True if streaming stopped successfully
        """
        stop_cmd = self.profile.commands.get('stop_streaming', '')
        if stop_cmd:
            response = self.write_command(stop_cmd)
            return response is not None
        return True
    
    def get_device_info(self) -> Dict[str, str]:
        """
        Get device information.
        
        Override to provide more detailed device information.
        
        Returns:
            Dictionary with device info (name, version, etc.)
        """
        return {
            'name': self.profile.name,
            'driver': self.__class__.__name__,
            'baudrate': str(self.profile.baudrate)
        }
    
    def set_baudrate(self, baudrate: int) -> None:
        """
        Change baudrate (if supported).
        
        Override if your hardware supports runtime baudrate changes.
        The default implementation only updates the profile.
        
        Args:
            baudrate: New baudrate value
        """
        self.profile.baudrate = baudrate
    
    @staticmethod
    def list_available_ports() -> List[tuple]:
        """
        List available serial ports.
        
        Returns:
            List of (port, description) tuples
        """
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            return [(port.device, port.description) for port in ports]
        except ImportError:
            raise ImportError("pyserial is required to list available ports. Install with 'pip install pyserial'.")
