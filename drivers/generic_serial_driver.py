"""
Generic serial driver - wraps existing SerialDevice.

This driver provides backward compatibility with the existing SerialDevice
implementation while conforming to the HardwareDriver interface.
"""

from typing import Optional, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware_driver import HardwareDriver, HardwareProfile
from serialdevice import SerialDevice


class GenericSerialDriver(HardwareDriver):
    """
    Generic driver for simple CSV-based serial devices.
    
    This wraps the existing SerialDevice implementation and provides
    CSV parsing with configurable separators and scale factors.
    
    Compatible with the original SerialPlotter implementation.
    """
    
    def __init__(self, profile: HardwareProfile):
        """
        Initialize generic serial driver.
        
        Args:
            profile: HardwareProfile with configuration
        """
        super().__init__(profile)
        self.device = SerialDevice(
            baudrate=profile.baudrate,
            timeout=profile.timeout
        )
    
    def connect(self, port: str) -> bool:
        """
        Connect to serial port.
        
        Args:
            port: Serial port identifier (e.g., 'COM3')
            
        Returns:
            True if connection successful
        """
        try:
            result = self.device.open_connection(port)
            self.is_connected = self.device.is_open
            return self.is_connected
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from serial port."""
        if self.device.is_open:
            self.device.close()
        self.is_connected = False
    
    def read_sample(self) -> Optional[List[float]]:
        """
        Read one line and parse CSV values.
        
        Reads a line from the serial device, splits it by the configured
        separator, converts values to floats, and optionally applies scale factors.
        
        Returns:
            List of float values, or None on error
        """
        if not self.is_data_available():
            return None
        
        try:
            line = self.device.readLine().decode().strip()
            
            # Get separator from profile (default to comma)
            separator = self.profile.data_format.get('separator', ',')
            
            # Split and parse values
            value_strings = line.split(separator)
            data = [float(v.strip()) for v in value_strings if v.strip()]
            
            # Apply scale factors if configured
            scale_factors = self.profile.data_format.get('scale_factors', [])
            if scale_factors and len(scale_factors) > 0:
                # Apply scale factors to corresponding channels
                data = [d * s if i < len(scale_factors) else d 
                       for i, (d, s) in enumerate(zip(data, scale_factors))]
            
            return data
            
        except Exception as e:
            print(f"Read error: {e}")
            return None
    
    def write_command(self, command: str) -> Optional[str]:
        """
        Send command and read response.
        
        Args:
            command: Command string to send (without terminator)
            
        Returns:
            Response string, or None on error
        """
        try:
            # Add terminator from profile
            full_command = command + self.profile.terminator
            self.device.write(full_command.encode('utf-8'))
            
            # Read response
            response = self.device.readLine().decode().strip()
            return response
            
        except Exception as e:
            print(f"Command error: {e}")
            return None
    
    def is_data_available(self) -> bool:
        """
        Check if data is available to read.
        
        Returns:
            True if serial port is open and has waiting data
        """
        return self.device.is_open and self.device.getInWaiting() > 0
    
    def flush(self) -> None:
        """Flush input/output buffers."""
        if self.device.is_open:
            self.device.flush()
    
    def set_baudrate(self, baudrate: int) -> None:
        """
        Change baudrate.
        
        Args:
            baudrate: New baudrate value
        """
        super().set_baudrate(baudrate)
        if self.device.is_open:
            self.device.setBaudrate(baudrate)
    
    def get_device_info(self) -> dict:
        """
        Get device information.
        
        Returns:
            Dictionary with device details
        """
        info = super().get_device_info()
        info['port'] = self.device.port if hasattr(self.device, 'port') else 'Unknown'
        return info
