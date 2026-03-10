"""
Template for creating custom hardware drivers.

INSTRUCTIONS:
1. Copy this file and rename it (e.g., my_device_driver.py)
2. Rename the class (e.g., MyDeviceDriver)
3. Implement the required methods for your hardware protocol
4. Create a corresponding JSON profile in hardware_profiles/
5. Test with your hardware

EXAMPLE USE CASES:
- Binary protocols (non-CSV data)
- Devices requiring initialization sequences
- Polling-mode devices (not continuous streaming)
- Devices with special command formats
"""

from typing import Optional, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware_driver import HardwareDriver, HardwareProfile


class TemplateDriver(HardwareDriver):
    """
    Template driver - customize for your hardware.
    
    Follow the instructions in each method to implement support
    for your specific hardware device.
    """
    
    def __init__(self, profile: HardwareProfile):
        """
        Initialize your driver.
        
        Args:
            profile: HardwareProfile with configuration
        """
        super().__init__(profile)
        
        # TODO: Add your device-specific initialization here
        # Example:
        # self.device = None  # Your communication interface
        # self.packet_size = 20  # For binary protocols
        # self.buffer = bytearray()
    
    def connect(self, port: str) -> bool:
        """
        Connect to your hardware.
        
        TODO: Implement connection logic for your device.
        
        Example for serial device:
            import serial
            self.device = serial.Serial(
                port=port,
                baudrate=self.profile.baudrate,
                timeout=self.profile.timeout
            )
            self.is_connected = True
            return True
        
        Example for network device:
            import socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((port, 8080))
            self.is_connected = True
            return True
        
        Args:
            port: Connection identifier (serial port, IP address, etc.)
            
        Returns:
            True if connection successful
        """
        # TODO: Implement your connection logic
        raise NotImplementedError("Implement connect() for your hardware")
    
    def disconnect(self) -> None:
        """
        Disconnect from your hardware.
        
        TODO: Implement disconnection and cleanup.
        
        Example:
            if self.device:
                self.device.close()
            self.is_connected = False
        """
        # TODO: Implement your disconnection logic
        raise NotImplementedError("Implement disconnect() for your hardware")
    
    def read_sample(self) -> Optional[List[float]]:
        """
        Read one sample from your hardware.
        
        TODO: Implement data reading and parsing for your protocol.
        
        Example for binary protocol:
            if not self.is_data_available():
                return None
            
            # Read fixed-size packet
            data = self.device.read(self.packet_size)
            
            # Parse binary data (example: 4 channels, 16-bit unsigned)
            import struct
            ch1, ch2, ch3, ch4 = struct.unpack('<HHHH', data[0:8])
            
            # Convert to physical units
            scale = 5.0 / 65535.0  # 0-65535 -> 0-5V
            return [ch1 * scale, ch2 * scale, ch3 * scale, ch4 * scale]
        
        Example for polling mode (request-response):
            # Send command to request data
            cmd = self.profile.commands.get('get_data', 'DATA?')
            response = self.write_command(cmd)
            
            if response:
                # Parse response
                values = response.split(',')
                return [float(v) for v in values]
            return None
        
        Example for text protocol with custom format:
            line = self.device.readline().decode()
            # Parse format like "CH1:1.23 CH2:4.56 CH3:7.89"
            values = []
            for part in line.split():
                if ':' in part:
                    values.append(float(part.split(':')[1]))
            return values
        
        Returns:
            List of channel values, or None on error
        """
        # TODO: Implement your data reading logic
        raise NotImplementedError("Implement read_sample() for your hardware")
    
    def write_command(self, command: str) -> Optional[str]:
        """
        Send command to your hardware.
        
        TODO: Implement command sending for your protocol.
        
        Example for simple text protocol:
            full_cmd = command + self.profile.terminator
            self.device.write(full_cmd.encode())
            response = self.device.readline().decode().strip()
            return response
        
        Example for binary protocol:
            # Convert command to bytes
            cmd_byte = {'START': 0x01, 'STOP': 0x02}.get(command, 0x00)
            self.device.write(bytes([cmd_byte]))
            
            # Read response
            response = self.device.read(1)
            return 'OK' if response[0] == 0xAA else 'ERROR'
        
        Example with no response:
            self.device.write((command + '\n').encode())
            return 'OK'  # Assume success
        
        Args:
            command: Command string to send
            
        Returns:
            Response string, or None on error
        """
        # TODO: Implement your command sending logic
        raise NotImplementedError("Implement write_command() for your hardware")
    
    def is_data_available(self) -> bool:
        """
        Check if data is available to read.
        
        TODO: Implement availability check for your device.
        
        Example for buffered serial device:
            return self.device and self.device.in_waiting > 0
        
        Example for packet-based protocol:
            return self.device and self.device.in_waiting >= self.packet_size
        
        Example for always-ready device:
            return self.is_connected
        
        Returns:
            True if data can be read immediately
        """
        # TODO: Implement your availability check
        raise NotImplementedError("Implement is_data_available() for your hardware")
    
    def flush(self) -> None:
        """
        Flush input/output buffers.
        
        TODO: Implement buffer flushing.
        
        Example:
            if self.device:
                self.device.reset_input_buffer()
                self.device.reset_output_buffer()
        """
        # TODO: Implement your flush logic
        pass
    
    # ===================================================================
    # OPTIONAL METHODS - Override if your hardware needs these features
    # ===================================================================
    
    def initialize(self) -> bool:
        """
        Send initialization commands after connection.
        
        Override if your hardware needs a startup sequence.
        The default implementation sends the 'initialize' command from profile.
        
        Example:
            # Send multiple initialization commands
            init_commands = [
                self.profile.commands.get('reset', 'RST'),
                self.profile.commands.get('set_mode', 'MODE:AUTO'),
                self.profile.commands.get('set_rate', 'RATE:1000')
            ]
            
            for cmd in init_commands:
                response = self.write_command(cmd)
                if response != 'OK':
                    return False
            
            return True
        
        Returns:
            True if initialization successful
        """
        return super().safe_initialize()
    
    def start_streaming(self) -> bool:
        """
        Start continuous data stream.
        
        Override if your hardware needs a command to start streaming.
        The default implementation sends the 'start_streaming' command from profile.
        
        Example:
            cmd = self.profile.commands.get('start_streaming', 'STREAM ON')
            response = self.write_command(cmd)
            return response == 'OK'
        
        Returns:
            True if streaming started successfully
        """
        return super().start_streaming()
    
    def stop_streaming(self) -> bool:
        """
        Stop continuous data stream.
        
        Override if your hardware needs a command to stop streaming.
        The default implementation sends the 'stop_streaming' command from profile.
        
        Example:
            cmd = self.profile.commands.get('stop_streaming', 'STREAM OFF')
            response = self.write_command(cmd)
            return response == 'OK'
        
        Returns:
            True if streaming stopped successfully
        """
        return super().stop_streaming()
    
    def get_device_info(self) -> dict:
        """
        Get device information.
        
        Override to provide hardware-specific information.
        
        Example:
            info = super().get_device_info()
            version_cmd = self.profile.commands.get('get_version', 'VER?')
            version = self.write_command(version_cmd)
            info['version'] = version
            return info
        
        Returns:
            Dictionary with device information
        """
        return super().get_device_info()


# ======================================================================
# EXAMPLE: Custom Driver with Initialization and Binary Protocol
# ======================================================================

class ExampleBinaryDriver(HardwareDriver):
    """
    Example driver for a device with binary protocol.
    
    This is a complete example showing how to implement a driver
    for hardware that uses binary data instead of CSV text.
    """
    
    def __init__(self, profile: HardwareProfile):
        super().__init__(profile)
        self.serial = None
        self.packet_size = 20  # Header(1) + Data(16) + Checksum(2) + Footer(1)
    
    def connect(self, port: str) -> bool:
        try:
            import serial
            self.serial = serial.Serial(
                port=port,
                baudrate=self.profile.baudrate,
                timeout=self.profile.timeout
            )
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        if self.serial and self.serial.is_open:
            self.serial.close()
        self.is_connected = False
    
    def read_sample(self) -> Optional[List[float]]:
        if not self.is_data_available():
            return None
        
        try:
            import struct
            
            # Read packet
            packet = self.serial.read(self.packet_size)
            
            # Validate header (0xAA)
            if packet[0] != 0xAA:
                return None
            
            # Validate footer (0x55)
            if packet[-1] != 0x55:
                return None
            
            # Decode 4 channels (16-bit unsigned integers)
            ch1, ch2, ch3, ch4 = struct.unpack('<HHHH', packet[1:9])
            
            # Convert to voltages (0-65535 -> 0-5V)
            scale = 5.0 / 65535.0
            return [ch1 * scale, ch2 * scale, ch3 * scale, ch4 * scale]
            
        except Exception as e:
            print(f"Parse error: {e}")
            return None
    
    def write_command(self, command: str) -> Optional[str]:
        try:
            # Commands are ASCII
            self.serial.write((command + '\r\n').encode())
            response = self.serial.readline().decode().strip()
            return response
        except Exception as e:
            print(f"Command error: {e}")
            return None
    
    def is_data_available(self) -> bool:
        return (self.serial and self.serial.is_open and 
                self.serial.in_waiting >= self.packet_size)
    
    def flush(self) -> None:
        if self.serial and self.serial.is_open:
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()

# ======================================================================
# EXAMPLE: Complete Custom Driver Implementation
# ======================================================================

class ExampleBinaryDriver(HardwareDriver):
    """
    Example driver for a device with binary protocol.
    
    This is a complete example showing how to implement a driver
    for hardware that uses binary data instead of CSV text.
    """
    
    def __init__(self, profile: HardwareProfile):
        super().__init__(profile)
        self.serial = None
        self.packet_size = 20  # Header(1) + Data(16) + Checksum(2) + Footer(1)
    
    def connect(self, port: str) -> bool:
        try:
            import serial
            self.serial = serial.Serial(
                port=port,
                baudrate=self.profile.baudrate,
                timeout=self.profile.timeout
            )
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        if self.serial and self.serial.is_open:
            self.serial.close()
        self.is_connected = False
    
    def read_sample(self) -> Optional[List[float]]:
        if not self.is_data_available():
            return None
        
        try:
            import struct
            
            # Read packet
            packet = self.serial.read(self.packet_size)
            
            # Validate header (0xAA)
            if packet[0] != 0xAA:
                return None
            
            # Validate footer (0x55)
            if packet[-1] != 0x55:
                return None
            
            # Decode 4 channels (16-bit unsigned integers)
            ch1, ch2, ch3, ch4 = struct.unpack('<HHHH', packet[1:9])
            
            # Convert to voltages (0-65535 -> 0-5V)
            scale = 5.0 / 65535.0
            return [ch1 * scale, ch2 * scale, ch3 * scale, ch4 * scale]
            
        except Exception as e:
            print(f"Parse error: {e}")
            return None
    
    def write_command(self, command: str) -> Optional[str]:
        try:
            # Commands are ASCII
            self.serial.write((command + '\r\n').encode())
            response = self.serial.readline().decode().strip()
            return response
        except Exception as e:
            print(f"Command error: {e}")
            return None
    
    def is_data_available(self) -> bool:
        return (self.serial and self.serial.is_open and 
                self.serial.in_waiting >= self.packet_size)
    
    def flush(self) -> None:
        if self.serial and self.serial.is_open:
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
