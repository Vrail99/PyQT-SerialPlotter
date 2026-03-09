import serial
import serial.tools.list_ports
from serial.serialutil import SerialException, SerialTimeoutException
from typing import Optional, List
import threading  # Add threading for continuous reading

# Import HardwareDriver for interface compliance
from hardware_driver import HardwareDriver, HardwareProfile


class SerialDevice(serial.Serial, HardwareDriver):
    def __init__(self, port=None, baudrate: int = 115200, timeout: float = 1, terminationCharacter: bytes = b'\n', profile: Optional[HardwareProfile] = None):
        """
        Initialize the SerialDevice object.
        
        Args:
            port: The serial port to connect to (e.g., 'COM3' or '/dev/ttyUSB0').
            baudrate: The baud rate for the serial connection.
            timeout: The timeout for the serial connection in seconds (float).
            terminationCharacter: Line termination character.
            profile: Optional HardwareProfile for driver interface compliance.
        """
        # Initialize serial.Serial parent
        serial.Serial.__init__(self, port, baudrate, timeout=timeout)
        
        # Initialize HardwareDriver parent if profile provided
        if profile:
            HardwareDriver.__init__(self, profile)
            # Override with profile settings
            self.baudrate = profile.baudrate
            self.timeout = profile.timeout
            self.terminationCharacter = profile.terminator.encode() if isinstance(profile.terminator, str) else profile.terminator
        else:
            # Create a minimal profile for backward compatibility
            self.profile = None
            self.is_connected = False
            self.terminationCharacter = terminationCharacter
        
        self.write_timeout = timeout
        self.linereader = ReadLine(self)
        self.linereader.setTerminationCharacter(self.terminationCharacter)
        self.inputbuffer = bytearray()

   
    def open_connection(self, port):
        """
        Open the serial connection if it is not already open.
        """
        if port == " ":
            return "Not a valid Port"
        if self.port == port and self.is_open:
            return "Connection is already open."
        try:
            self.port = port
            self.open()
        except SerialException as e:
            return "Could not open the port. Check if the device is connected."
        
        return f"Connection opened on port {self.port}."

    def close_connection(self):
        """
        Close the serial connection if it is open.
        """
        if self.is_open:
            self.close()
            return f"Connection on port {self.port} closed."
        else:
            return "No open connection to close."
        
    def setTimeout(self, timeout):
        """
        Set the timeout for the serial connection.
        :param timeout: The timeout in seconds.
        """
        self.timeout = timeout

    def setBaudrate(self, baudrate):
        """
        Set the baud rate for the serial connection.
        :param baudrate: The baud rate to set.
        """
        self.baudrate = baudrate
        print(f"Baudrate set to {baudrate}")

    @staticmethod
    def list_devices():
        """
        List available serial devices.
        :return: A list of available serial device names.
        """
        ports = serial.tools.list_ports.comports()
        return [(port.device, port.description) for port in ports]

    def writeCommand(self, content, length=20):
        """
        Write a fixed-length string to the serial device.
        :param content: The content to write.
        :param length: The fixed length of the string.
        """
        if not self.is_open:
            raise ConnectionError("Connection is not open.")
        if len(content) > length:
            raise ValueError("Content length exceeds the fixed length.")
        padded_content = content.ljust(length)+ '\n' # Pad the content to the fixed length
        self.write(padded_content.encode('utf-8'))
        

    def _readline(self, max_tries=1) -> bytearray:
        """Reads a Line from the Serial bus, without a newline character.
        reads a maximum of 4096 bytes at once.

        Keyword Arguments:

        terminationCharacter -- the character that defines the end of a line (default b'\n')

        Returns a bytearray with the line
        """
        nr_tries = 0
        while True:
            # If buffer was filled before, check for new line
            # check if the termination character is in the buffer
            idx = self.inputbuffer.find(self.terminationCharacter)
            if idx > 0:
                #Return the number up to the termination character
                ret = self.inputbuffer[:idx]
                #delete all until this entry
                del self.inputbuffer[:idx+1]
                nr_tries = 0
                return ret
            # read a maximum of 4096 bytes at once
            i = max(1, min(4096, self.in_waiting))
            data = self.read(i)
            if (len(data) == 0):
                nr_tries += 1
                if nr_tries > max_tries:
                    raise serial.SerialTimeoutException("Timeout while reading from serial port")
                continue
            self.inputbuffer += data # add data to buffer

    def readLine(self, max_tries=1) -> bytearray:
        """
        Read a line from the serial device.
        :return: The line read from the serial device.
        """
        if not self.is_open:
            raise ConnectionError("Connection is not open.")
        try:
            data = self._readline(max_tries=max_tries)
        except serial.SerialTimeoutException:
            print("Timeout while reading from serial port")
            self.close()
            raise
        return data

    def flush(self):
        """
        Flush the serial connection.
        """
        if not self.is_open:
            raise ConnectionError("Connection is not open.")
        self.reset_input_buffer()
        self.reset_output_buffer()
        self.inputbuffer = bytearray()  # Clear the input buffer

    def read_lines_permanently(self, callback):
        """
        Read lines from the serial port continuously and pass them to a callback function.
        :param callback: A function to handle each line read from the serial port.
        """
        if not self.is_open:
            raise ConnectionError("Connection is not open.")

        def read_loop():
            while self.is_open:
                try:
                    line = self.readLine().decode()
                    callback(line)
                except Exception as e:
                    print(f"Error reading line: {e}")
                    break

        thread = threading.Thread(target=read_loop, daemon=True)
        thread.start()

    def readBytes(self, length: int) -> bytearray:
        """Reads a number of bytes from the serial bus

        :param length: the number of bytes to read
        :return: the bytes read"""
        data = bytearray()
        if self.is_open:
            try:
                data = self.read(length)
            except serial.SerialTimeoutException:
                print("Timeout while reading from serial port")

        return data
    
    def getInWaiting(self):
        return self.in_waiting + len(self.inputbuffer)
    
    # ==========================================
    # HardwareDriver interface implementation
    # ==========================================
    
    def connect(self, port: str) -> bool:
        """
        Connect to serial port (HardwareDriver interface).
        
        Args:
            port: Serial port identifier (e.g., 'COM3')
            
        Returns:
            True if connection successful
        """
        try:
            result = self.open_connection(port)
            self.is_connected = self.is_open
            return self.is_connected
        except Exception as e:
            print(f"Connection error: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from serial port (HardwareDriver interface)."""
        if self.is_open:
            self.close()
        self.is_connected = False
    
    def initialize(self) -> bool:
        """
        Initialize device with optional initialization command.
        
        Returns:
            True if initialization successful
        """
        try:
            # Send initialization command if defined in profile
            if self.profile and 'initialize' in self.profile.commands:
                command = self.profile.commands['initialize']
                response = self.write_command(command)
                if response is None:
                    print(f"Warning: No response to initialization command")
            return True
        except SerialTimeoutException as e:
            print(f"Initialization timeout: {e}")
            raise TimeoutError(f"Initialization timeout: {e}")
        except Exception as e:
            print(f"Initialization warning: {e}")
            return True
    
    def read_sample(self) -> Optional[List[float]]:
        """
        Read one line and parse CSV values (HardwareDriver interface).
        
        Reads a line from the serial device, splits it by the configured
        separator, converts values to floats, and optionally applies scale factors.
        
        Returns:
            List of float values, or None on error
        """
        if not self.is_data_available():
            return None
        
        try:
            line = self.readLine().decode().strip()
            
            # Get separator from profile (default to comma)
            separator = ','
            scale_factors = []
            if self.profile and self.profile.data_format:
                separator = self.profile.data_format.get('separator', ',')
                scale_factors = self.profile.data_format.get('scale_factors', [])
            
            # Split and parse values
            value_strings = line.split(separator)
            data = [float(v.strip()) for v in value_strings if v.strip()]
            
            # Apply scale factors if configured
            if scale_factors and len(scale_factors) > 0:
                data = [d * s if i < len(scale_factors) else d 
                       for i, (d, s) in enumerate(zip(data, scale_factors))]
            
            return data
            
        except SerialTimeoutException as e:
            print(f"Read timeout: {e}")
            raise TimeoutError(f"Read timeout: {e}")
        except Exception as e:
            print(f"Read error: {e}")
            return None
    
    def write_command(self, command: str) -> Optional[str]:
        """
        Send command and read response (HardwareDriver interface).
        
        Args:
            command: Command string to send (without terminator)
            
        Returns:
            Response string, or None on error
        """
        try:
            # Add terminator from profile
            if self.profile:
                full_command = command + self.profile.terminator
            else:
                full_command = command + '\n'
            
            self.write(full_command.encode('utf-8'))
            
            # Read response
            response = self.readLine().decode().strip()
            return response
            
        except SerialTimeoutException as e:
            print(f"Write command timed out: {e}")
            raise TimeoutError(f"Write command timed out: {e}")
    
    def is_data_available(self) -> bool:
        """
        Check if data is available to read (HardwareDriver interface).
        
        Returns:
            True if serial port is open and has waiting data
        """
        return self.is_open and self.getInWaiting() > 0
    
    def set_baudrate(self, baudrate: int) -> None:
        """
        Change baudrate (HardwareDriver interface).
        
        Args:
            baudrate: New baudrate value
        """
        if self.profile:
            self.profile.baudrate = baudrate
        self.baudrate = baudrate
        if self.is_open:
            # Note: setBaudrate from serial.Serial parent updates the hardware
            pass
    
    def list_available_ports(self) -> List[tuple]:
        """
        List available serial ports (HardwareDriver interface).
        
        Returns:
            List of tuples (port_name, port_description)
        """
        return self.list_devices()
    
    def get_device_info(self) -> dict:
        """
        Get device information (HardwareDriver interface).
        
        Returns:
            Dictionary with device details
        """
        info = {
            'driver_type': 'SerialDevice',
            'baudrate': self.baudrate,
            'timeout': self.timeout,
            'port': self.port if hasattr(self, 'port') else 'Unknown',
            'is_open': self.is_open
        }
        if self.profile:
            info['profile_name'] = self.profile.name
        return info

##############################################################################################################
################                        Readline Helperclass                          ########################
##############################################################################################################

class ReadLine:
    """Linereader Helperclass, which introduces a buffer for incoming values. It is
    faster than the built-in function readline of pyserial."""

    def __init__(self, s: serial.Serial):
        """Inits the helperclass with an empty buffer

        Keyword Arguments:

        s -- the serialBus instance to use"""
        self.buf = bytearray()
        self.s = s
        self.terminationCharacter = b'\n'
        self.stopped = False

    def setSerialBus(self, s: serial.Serial):
        """Sets the serial bus to use for reading

        :param s: the serial bus to use"""
        self.s = s

    def setTerminationCharacter(self, terminationCharacter : bytes):
        """Sets the termination character for the readline function

        :param terminationCharacter: the termination character to use"""
        self.terminationCharacter = terminationCharacter

    def readline(self) -> bytearray | None:
        """Reads a Line from the Serial bus, without a newline character.
        reads a maximum of 4096 bytes at once.

        Keyword Arguments:

        terminationCharacter -- the character that defines the end of a line (default b'\n')

        Returns a bytearray with the line
        """
        self.counter = 0
        while True:
            # read a maximum of 4096 bytes at once
            i = max(1, min(4096, self.s.in_waiting))
            data = self.s.read(i)
            if (len(data) == 0):
                self.counter += 1
                if self.counter > 1:
                    raise serial.SerialTimeoutException("Timeout while reading from serial port")
                continue
            self.buf += data # add data to buffer
            # check if the termination character is in the buffer
            idx = self.buf.find(self.terminationCharacter)
            if idx > 0:
                #Return the number up to the termination character
                ret = self.buf[:idx]
                #delete all until this entry
                del self.buf[:idx+1]
                self.counter = 0
                return ret

    def getBufLen(self) -> int:
        """Returns the number of bytes in the buffer"""
        return len(self.buf)

    def flush(self) -> None:
        """Empties the buffer"""
        self.buf = bytearray()

    def stopReading(self, stop):
        """Defines if the readline function returns a value when reading a stream """
        self.stopped = stop

if __name__ == "__main__":
    # Example usage
    device = SerialDevice()
    device.list_devices()