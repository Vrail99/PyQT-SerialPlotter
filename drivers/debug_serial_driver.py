from time import time

import serial
import serial.tools.list_ports
from serial.serialutil import SerialException, SerialTimeoutException
from typing import Optional, List
import time

# Import HardwareDriver for interface compliance
from hardware_driver import HardwareDriver, HardwareProfile

samples_parsed = 0
parse_times = []

class DebugSerialDriver(serial.Serial, HardwareDriver):
    def __init__(self, profile: HardwareProfile):
        """
        Initialize the SerialDevice object.
        """
        # Initialize serial.Serial parent
        serial.Serial.__init__(self)
        #profile.port, profile.baudrate, timeout=profile.timeout, write_timeout=profile.timeout)
        
        # Initialize HardwareDriver parent if profile provided
        HardwareDriver.__init__(self, profile)
        # Override with profile settings
        self.baudrate = profile.baudrate
        self.timeout = profile.timeout
        self.write_timeout = profile.timeout
        self.terminationCharacter = profile.terminator.encode() if isinstance(profile.terminator, str) else profile.terminator
        
        self.inputbuffer = bytearray()

        # Diagnostics
        self.byte_count = 0
        self.line_count = 0
        self.parse_success = 0
        self.parse_failures = 0
        self.diagnostic_start = time.perf_counter()

        self._sps_last_rate = None
        self._sps_window_start = time.perf_counter()
        self._sps_window_count = 0
        self._sps_update_period_s = 1.0

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

    def get_measured_sps(self):
        return self._sps_last_rate  # float | None

    def reset_sps_measurement(self):
        self._sps_last_rate = None
        self._sps_window_start = time.perf_counter()
        self._sps_window_count = 0

    @staticmethod
    def list_devices():
        """
        List available serial devices.
        :return: A list of available serial device names.
        """
        ports = serial.tools.list_ports.comports()
        return [(port.device, port.description) for port in ports]
        

    def _readline(self, max_tries=1) -> bytearray:
        """Existing readline with byte counting"""
        nr_tries = 0
        while True:
            idx = self.inputbuffer.find(self.terminationCharacter)
            if idx > 0:
                ret = self.inputbuffer[:idx]
                del self.inputbuffer[:idx+1]
                
                # Count complete line
                self.line_count += 1
                self.byte_count += len(ret) + 1  # +1 for newline
                
                # Log every 100 lines
                if self.line_count % 100 == 0:
                    elapsed = time.perf_counter() - self.diagnostic_start
                    print(f"Lines/sec: {self.line_count/elapsed:.1f} | Bytes/sec: {self.byte_count/elapsed:.0f} bytes")
                
                nr_tries = 0
                return ret
            
            # read bytes
            i = max(1, min(4096, self.in_waiting))
            data = self.read(i)
            self.byte_count += len(data)  # Count ALL bytes received
            
            if len(data) == 0:
                nr_tries += 1
                if nr_tries > max_tries:
                    raise TimeoutError("Timeout while reading from serial port")
                continue
            
            self.inputbuffer += data

    def read_sample(self) -> Optional[List[float]]:
        """Parse with success/failure tracking"""
        if not self.is_data_available():
            return None
        
        try:
            line = self.readLine().decode().strip()
            
            # Parse the "%lu,%f\n" format
            parts = line.split(',')
            if len(parts) != 2:
                self.parse_failures += 1
                return None
            
            value = float(parts[1])
            self.parse_success += 1

            self._sps_window_count += 1
            now = time.perf_counter()
            elapsed = now - self._sps_window_start
            if elapsed >= self._sps_update_period_s:
                self._sps_last_rate = self._sps_window_count / elapsed
                self._sps_window_start = now
                self._sps_window_count = 0
            
            # Log stats every 100 successful parses
            if self.parse_success % 100 == 0:
                total = self.parse_success + self.parse_failures
                fail_rate = (self.parse_failures / total * 100) if total > 0 else 0
                print(f"Parse success: {self.parse_success} | Failures: {self.parse_failures} ({fail_rate:.1f}%)")
            
            return [value]
            
        except (ValueError, IndexError):
            self.parse_failures += 1
            return None
        except:
            self.parse_failures += 1
            return None

    def readLine(self, max_tries=1) -> bytearray:
        """
        Read a line from the serial device.
        :return: The line read from the serial device.
        """
        if not self.is_open:
            raise ConnectionError("Connection is not open.")

        data = self._readline(max_tries=max_tries)

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

    def readBytes(self, length: int) -> bytearray:
        """Reads a number of bytes from the serial bus

        :param length: the number of bytes to read
        :return: the bytes read"""
        data = bytearray()
        if self.is_open:
            data = self.read(length)

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
            self.port = port
            self.open()
            self.is_connected = self.is_open
            return self.is_connected
        except SerialTimeoutException as e:
            raise TimeoutError(f"Connection timeout: {e}")
        except SerialException as e:
            raise ConnectionError(f"Serial connection error: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from serial port (HardwareDriver interface)."""
        if self.is_open:
            self.close()
        self.is_connected = False
    
    def write_command(self, command: str) -> Optional[str]:
        """
        Send command and read response (HardwareDriver interface).
        
        Args:
            command: Command string to send (without terminator)
            
        Returns:
            Response string, or None on error
        """
        if not self.is_open:
            raise ConnectionError("Connection is not open.")
        
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
            raise TimeoutError(f"Write timeout.")
        except:
            raise
    
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