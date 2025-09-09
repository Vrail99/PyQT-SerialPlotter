import serial
import serial.tools.list_ports
#from fast_readline import ReadLine
import threading  # Add threading for continuous reading

class SerialDevice(serial.Serial):
    def __init__(self, port=None, baudrate=115200, timeout=15, terminationCharacter=b'\n'):
        """
        Initialize the SerialDevice object.
        :param port: The serial port to connect to (e.g., 'COM3' or '/dev/ttyUSB0').
        :param baudrate: The baud rate for the serial connection.
        :param timeout: The timeout for the serial connection in seconds.
        """
        super().__init__(port, baudrate, timeout=timeout)
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.linereader = ReadLine(self)
        self.linereader.setTerminationCharacter(terminationCharacter)

        self.inputbuffer = bytearray()
        self.terminationCharacter = b"\n"

   
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
        except:
            print("Oh oh")
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
        

    def _readline(self) -> bytearray:
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
                if nr_tries > 1:
                    raise serial.SerialTimeoutException("Timeout while reading from serial port")
                continue
            self.inputbuffer += data # add data to buffer

    def readLine(self):
        """
        Read a line from the serial device.
        :return: The line read from the serial device.
        """
        if not self.is_open:
            raise ConnectionError("Connection is not open.")
        return self._readline()
    
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