"""
SerialDevice: pyserial wrapper that implements the HardwareDriver interface.
"""

import time
import serial
import serial.tools.list_ports
from serial.serialutil import SerialException, SerialTimeoutException
from typing import Optional, List

from hardware.base import HardwareDriver, HardwareProfile

class SerialDevice(serial.Serial, HardwareDriver):
    def __init__(self, port=None, baudrate: int = 115200, timeout: float = 1,
                 terminationCharacter: bytes = b"\n",
                 profile: Optional[HardwareProfile] = None):
        serial.Serial.__init__(self, port, baudrate, timeout=timeout, write_timeout=timeout)

        if profile:
            HardwareDriver.__init__(self, profile)
            self.baudrate = profile.baudrate
            self.timeout = profile.timeout
            self.terminationCharacter = (
                profile.terminator.encode()
                if isinstance(profile.terminator, str)
                else profile.terminator
            )
        else:
            self.profile = None
            self.terminationCharacter = terminationCharacter

        self.write_timeout = timeout
        self.inputbuffer = bytearray()

    # ─── Legacy helpers ───────────────────────────────────────────────────

    def open_connection(self, port: str) -> None:
        try:
            self.port = port
            self.open()
        except SerialException as e:
            raise TimeoutError(f"Connection timeout: {e}")

    def close_connection(self) -> str:
        if self.is_open:
            self.close()
            return f"Connection on port {self.port} closed."
        return "No open connection to close."

    def setTimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def setBaudrate(self, baudrate: int) -> None:
        self.baudrate = baudrate

    @staticmethod
    def list_devices() -> List[tuple]:
        return [(p.device, p.description) for p in serial.tools.list_ports.comports()]

    def writeCommand(self, content: str, length: int = 20) -> None:
        if not self.is_open:
            raise ConnectionError("Connection is not open.")
        if len(content) > length:
            raise ValueError("Content length exceeds fixed length.")
        self.write((content.ljust(length) + "\n").encode("utf-8"))

    def _readline(self, max_tries: int = 1) -> bytearray:
        nr_tries = 0
        while True:
            idx = self.inputbuffer.find(self.terminationCharacter)
            if idx > 0:
                ret = self.inputbuffer[:idx]
                del self.inputbuffer[:idx + 1]
                nr_tries = 0
                return ret
            i = max(1, min(4096, self.in_waiting))
            data = self.read(i)
            if len(data) == 0:
                nr_tries += 1
                if nr_tries > max_tries:
                    raise TimeoutError("Timeout while reading from serial port")
                continue
            self.inputbuffer += data

    def readLine(self, max_tries: int = 1) -> bytearray:
        if not self.is_open:
            raise ConnectionError("Connection is not open.")
        return self._readline(max_tries=max_tries)

    def readBytes(self, length: int) -> bytearray:
        return bytearray(self.read(length)) if self.is_open else bytearray()

    def getInWaiting(self) -> int:
        return self.in_waiting + len(self.inputbuffer)

    # ─── HardwareDriver interface ─────────────────────────────────────────

    def connect(self, port: str) -> bool:
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
        if self.is_open:
            self.close()
        self.is_connected = False

    def read_sample(self) -> Optional[List[float]]:
        if not self.is_data_available():
            return None

        try:
            line = self.readLine().decode().strip()
            separator = ","
            data = [float(v.strip()) for v in line.split(separator) if v.strip()]

            return data
        except SerialTimeoutException:
            raise TimeoutError(f"{self.profile.name}: Read timeout.")
        except SerialException as e:
            raise ConnectionError(f"{self.profile.name}: Serial error: {e}")
        except Exception:
            raise

    def write_command(self, command: str) -> Optional[str]:
        if not self.is_open:
            raise ConnectionError("Connection is not open.")
        terminator = self.profile.terminator if self.profile else "\n"
        self.write((command + terminator).encode("utf-8"))
        try:
            return self.readLine().decode().strip()
        except SerialTimeoutException as e:
            raise TimeoutError(f"Write timeout: {e}")

    def is_data_available(self) -> bool:
        return self.is_open and self.getInWaiting() > 0

    def flush(self) -> None:
        if not self.is_open:
            raise ConnectionError("Connection is not open.")
        self.reset_input_buffer()
        self.reset_output_buffer()
        self.inputbuffer = bytearray()

    def set_baudrate(self, baudrate: int) -> None:
        if self.profile:
            self.profile.baudrate = baudrate
        self.baudrate = baudrate

    def list_available_ports(self) -> List[tuple]:
        return self.list_devices()

    def get_device_info(self) -> dict:
        info = {
            "driver": "SerialDevice",
            "baudrate": self.baudrate,
            "timeout": self.timeout,
            "port": self.port if hasattr(self, "port") else "Unknown",
            "is_open": self.is_open,
        }
        if self.profile:
            info["profile_name"] = self.profile.name
        return info