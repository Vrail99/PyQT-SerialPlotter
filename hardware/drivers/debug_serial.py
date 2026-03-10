"""
Debug serial driver: diagnostics-enhanced driver for development and testing.
"""

import time
import serial
import serial.tools.list_ports
from serial.serialutil import SerialException, SerialTimeoutException
from typing import Optional, List

from hardware.base import HardwareDriver, HardwareProfile

samples_parsed = 0
parse_times = []


class DebugSerialDriver(serial.Serial, HardwareDriver):
    def __init__(self, profile: HardwareProfile) -> None:
        serial.Serial.__init__(self)
        HardwareDriver.__init__(self, profile)

        self.baudrate = profile.baudrate
        self.timeout = profile.timeout
        self.write_timeout = profile.timeout
        self.terminationCharacter = (
            profile.terminator.encode()
            if isinstance(profile.terminator, str)
            else profile.terminator
        )
        self.inputbuffer = bytearray()

        # Diagnostics counters
        self.byte_count = 0
        self.line_count = 0
        self.parse_success = 0
        self.parse_failures = 0
        self.diagnostic_start = time.perf_counter()

        self._sps_last_rate: Optional[float] = None
        self._sps_window_start = time.perf_counter()
        self._sps_window_count = 0
        self._sps_update_period_s = 1.0

    # ─── Diagnostics helpers ──────────────────────────────────────────────

    def get_measured_sps(self) -> Optional[float]:
        return self._sps_last_rate

    def reset_sps_measurement(self) -> None:
        self._sps_last_rate = None
        self._sps_window_start = time.perf_counter()
        self._sps_window_count = 0

    # ─── Legacy helpers ───────────────────────────────────────────────────

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

    def _readline(self, max_tries: int = 1) -> bytearray:
        nr_tries = 0
        while True:
            idx = self.inputbuffer.find(self.terminationCharacter)
            if idx > 0:
                ret = self.inputbuffer[:idx]
                del self.inputbuffer[:idx + 1]
                self.line_count += 1
                self.byte_count += len(ret) + 1
                if self.line_count % 100 == 0:
                    elapsed = time.perf_counter() - self.diagnostic_start
                    print(f"Lines/sec: {self.line_count/elapsed:.1f} | Bytes/sec: {self.byte_count/elapsed:.0f}")
                nr_tries = 0
                return ret
            i = max(1, min(4096, self.in_waiting))
            data = self.read(i)
            self.byte_count += len(data)
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
            parts = line.split(",")
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

            if self.parse_success % 100 == 0:
                total = self.parse_success + self.parse_failures
                fail_rate = (self.parse_failures / total * 100) if total > 0 else 0
                print(f"Parse success: {self.parse_success} | Failures: {self.parse_failures} ({fail_rate:.1f}%)")

            return [value]
        except (ValueError, IndexError):
            self.parse_failures += 1
            return None
        except Exception:
            self.parse_failures += 1
            return None

    def write_command(self, command: str) -> Optional[str]:
        if not self.is_open:
            raise ConnectionError("Connection is not open.")
        terminator = self.profile.terminator if self.profile else "\n"
        self.write((command + terminator).encode("utf-8"))
        try:
            return self.readLine().decode().strip()
        except SerialTimeoutException as e:
            raise TimeoutError(f"Write timeout: {e}")
        except Exception:
            raise

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
        return {
            "driver": "DebugSerialDriver",
            "baudrate": self.baudrate,
            "timeout": self.timeout,
            "port": self.port if hasattr(self, "port") else "Unknown",
            "is_open": self.is_open,
            "profile_name": self.profile.name if self.profile else None,
        }

    def initialize(self) -> bool:
        return True
