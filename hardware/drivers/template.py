"""
Template for creating custom hardware drivers.

INSTRUCTIONS:
1. Copy this file and rename it (e.g., my_device.py).
2. Rename the class (e.g., MyDeviceDriver).
3. Implement all required methods for your hardware protocol.
4. Create a matching JSON profile in hardware_profiles/.
5. The profile's "driver" field must match the class name exactly.

EXAMPLE USE CASES:
- Binary protocols (non-CSV data)
- Devices requiring initialisation sequences
- Polling-mode devices (request-response, not continuous streaming)
- Devices with special command formats
"""

from typing import Optional, List

from hardware.base import HardwareDriver, HardwareProfile


class TemplateDriver(HardwareDriver):
    """
    Template driver — customise for your hardware.

    Implement each method following the embedded examples.
    """

    def __init__(self, profile: HardwareProfile) -> None:
        super().__init__(profile)
        # TODO: Add device-specific attributes here.
        # self.device = None
        # self.packet_size = 20

    def connect(self, port: str) -> bool:
        """
        Connect to your hardware.

        Example (serial)::

            import serial
            self.device = serial.Serial(port, self.profile.baudrate, timeout=self.profile.timeout)
            self.is_connected = True
            return True

        Example (network)::

            import socket
            self.socket = socket.create_connection((port, 8080))
            self.is_connected = True
            return True
        """
        raise NotImplementedError("Implement connect() for your hardware")

    def disconnect(self) -> None:
        """
        Disconnect and clean up.

        Example::

            if self.device:
                self.device.close()
            self.is_connected = False
        """
        raise NotImplementedError("Implement disconnect() for your hardware")

    def read_sample(self) -> Optional[List[float]]:
        """
        Read one sample from your hardware.

        Example (binary, 4 × 16-bit channels)::

            if not self.is_data_available():
                return None
            import struct
            raw = self.device.read(8)
            ch1, ch2, ch3, ch4 = struct.unpack('<HHHH', raw)
            scale = 5.0 / 65535.0
            return [ch * scale for ch in (ch1, ch2, ch3, ch4)]

        Example (polling / request-response)::

            response = self.write_command(self.profile.commands.get('get_data', 'DATA?'))
            return [float(v) for v in response.split(',')] if response else None
        """
        raise NotImplementedError("Implement read_sample() for your hardware")

    def write_command(self, command: str) -> Optional[str]:
        """
        Send a command and return the response.

        Example::

            self.device.write((command + self.profile.terminator).encode())
            return self.device.readline().decode().strip()
        """
        raise NotImplementedError("Implement write_command() for your hardware")

    def is_data_available(self) -> bool:
        """
        Return True when data is ready to read.

        Example::

            return self.device is not None and self.device.in_waiting > 0
        """
        raise NotImplementedError("Implement is_data_available() for your hardware")

    def flush(self) -> None:
        """
        Flush buffers.

        Example::

            if self.device:
                self.device.reset_input_buffer()
        """
        raise NotImplementedError("Implement flush() for your hardware")

    def initialize(self) -> bool:
        """
        Optional post-connect initialisation (send startup commands, etc.).

        Return True when the device is ready for streaming.
        """
        return True
