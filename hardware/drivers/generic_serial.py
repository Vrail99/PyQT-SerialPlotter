"""
Generic serial driver: thin wrapper around SerialDevice for CSV-based devices.
"""

from hardware.base import HardwareProfile
from hardware.drivers.serial_device import SerialDevice


class GenericSerialDriver(SerialDevice):
    """
    Generic driver for CSV-streaming serial devices.

    Delegates everything to SerialDevice which already implements the full
    HardwareDriver interface.
    """

    def __init__(self, profile: HardwareProfile) -> None:
        super().__init__(
            port=None,
            baudrate=profile.baudrate,
            timeout=profile.timeout,
            terminationCharacter=(
                profile.terminator.encode()
                if isinstance(profile.terminator, str)
                else profile.terminator
            ),
            profile=profile,
        )
