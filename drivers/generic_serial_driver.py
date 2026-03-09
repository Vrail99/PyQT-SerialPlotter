"""
Generic serial driver - compatibility wrapper for SerialDevice.

This driver provides backward compatibility by simply instantiating SerialDevice
with a HardwareProfile. The SerialDevice class now directly implements the
HardwareDriver interface, so this wrapper is just for convenience.
"""

from typing import Optional, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware_driver import HardwareDriver, HardwareProfile
from drivers.serialdevice import SerialDevice


class GenericSerialDriver(SerialDevice):
    """
    Generic driver for simple CSV-based serial devices.
    
    This is now just a thin wrapper around SerialDevice for backward compatibility.
    SerialDevice directly implements the HardwareDriver interface, so this class
    simply passes the profile to SerialDevice.__init__().
    
    Compatible with the original SerialPlotter implementation.
    """
    
    def __init__(self, profile: HardwareProfile):
        """
        Initialize generic serial driver.
        
        Args:
            profile: HardwareProfile with configuration
        """
        # Pass profile to SerialDevice which now handles HardwareDriver interface
        super().__init__(
            port=None,
            baudrate=profile.baudrate,
            timeout=profile.timeout,
            terminationCharacter=profile.terminator.encode() if isinstance(profile.terminator, str) else profile.terminator,
            profile=profile
        )
