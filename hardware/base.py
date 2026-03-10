"""
Hardware driver abstraction: base class and hardware profile.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class HardwareProfile:
    """
    Hardware configuration profile.

    Holds all parameters needed for a specific device: communication settings,
    command definitions, and data-format specifications.
    """
    name: str
    driver_class: str
    baudrate: int = 115200
    timeout: float = 1.0
    terminator: str = "\n"
    return_on_init: Optional[str] = None
    commands: Dict[str, str] = field(default_factory=dict)
    data_format: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.data_format:
            self.data_format = {"type": "csv", "separator": ",", "scale_factors": []}


class HardwareDriver(ABC):
    """
    Abstract base class for hardware communication drivers.

    All drivers must implement this interface to be compatible with the
    SerialPlotter application.
    """

    def __init__(self, profile: HardwareProfile) -> None:
        self.profile = profile
        self.is_connected = False

    @abstractmethod
    def connect(self, port: str) -> bool:
        """Connect to hardware on the given port. Returns True on success."""

    def initialize(self) -> bool:
        """Perform any necessary initialization after connecting, e.g. handshake."""
        if self.profile.return_on_init is None:
            return True
        if self.profile.commands.get("initialize", '') != None:
            response = self.write_command(self.profile.commands["initialize"])
            if response != self.profile.return_on_init:
                return False
        return True

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect and release resources."""

    @abstractmethod
    def read_sample(self) -> Optional[List[float]]:
        """Read one complete data sample. Returns list of channel values or None."""

    @abstractmethod
    def write_command(self, command: str) -> Optional[str]:
        """Send a command and return the response string, or None on failure."""

    @abstractmethod
    def is_data_available(self) -> bool:
        """Return True if data can be read immediately."""

    @abstractmethod
    def flush(self) -> None:
        """Flush input/output buffers."""
