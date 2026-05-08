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
    commands: Dict[str, Any] = field(default_factory=dict)
    output_data_format: Dict[str, Any] = field(default_factory=dict)
    input_data_format: Dict[str, Any] = field(default_factory=dict)
    datalines: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.output_data_format:
            self.output_data_format = {"type": "csv", "separator": ",", "scale_factors": []}
        if not self.input_data_format:
            self.input_data_format = {"type": "csv", "separator": ",", "scale_factors": []}
        if not self.datalines:
            self.datalines = [{"name": "Line 0", "index": 0, "visible": True}]


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
        init_command = self._resolve_static_command(self.profile.commands.get("initialize"))
        if init_command is not None:
            response = self.write_command(init_command)
            if response != self.profile.return_on_init:
                return False
        return True

    @staticmethod
    def _resolve_static_command(command_spec: Any) -> Optional[str]:
        """Resolve static command specs used outside the command dialog."""
        if command_spec is None:
            return None
        if isinstance(command_spec, str):
            return command_spec
        if isinstance(command_spec, dict):
            command = command_spec.get("command")
            if command is not None:
                return str(command)
            template = command_spec.get("template")
            if isinstance(template, str) and "{value}" not in template:
                return template
        return None

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
