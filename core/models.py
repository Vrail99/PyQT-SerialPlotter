"""
Shared state and data-transfer objects.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class AcquisitionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    ACQUIRING = "acquiring"
    ERROR = "error"


@dataclass
class ConnectionInfo:
    port: Optional[str] = None
    profile_name: Optional[str] = None
    is_connected: bool = False
    error_message: Optional[str] = None


@dataclass
class AcquisitionInfo:
    is_acquiring: bool = False
    samples_processed: int = 0
    sample_rate: float = 0.0
