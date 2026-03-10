"""
Application state management.

Provides clean state tracking for the acquisition system.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class AcquisitionState(Enum):
    """Enumeration of acquisition states."""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    ACQUIRING = "acquiring"
    ERROR = "error"


@dataclass
class ConnectionInfo:
    """Information about current connection."""
    port: Optional[str] = None
    profile_name: Optional[str] = None
    is_connected: bool = False
    error_message: Optional[str] = None


@dataclass
class AcquisitionInfo:
    """Information about current acquisition."""
    is_acquiring: bool = False
    samples_processed: int = 0
    sample_rate: float = 0.0
