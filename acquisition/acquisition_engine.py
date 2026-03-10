"""
Acquisition engine: reads samples from the connected hardware and emits signals.
"""

import time

from PySide6.QtCore import QObject, Signal as pyqtSignal
from serial.serialutil import SerialException

from acquisition.connection_manager import ConnectionManager
from core.error_handler import ErrorHandler


class AcquisitionEngine(QObject):
    """
    Reads all available samples from the hardware driver and emits signals
    for decoupled downstream processing.

    Signals:
        sampleReceived(float, list): Emitted per sample with (timestamp, values).
        sampleRateUpdated(float):    Emitted periodically with the current sample rate.
        acquisitionError(str):       Emitted on recoverable and fatal errors.
    """

    sampleReceived = pyqtSignal(float, list)
    sampleRateUpdated = pyqtSignal(float)
    acquisitionError = pyqtSignal(str)

    def __init__(self, connection_manager: ConnectionManager,
                 error_handler: ErrorHandler,
                 num_channels: int) -> None:
        super().__init__()
        self.connection_manager = connection_manager
        self.error_handler = error_handler
        self.num_channels = num_channels

        self.sample_count = 0
        self.last_rate_update = time.time()
        self.rate_update_interval = 1.0
        self.samples_in_interval = 0

    def read_available_samples(self) -> int:
        """Read all buffered samples. Returns the number of samples processed."""
        if not self.connection_manager.is_connected():
            return 0

        samples_processed = 0
        try:
            while self.connection_manager.is_data_available():
                values = self.connection_manager.read_sample()
                if values is None:
                    continue

                timestamp = time.time()
                self.sampleReceived.emit(timestamp, values[: self.num_channels])

                self.sample_count += 1
                self.samples_in_interval += 1
                samples_processed += 1

                now = time.time()
                if now - self.last_rate_update >= self.rate_update_interval:
                    rate = self.samples_in_interval / (now - self.last_rate_update)
                    self.sampleRateUpdated.emit(rate)
                    self.last_rate_update = now
                    self.samples_in_interval = 0

        except ValueError as e:
            msg = f"DAQ: Data parsing error: {e}"
            self.error_handler.warning(msg)
            self.acquisitionError.emit(msg)
        except SerialException as e:
            msg = f"DAQ: SerialException: {e}"
            self.error_handler.error(msg)
            self.acquisitionError.emit(msg)
            self.connection_manager.disconnect()
        except TimeoutError as e:
            msg = f"DAQ: TimeoutError: {e}"
            self.error_handler.warning(msg)
            self.acquisitionError.emit(msg)

        return samples_processed

    def reset_stats(self) -> None:
        self.sample_count = 0
        self.last_rate_update = time.time()
        self.samples_in_interval = 0

    def get_sample_count(self) -> int:
        return self.sample_count
