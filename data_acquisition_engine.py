"""
Data acquisition engine for serial data reading.

Handles the main update loop and emits signals for data samples.
"""

import time
from typing import Optional, List
from PySide6.QtCore import QObject, Signal as pyqtSignal
from connection_manager import ConnectionManager
from error_handler import ErrorHandler
from file_stream_manager import FileStreamManager
from serial.serialutil import SerialException

class DataAcquisitionEngine(QObject):
    """
    Manages data acquisition from connected hardware.
    
    Reads samples and emits signals for decoupled processing.
    
    Signals:
        sampleReceived: Emitted when a new sample arrives (timestamp, values)
        sampleRateUpdated: Emitted with current sample rate
        acquisitionError: Emitted on errors
    """
    
    # Signals
    sampleReceived = pyqtSignal(float, list)  # timestamp, values
    sampleRateUpdated = pyqtSignal(float)  # sample rate
    acquisitionError = pyqtSignal(str)  # error message
    
    def __init__(self, connection_manager: ConnectionManager,
                 error_handler: ErrorHandler,
                 num_channels: int):
        """
        Initialize acquisition engine.
        
        Args:
            connection_manager: Connection manager instance
            error_handler: Error handler instance
            num_channels: Number of data channels
        """
        super().__init__()
        self.connection_manager = connection_manager
        self.error_handler = error_handler
        self.num_channels = num_channels
        
        # Sample tracking
        self.sample_count = 0
        self.last_rate_update = time.time()
        self.rate_update_interval = 1.0  # Update every second
        self.samples_in_interval = 0
    
    def read_available_samples(self) -> int:
        """
        Read all available samples from device.
        
        Returns:
            Number of samples read
        """
        if not self.connection_manager.is_connected():
            return 0
        
        samples_processed = 0
        
        try:
            while self.connection_manager.is_data_available():
                # Read sample from driver
                values = self.connection_manager.read_sample()
                
                if values is None:
                    continue
                
                # Record timestamp
                timestamp = time.time()
                
                # Filter to configured channels
                channel_values = values[:self.num_channels]
                
                # Emit signal for all subscribers
                self.sampleReceived.emit(timestamp, channel_values)
                
                # Track samples for rate calculation
                self.sample_count += 1
                self.samples_in_interval += 1
                
                samples_processed += 1
                
                # Update sample rate periodically
                current_time = time.time()
                if current_time - self.last_rate_update >= self.rate_update_interval:
                    rate = self.samples_in_interval / (current_time - self.last_rate_update)
                    self.sampleRateUpdated.emit(rate)
                    self.last_rate_update = current_time
                    self.samples_in_interval = 0
                
        except ValueError as e:
            error_msg = f"DAQ: Data parsing error: {e}"
            self.error_handler.warning(error_msg)
            self.acquisitionError.emit(error_msg)
            
        except SerialException as e:
            error_msg = f"DAQ: SerialException: {e}"
            self.error_handler.error(error_msg)
            self.acquisitionError.emit(error_msg)
            self.connection_manager.disconnect()
        
        except TimeoutError as e:
            error_msg = f"DAQ: TimeoutError: {e}"
            self.error_handler.warning(error_msg)
            self.acquisitionError.emit(error_msg)
        
        return samples_processed
    
    def reset_stats(self) -> None:
        """Reset acquisition statistics."""
        self.sample_count = 0
        self.last_rate_update = time.time()
        self.samples_in_interval = 0
    
    def get_sample_count(self) -> int:
        """Get total samples acquired."""
        return self.sample_count
