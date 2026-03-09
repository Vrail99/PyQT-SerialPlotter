"""
Data buffer management for multi-channel data acquisition.

This module handles storage and manipulation of time-series data
from multiple channels.
"""

import numpy as np
from typing import List, Optional


class DataBufferManager:
    """
    Manages data buffers for multiple channels.
    
    Uses numpy arrays to store time-series data with rolling updates.
    Note: Current implementation uses np.roll() - not optimized yet.
    """
    
    def __init__(self, num_channels: int, buffer_size: int):
        """
        Initialize data buffer manager.
        
        Args:
            num_channels: Number of data channels to manage
            buffer_size: Size of each buffer (number of samples)
        """
        self.num_channels = num_channels
        self.buffer_size = buffer_size
        
        # Initialize buffers for each channel
        self.x_data: List[np.ndarray] = []
        self.y_data: List[np.ndarray] = []
        
        for i in range(num_channels):
            self.x_data.append(np.arange(buffer_size))
            self.y_data.append(np.ones(buffer_size))
    
    def append_sample(self, channel: int, value: float, smoothing_alpha: float = 1.0) -> None:
        """
        Add a new sample to a channel's buffer.
        
        Args:
            channel: Channel index
            value: New data value
            smoothing_alpha: Smoothing factor (1.0 = no smoothing, 0.0 = maximum smoothing)
        """
        if channel >= self.num_channels or channel < 0:
            return
        
        # Roll the data (shift left)
        self.y_data[channel] = np.roll(self.y_data[channel], -1)
        
        # Apply exponential smoothing if enabled
        if smoothing_alpha < 1.0:
            previous_value = self.y_data[channel][-2]
            smoothed_value = value * smoothing_alpha + (1 - smoothing_alpha) * previous_value
            self.y_data[channel][-1] = smoothed_value
        else:
            self.y_data[channel][-1] = value
    
    def get_channel_data(self, channel: int) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get x and y data for a specific channel.
        
        Args:
            channel: Channel index
            
        Returns:
            Tuple of (x_data, y_data) or (None, None) if invalid channel
        """
        if channel >= self.num_channels or channel < 0:
            return None, None
        
        return self.x_data[channel], self.y_data[channel]
    
    def get_all_y_data(self) -> List[np.ndarray]:
        """Get all Y data arrays."""
        return self.y_data
    
    def resize_buffers(self, new_size: int) -> None:
        """
        Resize all buffers, preserving existing data where possible.
        
        Args:
            new_size: New buffer size
        """
        self.buffer_size = new_size
        
        for i in range(self.num_channels):
            old_data = self.y_data[i]
            old_size = len(old_data)
            
            # Create new arrays
            new_x = np.arange(new_size)
            new_y = np.zeros(new_size)
            
            # Copy old data (up to the minimum of old and new size)
            copy_size = min(old_size, new_size)
            new_y[:copy_size] = old_data[:copy_size]
            
            # Update buffers
            self.x_data[i] = new_x
            self.y_data[i] = new_y
    
    def clear(self) -> None:
        """Reset all buffers to initial state."""
        for i in range(self.num_channels):
            self.x_data[i] = np.arange(self.buffer_size)
            self.y_data[i] = np.ones(self.buffer_size)
    
    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return self.buffer_size
    
    def get_num_channels(self) -> int:
        """Get number of channels."""
        return self.num_channels
