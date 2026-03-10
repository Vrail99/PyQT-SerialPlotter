"""
Data buffer management for multi-channel time-series acquisition.
"""

import numpy as np
from typing import List, Optional, Tuple


class DataBufferManager:
    """
    Rolling numpy buffers for multiple channels.

    Uses ``np.roll`` to shift data on each new sample.
    """

    def __init__(self, num_channels: int, buffer_size: int) -> None:
        self.num_channels = num_channels
        self.buffer_size = buffer_size
        self.x_data: List[np.ndarray] = [np.arange(buffer_size) for _ in range(num_channels)]
        self.y_data: List[np.ndarray] = [np.ones(buffer_size) for _ in range(num_channels)]

    def append_sample(self, channel: int, value: float, smoothing_alpha: float = 1.0) -> None:
        if not (0 <= channel < self.num_channels):
            return
        self.y_data[channel] = np.roll(self.y_data[channel], -1)
        if smoothing_alpha < 1.0:
            prev = self.y_data[channel][-2]
            self.y_data[channel][-1] = value * smoothing_alpha + (1 - smoothing_alpha) * prev
        else:
            self.y_data[channel][-1] = value

    def get_channel_data(self, channel: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not (0 <= channel < self.num_channels):
            return None, None
        return self.x_data[channel], self.y_data[channel]

    def get_all_y_data(self) -> List[np.ndarray]:
        return self.y_data

    def resize_buffers(self, new_size: int) -> None:
        self.buffer_size = new_size
        for i in range(self.num_channels):
            old = self.y_data[i]
            new_y = np.zeros(new_size)
            copy_size = min(len(old), new_size)
            new_y[:copy_size] = old[:copy_size]
            self.x_data[i] = np.arange(new_size)
            self.y_data[i] = new_y

    def clear(self) -> None:
        for i in range(self.num_channels):
            self.x_data[i] = np.arange(self.buffer_size)
            self.y_data[i] = np.ones(self.buffer_size)

    def get_buffer_size(self) -> int:
        return self.buffer_size

    def get_num_channels(self) -> int:
        return self.num_channels
