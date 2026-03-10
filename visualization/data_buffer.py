"""
Data buffer management for multi-channel time-series acquisition.
"""

import numpy as np
from typing import List, Optional, Tuple


class DataBufferManager:
    """
    Circular numpy buffers for multiple channels.

    Samples are appended in O(1) time by overwriting the oldest element.
    Read APIs still return values ordered from oldest to newest.
    """

    def __init__(self, num_channels: int, buffer_size: int) -> None:
        self.num_channels = num_channels
        self.buffer_size = buffer_size
        self.x_data: List[np.ndarray] = [np.arange(buffer_size) for _ in range(num_channels)]
        self.y_data: List[np.ndarray] = [np.ones(buffer_size) for _ in range(num_channels)]
        self._oldest_index: List[int] = [0 for _ in range(num_channels)]

    def _ordered_channel_data(self, channel: int) -> np.ndarray:
        """Return channel data ordered from oldest to newest."""
        start = self._oldest_index[channel]
        data = self.y_data[channel]
        if start == 0:
            return data
        return np.concatenate((data[start:], data[:start]))

    def append_sample(self, channel: int, value: float, smoothing_alpha: float = 1.0) -> None:
        if not (0 <= channel < self.num_channels):
            return

        write_idx = self._oldest_index[channel]
        prev_idx = (write_idx - 1) % self.buffer_size

        if smoothing_alpha < 1.0:
            prev = self.y_data[channel][prev_idx]
            self.y_data[channel][write_idx] = value * smoothing_alpha + (1 - smoothing_alpha) * prev
        else:
            self.y_data[channel][write_idx] = value

        self._oldest_index[channel] = (write_idx + 1) % self.buffer_size

    def get_channel_data(self, channel: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not (0 <= channel < self.num_channels):
            return None, None
        return self.x_data[channel], self._ordered_channel_data(channel)

    def get_all_y_data(self) -> List[np.ndarray]:
        return [self._ordered_channel_data(i) for i in range(self.num_channels)]

    def resize_buffers(self, new_size: int) -> None:
        self.buffer_size = new_size
        for i in range(self.num_channels):
            old = self._ordered_channel_data(i)
            new_y = np.zeros(new_size)
            copy_size = min(len(old), new_size)
            new_y[:copy_size] = old[:copy_size]
            self.x_data[i] = np.arange(new_size)
            self.y_data[i] = new_y
            self._oldest_index[i] = 0

    def clear(self) -> None:
        for i in range(self.num_channels):
            self.x_data[i] = np.arange(self.buffer_size)
            self.y_data[i] = np.ones(self.buffer_size)
            self._oldest_index[i] = 0

    def get_buffer_size(self) -> int:
        return self.buffer_size

    def get_num_channels(self) -> int:
        return self.num_channels
