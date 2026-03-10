"""
Statistical analysis for time-series data channels.
"""

import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class ChannelStatistics:
    """Statistics for a single data channel."""
    min_value: float = float("inf")
    max_value: float = float("-inf")
    mean: float = 0.0
    std: float = 0.0
    slope: float = 0.0


class StatisticsCalculator:
    """Computes statistics across multiple data channels."""

    def __init__(self, num_channels: int) -> None:
        self.num_channels = num_channels
        self.stats: List[ChannelStatistics] = [ChannelStatistics() for _ in range(num_channels)]

    def compute_statistics(self, data: np.ndarray, channel: int,
                           timestep: float = 1.0) -> ChannelStatistics:
        if not (0 <= channel < self.num_channels):
            return ChannelStatistics()
        stats = ChannelStatistics(
            min_value=float(np.min(data)),
            max_value=float(np.max(data)),
            mean=float(np.mean(data)),
            std=float(np.std(data)),
            slope=self._slope(data, timestep),
        )
        self.stats[channel] = stats
        return stats

    def compute_all_statistics(self, data_list: List[np.ndarray],
                               timestep: float = 1.0) -> List[ChannelStatistics]:
        return [
            self.compute_statistics(data, i, timestep)
            for i, data in enumerate(data_list)
            if i < self.num_channels
        ]

    @staticmethod
    def _slope(data: np.ndarray, timestep: float) -> float:
        if len(data) < 2:
            return 0.0
        total_time = len(data) * timestep
        return float((data[-1] - data[0]) / total_time) if total_time else 0.0

    def get_statistics(self, channel: int) -> ChannelStatistics:
        if not (0 <= channel < self.num_channels):
            return ChannelStatistics()
        return self.stats[channel]

    def get_all_statistics(self) -> List[ChannelStatistics]:
        return self.stats
