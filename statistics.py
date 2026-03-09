"""
Statistical analysis for time-series data.

This module provides utilities for computing statistics on data channels.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ChannelStatistics:
    """Statistics for a single data channel."""
    min_value: float = float('inf')
    max_value: float = float('-inf')
    mean: float = 0.0
    std: float = 0.0
    slope: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            'min': self.min_value,
            'max': self.max_value,
            'mean': self.mean,
            'std': self.std,
            'slope': self.slope
        }


class StatisticsCalculator:
    """
    Calculates statistics for multiple data channels.
    """
    
    def __init__(self, num_channels: int):
        """
        Initialize statistics calculator.
        
        Args:
            num_channels: Number of channels to track
        """
        self.num_channels = num_channels
        self.stats: List[ChannelStatistics] = [
            ChannelStatistics() for _ in range(num_channels)
        ]
    
    def compute_statistics(self, data: np.ndarray, channel: int, timestep: float = 1.0) -> ChannelStatistics:
        """
        Compute statistics for a single channel.
        
        Args:
            data: Data array for the channel
            channel: Channel index
            timestep: Time between samples (for slope calculation)
            
        Returns:
            ChannelStatistics object with computed values
        """
        if channel >= self.num_channels or channel < 0:
            return ChannelStatistics()
        
        stats = ChannelStatistics(
            min_value=float(np.min(data)),
            max_value=float(np.max(data)),
            mean=float(np.mean(data)),
            std=float(np.std(data)),
            slope=self._calculate_slope(data, timestep)
        )
        
        self.stats[channel] = stats
        return stats
    
    def compute_all_statistics(self, data_list: List[np.ndarray], timestep: float = 1.0) -> List[ChannelStatistics]:
        """
        Compute statistics for all channels.
        
        Args:
            data_list: List of data arrays, one per channel
            timestep: Time between samples
            
        Returns:
            List of ChannelStatistics objects
        """
        results = []
        for i, data in enumerate(data_list):
            if i < self.num_channels:
                results.append(self.compute_statistics(data, i, timestep))
        return results
    
    @staticmethod
    def _calculate_slope(data: np.ndarray, timestep: float) -> float:
        """
        Calculate slope between first and last data points.
        
        Args:
            data: Data array
            timestep: Time between samples
            
        Returns:
            Slope value (change per unit time)
        """
        if len(data) < 2:
            return 0.0
        
        total_time = len(data) * timestep
        if total_time == 0:
            return 0.0
        
        return float((data[-1] - data[0]) / total_time)
    
    def get_statistics(self, channel: int) -> ChannelStatistics:
        """
        Get cached statistics for a channel.
        
        Args:
            channel: Channel index
            
        Returns:
            ChannelStatistics object
        """
        if channel >= self.num_channels or channel < 0:
            return ChannelStatistics()
        
        return self.stats[channel]
    
    def get_all_statistics(self) -> List[ChannelStatistics]:
        """Get statistics for all channels."""
        return self.stats


class SampleRateTracker:
    """
    Tracks actual sample rate during data acquisition.
    """
    
    def __init__(self, update_interval: int = 1000):
        """
        Initialize sample rate tracker.
        
        Args:
            update_interval: Number of samples between rate calculations
        """
        self.update_interval = update_interval
        self.sample_count = 0
        self.last_update_time = None
        self.current_rate = 0.0
    
    def add_sample(self, current_time: float) -> bool:
        """
        Record a sample and check if rate should be updated.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            True if sample rate was recalculated, False otherwise
        """
        self.sample_count += 1
        
        if self.last_update_time is None:
            self.last_update_time = current_time
            return False
        
        if self.sample_count >= self.update_interval:
            # Calculate rate
            dt = current_time - self.last_update_time
            if dt > 0:
                self.current_rate = self.sample_count / dt
            
            # Reset counters
            self.sample_count = 0
            self.last_update_time = current_time
            return True
        
        return False
    
    def get_sample_rate(self) -> float:
        """Get current sample rate in samples per second."""
        return self.current_rate
    
    def reset(self) -> None:
        """Reset the tracker."""
        self.sample_count = 0
        self.last_update_time = None
        self.current_rate = 0.0
