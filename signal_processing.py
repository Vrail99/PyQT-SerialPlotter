"""
Signal processing utilities for time-series data.

This module provides FFT calculation and filtering functions.
"""

import numpy as np
from typing import Tuple


class FFTCalculator:
    """
    Calculates Fast Fourier Transform for frequency analysis.
    """
    
    @staticmethod
    def calculate_fft(signal: np.ndarray, timestep: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate FFT of a signal.
        
        Args:
            signal: Input time-domain signal
            timestep: Time between samples (in seconds)
            
        Returns:
            Tuple of (magnitudes, frequencies) - positive frequencies only
        """
        N = len(signal)
        
        # Calculate FFT
        fft_result = np.fft.fft(signal)
        fft_magnitude = 2 * np.abs(fft_result / N)
        
        # Calculate frequency bins
        fft_freq = np.fft.fftfreq(N, d=timestep)
        
        # Return only positive frequencies (ignore DC and negative)
        pos_mask = fft_freq > 0
        
        return fft_magnitude[pos_mask], fft_freq[pos_mask]
    
    @staticmethod
    def calculate_power_spectrum(signal: np.ndarray, timestep: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate power spectral density.
        
        Args:
            signal: Input time-domain signal
            timestep: Time between samples (in seconds)
            
        Returns:
            Tuple of (power, frequencies)
        """
        magnitude, frequencies = FFTCalculator.calculate_fft(signal, timestep)
        power = magnitude ** 2
        return power, frequencies


class SignalFilter:
    """
    Signal filtering utilities.
    """
    
    @staticmethod
    def exponential_smoothing(new_value: float, old_value: float, alpha: float) -> float:
        """
        Apply exponential smoothing to a new data point.
        
        Args:
            new_value: New incoming value
            old_value: Previous smoothed value
            alpha: Smoothing factor (0-1, where 1=no smoothing, 0=maximum smoothing)
            
        Returns:
            Smoothed value
        """
        alpha = np.clip(alpha, 0.0, 1.0)
        return new_value * alpha + (1 - alpha) * old_value
    
    @staticmethod
    def create_fir_highpass_filter(cutoff: float, fs: float, filter_length: int = 101) -> np.ndarray:
        """
        Create FIR high-pass filter coefficients.
        
        Args:
            cutoff: Cutoff frequency in Hz
            fs: Sampling frequency in Hz
            filter_length: Number of filter taps (should be odd)
            
        Returns:
            Filter coefficients array
        """
        # Ensure odd length
        if filter_length % 2 == 0:
            filter_length += 1
        
        # Create time vector
        t = np.arange(-filter_length // 2 + 1, filter_length // 2 + 1)
        
        # Create ideal low-pass filter
        h_low_pass = np.sinc(2 * cutoff * (t / fs))
        
        # Apply window
        window = np.hamming(filter_length)
        h_low_pass_windowed = h_low_pass * window
        
        # Create delta function
        delta_function = np.zeros(filter_length)
        delta_function[filter_length // 2] = 1
        
        # Subtract low-pass from delta to get high-pass
        h_high_pass = delta_function - h_low_pass_windowed
        
        return h_high_pass
    
    @staticmethod
    def apply_fir_filter(signal: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        """
        Apply FIR filter to a signal.
        
        Args:
            signal: Input signal
            coefficients: Filter coefficients
            
        Returns:
            Filtered signal
        """
        return np.convolve(signal, coefficients, mode='same')


class DataProcessor:
    """
    High-level data processing coordinator.
    """
    
    def __init__(self, timestep: float = 1e-3, smoothing_alpha: float = 1.0):
        """
        Initialize data processor.
        
        Args:
            timestep: Time between samples
            smoothing_alpha: Smoothing factor for exponential filter
        """
        self.timestep = timestep
        self.smoothing_alpha = smoothing_alpha
        self.fft_calculator = FFTCalculator()
        self.filter = SignalFilter()
    
    def process_sample(self, new_value: float, previous_value: float) -> float:
        """
        Process a new sample with smoothing.
        
        Args:
            new_value: New incoming value
            previous_value: Previous value
            
        Returns:
            Processed value
        """
        return self.filter.exponential_smoothing(
            new_value, previous_value, self.smoothing_alpha
        )
    
    def calculate_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate frequency spectrum of signal.
        
        Args:
            signal: Time-domain signal
            
        Returns:
            Tuple of (magnitude, frequency)
        """
        return self.fft_calculator.calculate_fft(signal, self.timestep)
    
    def set_timestep(self, timestep: float) -> None:
        """Update timestep for FFT calculations."""
        self.timestep = timestep
    
    def set_smoothing(self, alpha: float) -> None:
        """Update smoothing factor."""
        self.smoothing_alpha = np.clip(alpha, 0.0, 1.0)
