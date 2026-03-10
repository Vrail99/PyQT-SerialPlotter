"""
Signal processing utilities: FFT calculation, FIR filtering, and exponential smoothing.
"""

import numpy as np
from typing import Tuple


class FFTCalculator:
    """Computes the Fast Fourier Transform for frequency analysis."""

    @staticmethod
    def calculate_fft(signal: np.ndarray, timestep: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (magnitudes, frequencies) for positive frequencies only."""
        N = len(signal)
        fft_magnitude = 2 * np.abs(np.fft.fft(signal) / N)
        fft_freq = np.fft.fftfreq(N, d=timestep)
        pos = fft_freq > 0
        return fft_magnitude[pos], fft_freq[pos]

    @staticmethod
    def calculate_power_spectrum(signal: np.ndarray, timestep: float) -> Tuple[np.ndarray, np.ndarray]:
        magnitude, frequencies = FFTCalculator.calculate_fft(signal, timestep)
        return magnitude ** 2, frequencies


class SignalFilter:
    """Signal filtering utilities."""

    @staticmethod
    def exponential_smoothing(new_value: float, old_value: float, alpha: float) -> float:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return new_value * alpha + (1 - alpha) * old_value

    @staticmethod
    def create_fir_highpass_filter(cutoff: float, fs: float, filter_length: int = 101) -> np.ndarray:
        if filter_length % 2 == 0:
            filter_length += 1
        t = np.arange(-filter_length // 2 + 1, filter_length // 2 + 1)
        h_lp = np.sinc(2 * cutoff * (t / fs)) * np.hamming(filter_length)
        delta = np.zeros(filter_length)
        delta[filter_length // 2] = 1
        return delta - h_lp

    @staticmethod
    def apply_fir_filter(signal: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        return np.convolve(signal, coefficients, mode="same")


class DataProcessor:
    """High-level data processing coordinator."""

    def __init__(self, timestep: float = 1e-3, smoothing_alpha: float = 1.0) -> None:
        self.timestep = timestep
        self.smoothing_alpha = smoothing_alpha

    def process_sample(self, new_value: float, previous_value: float) -> float:
        return SignalFilter.exponential_smoothing(new_value, previous_value, self.smoothing_alpha)

    def calculate_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return FFTCalculator.calculate_fft(signal, self.timestep)

    def set_timestep(self, timestep: float) -> None:
        self.timestep = timestep

    def set_smoothing(self, alpha: float) -> None:
        self.smoothing_alpha = float(np.clip(alpha, 0.0, 1.0))
