# SerialPlotter - Refactored Architecture

## Overview

The SerialPlotter application has been refactored for improved maintainability with separated concerns across multiple modules.

## File Structure

```
serialplotter.py      # Main application widget (refactored)
config.py             # Configuration management with dataclasses
data_buffer.py        # Multi-channel data buffer management
statistics.py         # Statistical calculations for channels
signal_processing.py  # FFT and filtering utilities
plot_manager.py       # PyQtGraph plot management
serialdevice.py       # Serial port communication (existing)
parameter_config.json # Plot and export settings
dataline_config.json  # Channel configuration
```

## Architecture

### 1. **config.py** - Configuration Management
- **PlotConfig**: Plot display settings (timestep, units, filters)
- **ExportConfig**: Data export settings
- **DatalineConfig**: Per-channel configuration
- **ApplicationConfig**: Complete application configuration
- **Constants**: Application-wide constants

**Benefits:**
- Type-safe configuration with dataclasses
- Autocomplete support in IDEs
- Clear default values
- Single source of truth for settings

### 2. **data_buffer.py** - Data Buffer Management
- **DataBufferManager**: Manages numpy arrays for multi-channel data
- Handles rolling updates (currently using np.roll)
- Supports exponential smoothing filter
- Resizable buffers

**Note:** Currently uses np.roll() - not yet optimized with circular buffers.

### 3. **statistics.py** - Statistical Analysis
- **ChannelStatistics**: Dataclass for statistics results
- **StatisticsCalculator**: Computes min/max/mean/std/slope
- **SampleRateTracker**: Tracks actual sample rate during acquisition

**Benefits:**
- Separated statistics logic from plotting
- Can be tested independently
- Reusable for other data analysis tasks

### 4. **signal_processing.py** - Signal Processing
- **FFTCalculator**: Fast Fourier Transform calculations
- **SignalFilter**: Exponential smoothing and FIR filters
- **DataProcessor**: High-level coordinator for processing

**Benefits:**
- Signal processing isolated from GUI code
- Easy to add new filters
- Can be unit tested with synthetic data

### 5. **plot_manager.py** - Plot Management
- **PlotManager**: Creates and updates pyqtgraph plots
- Manages data lines, legends, axes
- Handles both time-domain and frequency-domain plots

**Benefits:**
- All pyqtgraph code in one place
- Plot updates separated from data management
- Easy to modify visualization without touching data logic

### 6. **serialplotter.py** - Main Application (Refactored)
- Orchestrates all modules
- Handles Qt events and user interactions
- Manages data acquisition loop
- Connects serial data to plots and statistics

**Key Improvements:**
- ~1000 lines reduced to ~400 lines
- Clear separation of concerns
- Fixed typo: `acquisition` (was `aquisition`)
- Removed dead/commented code
- Better method naming and organization

## Design Principles Applied

### 1. **Separation of Concerns**
Each module has a single, well-defined responsibility:
- Configuration → config.py
- Data storage → data_buffer.py
- Statistics → statistics.py
- Signal processing → signal_processing.py
- Visualization → plot_manager.py
- Coordination → serialplotter.py

### 2. **Dependency Injection**
Components receive dependencies rather than creating them:

```python
# Before (tight coupling)
def __init__(self):
    self.serialDevice = SerialDevice(baudrate=921600)

# After (dependency injection ready)
def __init__(self, serial_device=None):
    self.serial_device = serial_device or SerialDevice(baudrate=921600)
```

### 3. **Type Hints**
All new code uses type hints for better IDE support:

```python
def calculate_fft(signal: np.ndarray, timestep: float) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

### 4. **Dataclasses for Configuration**
Replaced dictionaries with dataclasses:

```python
# Before
self.parameters["Plot Parameters"]["timestep"]  # Error-prone

# After
self.config.plot.timestep  # Type-safe, autocomplete works
```

### 5. **Constants Extraction**
Magic numbers moved to Constants class:

```python
class Constants:
    DEFAULT_BAUDRATE = 921600
    DEFAULT_MAX_PLOT_LENGTH = 1000
    STATS_UPDATE_INTERVAL = 1000
```

## Performance Note

**No performance optimizations yet** - focus was on maintainability:
- Still using `np.roll()` (will be optimized to circular buffer later)
- Timer interval still 1ms (can be increased to 16-33ms later)
- FFT calculated every frame (can be throttled later)

See earlier analysis for performance optimization strategy.

## Migration Guide

### Old API → New API

```python
# Accessing serial device
self.serialDevice → self.serial_device

# Configuration
self.parameters["Plot Parameters"]["timestep"] → self.config.plot.timestep

# Data access
self.yData[i] → self.data_buffer.get_channel_data(i)[1]

# Statistics
self.stats[i] → self.statistics_calc.get_statistics(i)

# FFT
self.calcFFT(data, timestep) → self.fft_calculator.calculate_fft(data, timestep)
```

### Signal Name Fixes
- `aquisitionStarted` → `acquisitionStarted`
- `aquisitionStopped` → `acquisitionStopped`

## Testing

Each module can now be tested independently:

```python
# Example: Test FFT calculator
from signal_processing import FFTCalculator
import numpy as np

calc = FFTCalculator()
signal = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000))
magnitude, freq = calc.calculate_fft(signal, 1e-3)
# Assert peak at 50 Hz
```

## Future Enhancements

1. **Performance Optimizations** (see performance analysis)
   - Replace np.roll() with circular buffer
   - Increase timer interval to 16-33ms
   - Throttle FFT calculations
   
2. **Additional Features**
   - Data filtering pipelines
   - Advanced statistics (FFT peak detection, etc.)
   - Multiple plot layouts
   - Custom export formats

3. **Testing**
   - Unit tests for each module
   - Mock serial device for testing
   - Integration tests

## Running the Application

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run application
python serialplotter.py
```

## Dependencies

- PySide6 (Qt for Python)
- pyqtgraph (plotting)
- numpy (data processing)
- Custom serialdevice.py module
