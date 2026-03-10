# SerialPlotter — Architecture

## Overview

SerialPlotter is a real-time serial-data acquisition and plotting application built with
PySide6 and pyqtgraph.  The codebase is organised into responsibility-based Python
packages.  `serialplotter.py` acts as a thin coordinator; everything else lives in one
of the four domain packages below.

## Package Structure

```
serialplotter.py            # Entry point / main coordinator widget

core/
    config.py               # ApplicationConfig, PlotConfig, ExportConfig,
                            #   DatalineConfig, Constants
    models.py               # AcquisitionState (enum), ConnectionInfo,
                            #   AcquisitionInfo (DTOs)
    error_handler.py        # ErrorHandler, ErrorSeverity

acquisition/
    connection_manager.py   # ConnectionManager — driver lifecycle & port selection
    acquisition_engine.py   # AcquisitionEngine — sample reading loop
    file_stream_manager.py  # FileStreamManager — real-time CSV export

visualization/
    data_buffer.py          # DataBufferManager — rolling numpy buffers
    plot_manager.py         # PlotManager — pyqtgraph time & frequency plots
    signal_processing.py    # FFTCalculator, SignalFilter, DataProcessor
    statistics.py           # StatisticsCalculator, ChannelStatistics

ui/
    ui_builder.py           # UIBuilder — widget construction + action signals
    parameter_manager.py    # ParameterManager — ParameterTree management
    dialogs/
        driver_config.py    # DriverConfigDialog — hardware command dialog

hardware/
    base.py                 # HardwareDriver (ABC), HardwareProfile (dataclass)
    manager.py              # HardwareDriverManager — profile loading & driver factory
    drivers/
        serial_device.py    # SerialDevice — pyserial wrapper implementing HardwareDriver
        generic_serial.py   # GenericSerialDriver — thin wrapper for CSV devices
        debug_serial.py     # DebugSerialDriver — diagnostics-enhanced driver
        template.py         # TemplateDriver — starter template for new drivers

hardware_profiles/          # JSON hardware profile files (loaded at runtime)
parameter_config.json       # Default plot / export parameter values
dataline_config.json        # Channel name and visibility configuration
```

## Data Flow

```
Serial port
    └─ SerialDevice (hardware/drivers/serial_device.py)
           └─ ConnectionManager (acquisition/connection_manager.py)
                  └─ AcquisitionEngine.read_available_samples()
                         ├─ sampleReceived signal ──► SerialPlotter._on_sample_received()
                         │                                ├─ FileStreamManager.write_sample()
                         │                                └─ DataBufferManager.append_sample()
                         └─ sampleRateUpdated ──► ParameterManager.update_fs()

QTimer (1 ms) ──► SerialPlotter.update_plot()
    ├─ DataBufferManager.get_channel_data() ──► PlotManager.update_line_data()
    ├─ FFTCalculator.calculate_fft()        ──► PlotManager.update_frequency_data()
    └─ StatisticsCalculator.compute_all()  ──► ParameterManager.update_statistics()
```

## Naming Conventions

| Scope        | Convention                                    |
|------------- |-----------------------------------------------|
| Files        | `snake_case.py`                               |
| Classes      | `PascalCase`                                  |
| Packages     | `snake_case/`                                 |
| Imports      | Absolute package paths (`from core.config …`) |
| No shims     | Old root-level modules have been removed      |

## Adding a New Hardware Driver

1. Copy `hardware/drivers/template.py` to e.g. `hardware/drivers/my_device.py`.
2. Rename the class to e.g. `MyDeviceDriver`.
3. Implement all abstract methods (`connect`, `disconnect`, `read_sample`,
   `write_command`, `is_data_available`, `flush`).
4. Create a JSON profile in `hardware_profiles/` with `"driver": "MyDeviceDriver"`.
5. The manager resolves the module automatically via `_class_to_module()` which strips
   the `Driver` suffix: `MyDeviceDriver` → `hardware.drivers.my_device`.

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
