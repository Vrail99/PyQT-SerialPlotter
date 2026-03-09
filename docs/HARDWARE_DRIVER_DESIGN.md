# Hardware Driver Abstraction Layer - Complete Guide

## Overview

The PyQt-SerialPlotter now supports multiple hardware types through a plugin-style driver system. This allows you to connect different devices with varying communication protocols without modifying the main application code.

## Architecture

```
SerialPlotter (Main Application)
    ↓
HardwareDriverManager (Profile Loader & Factory)
    ↓
HardwareDriver (Abstract Interface)
    ↓
├── GenericSerialDriver (CSV text protocol)
├── Your Custom Drivers (binary, polling, etc.)
└── TemplateDriver (starting point for new drivers)
```

## Quick Start

### 1. Using Existing Profiles

1. Select a hardware profile from the **Profile** dropdown
2. Select the serial port from the **Port** dropdown
3. Click **Start DAQ & Plot**

### 2. Creating a Custom Driver

1. Copy `drivers/template_driver.py` to `drivers/my_device_driver.py`
2. Rename the class: `TemplateDriver` → `MyDeviceDriver`
3. Implement the required methods for your protocol
4. Create a JSON profile in `hardware_profiles/my_device.json`
5. Restart the application

## File Structure

```
Pyqt-SerialPlotter/
├── hardware_driver.py              # Abstract base class
├── hardware_driver_manager.py      # Profile loader & factory
├── drivers/                        # Driver implementations
│   ├── __init__.py
│   ├── generic_serial_driver.py    # For CSV-based devices
│   ├── template_driver.py          # Starting template
│   └── (your custom drivers)
├── hardware_profiles/              # JSON configurations
│   ├── generic_uart.json           # Default configuration
│   ├── custom_adc_example.json     # Example custom device
│   └── (your profiles)
└── serialplotter.py                # Main application
```

## Creating Hardware Profiles

### Profile Structure

A hardware profile is a JSON file that specifies:
- Driver class to use
- Communication parameters (baudrate, timeout, etc.)
- Command definitions
- Data format specification

### Example: Simple CSV Device

File: `hardware_profiles/my_device.json`

```json
{
    "name": "My Device",
    "driver": "GenericSerialDriver",
    "baudrate": 115200,
    "timeout": 1.0,
    "terminator": "\n",
    "commands": {
        "initialize": "INIT",
        "get_value": "READ?"
    },
    "data_format": {
        "type": "csv",
        "separator": ",",
        "scale_factors": [1.0, 1.0, 1.0]
    }
}
```

### Example: Device with Complex Commands

File: `hardware_profiles/advanced_adc.json`

```json
{
    "name": "Advanced ADC Board",
    "driver": "GenericSerialDriver",
    "baudrate": 921600,
    "timeout": 1.0,
    "terminator": "\r\n",
    "commands": {
        "initialize": "INIT",
        "start_streaming": "STREAM:ON",
        "stop_streaming": "STREAM:OFF",
        "set_sample_rate": "RATE:{rate}",
        "calibrate": "CAL:START",
        "reset": "RST",
        "get_status": "STAT?"
    },
    "data_format": {
        "type": "csv",
        "separator": ",",
        "channels": 5,
        "scale_factors": [2.5, 2.5, 2.5, 2.5, 2.5],
        "units": ["V", "V", "V", "V", "V"]
    }
}
```

### Profile Fields Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Display name in UI |
| `driver` | string | Yes | Driver class name (e.g., "GenericSerialDriver") |
| `baudrate` | integer | No | Default: 115200 |
| `timeout` | float | No | Read timeout in seconds, default: 1.0 |
| `terminator` | string | No | Line terminator, default: "\n" |
| `commands` | object | No | Named commands for the device |
| `data_format` | object | No | Data parsing configuration |

#### `data_format` Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | "csv" (currently only supported type) |
| `separator` | string | Column separator, default: "," |
| `scale_factors` | array | Multiply each channel by factor |
| `channels` | integer | Expected number of channels |
| `units` | array | Unit labels for each channel |

## Implementing Custom Drivers

### When to Create a Custom Driver

Create a custom driver if your hardware:
- Uses binary protocol (not text-based CSV)
- Requires polling mode (request-response instead of streaming)
- Has special initialization sequences
- Uses non-standard data formats

### Driver Implementation Steps

#### Step 1: Copy Template

```bash
cp drivers/template_driver.py drivers/my_custom_driver.py
```

#### Step 2: Rename Class

```python
# In my_custom_driver.py
class MyCustomDriver(HardwareDriver):
    """Driver for my custom hardware."""
```

#### Step 3: Implement Required Methods

All drivers **must** implement:

```python
def connect(self, port: str) -> bool:
    """Establish connection to hardware."""
    # Your code here
    
def disconnect(self) -> None:
    """Close connection and cleanup."""
    # Your code here

def read_sample(self) -> Optional[List[float]]:
    """Read one data sample, return list of channel values."""
    # Your code here

def write_command(self, command: str) -> Optional[str]:
    """Send command, return response."""
    # Your code here

def is_data_available(self) -> bool:
    """Check if data is ready to read."""
    # Your code here

def flush(self) -> None:
    """Clear input/output buffers."""
    # Your code here
```

#### Step 4: Optional Methods

Override if needed:

```python
def initialize(self) -> bool:
    """Run startup sequence after connection."""
    # Default: sends 'initialize' command from profile
    return super().initialize()

def start_streaming(self) -> bool:
    """Start continuous data stream."""
    # Default: sends 'start_streaming' command
    return super().start_streaming()

def stop_streaming(self) -> bool:
    """Stop continuous data stream."""
    # Default: sends 'stop_streaming' command
    return super().stop_streaming()
```

#### Step 5: Create Profile

File: `hardware_profiles/my_custom.json`

```json
{
    "name": "My Custom Device",
    "driver": "MyCustomDriver",
    "baudrate": 115200,
    "timeout": 1.0
}
```

### Example: Binary Protocol Driver

```python
# drivers/binary_adc_driver.py
from typing import Optional, List
import serial
import struct
from hardware_driver import HardwareDriver, HardwareProfile

class BinaryADCDriver(HardwareDriver):
    """Driver for ADC with binary protocol."""
    
    PACKET_SIZE = 20
    HEADER = 0xAA
    FOOTER = 0x55
    
    def __init__(self, profile: HardwareProfile):
        super().__init__(profile)
        self.serial = None
    
    def connect(self, port: str) -> bool:
        try:
            self.serial = serial.Serial(
                port=port,
                baudrate=self.profile.baudrate,
                timeout=self.profile.timeout
            )
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        if self.serial and self.serial.is_open:
            self.serial.close()
        self.is_connected = False
    
    def read_sample(self) -> Optional[List[float]]:
        """Read binary packet: [AA][data...][55]"""
        if not self.is_data_available():
            return None
        
        try:
            packet = self.serial.read(self.PACKET_SIZE)
            
            # Validate header and footer
            if packet[0] != self.HEADER or packet[-1] != self.FOOTER:
                return None
            
            # Unpack 4 channels (16-bit unsigned integers)
            ch1, ch2, ch3, ch4 = struct.unpack('<HHHH', packet[1:9])
            
            # Convert to voltages (0-65535 → 0-5V)
            scale = 5.0 / 65535.0
            return [ch1 * scale, ch2 * scale, ch3 * scale, ch4 * scale]
            
        except Exception as e:
            print(f"Parse error: {e}")
            return None
    
    def write_command(self, command: str) -> Optional[str]:
        try:
            self.serial.write((command + '\r\n').encode())
            return self.serial.readline().decode().strip()
        except Exception as e:
            return None
    
    def is_data_available(self) -> bool:
        return (self.serial and self.serial.is_open and 
                self.serial.in_waiting >= self.PACKET_SIZE)
    
    def flush(self) -> None:
        if self.serial and self.serial.is_open:
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
```

### Example: Polling Mode Driver

For devices that don't stream continuously:

```python
# drivers/polling_driver.py
class PollingDriver(HardwareDriver):
    """Driver for devices requiring polling."""
    
    def read_sample(self) -> Optional[List[float]]:
        """Request data sample from device."""
        # Send request command
        cmd = self.profile.commands.get('get_data', 'READ?')
        response = self.write_command(cmd)
        
        if response:
            # Parse response
            values = response.split(',')
            return [float(v) for v in values]
        
        return None
    
    def is_data_available(self) -> bool:
        """In polling mode, always ready to request."""
        return self.is_connected
```

## Usage in Application

### Starting the Application

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Run application
python serialplotter.py
```

### UI Workflow

1. **Select Profile**: Choose hardware type from Profile dropdown
2. **Select Port**: Choose COM port from Port dropdown
3. **Adjust Baudrate**: If needed (auto-filled from profile)
4. **Connect**: Port connects automatically when selected
5. **Start DAQ**: Click "Start DAQ & Plot" button
6. **Send Commands**: Use command input to test device commands
7. **Stop DAQ**: Click "Stop DAQ & Plot" button

### Programmatic Usage

```python
from hardware_driver_manager import HardwareDriverManager

# Create manager
manager = HardwareDriverManager()

# List available profiles
profiles = manager.get_profile_names()
print(f"Available: {profiles}")

# Create driver
driver = manager.create_driver("Generic UART Device")

# Connect and use
if driver.connect("COM3"):
    driver.initialize()
    driver.start_streaming()
    
    # Read data
    while driver.is_data_available():
        values = driver.read_sample()
        print(values)
    
    driver.stop_streaming()
    driver.disconnect()
```

## Testing

### Test with Loopback

Connect TX to RX on your serial adapter to test:

```python
# Send test data
driver.write_command("1.23,4.56,7.89")

# Read it back
values = driver.read_sample()
assert values == [1.23, 4.56, 7.89]
```

### Mock Testing

```python
class MockDriver(HardwareDriver):
    """Mock driver for testing."""
    
    def __init__(self, profile):
        super().__init__(profile)
        self.test_data = [[1.0, 2.0], [3.0, 4.0]]
        self.index = 0
    
    def connect(self, port: str) -> bool:
        self.is_connected = True
        return True
    
    def read_sample(self) -> Optional[List[float]]:
        if self.index < len(self.test_data):
            sample = self.test_data[self.index]
            self.index += 1
            return sample
        return None
    
    # ... implement other methods ...
```

## Troubleshooting

### Profile Not Appearing

**Problem**: Profile JSON exists but doesn't show in dropdown

**Solutions**:
1. Check JSON syntax: `python -m json.tool hardware_profiles/my_profile.json`
2. Verify `name` and `driver` fields are present
3. Check console for error messages
4. Restart application

### Driver Not Found

**Problem**: Error: "Driver module not found"

**Solutions**:
1. Check driver file naming: `MyCustomDriver` → `my_custom_driver.py`
2. Verify driver class name matches profile `driver` field
3. Ensure `__init__.py` exists in `drivers/` directory
4. Check for syntax errors in driver file

### Connection Fails

**Problem**: "Failed to connect to device"

**Solutions**:
1. Verify correct COM port selected
2. Check baudrate matches device
3. Ensure no other program is using the port
4. Verify driver's `connect()` method is correct
5. Check device is powered and connected

### No Data Received

**Problem**: Connected but no data appears

**Solutions**:
1. Check `is_data_available()` returns True
2. Verify `read_sample()` returns valid data
3. Test with device's documentation examples
4. Add debug prints in `read_sample()`
5. Check terminator character matches device

### Parse Errors

**Problem**: "Parse error" in console

**Solutions**:
1. Verify data format matches profile `separator`
2. Check for extra whitespace
3. Validate scale_factors array length
4. Print raw data before parsing
5. Test with simpler data first

## Advanced Features

### Dynamic Command Execution

Use profile commands in your code:

```python
# Get command from profile
init_cmd = driver.profile.commands.get('initialize', 'INIT')
response = driver.write_command(init_cmd)

# Parameterized commands
rate_cmd = driver.profile.commands.get('set_sample_rate', 'RATE:{rate}')
actual_cmd = rate_cmd.format(rate=1000)
driver.write_command(actual_cmd)
```

### Scale Factors

Apply scaling to raw values:

```json
{
    "data_format": {
        "separator": ",",
        "scale_factors": [0.001, 0.001, 1.0, 1.0, 10.0]
    }
}
```

This converts:
- Channels 0-1: millivolts to volts (×0.001)
- Channels 2-3: no scaling (×1.0)
- Channel 4: amplify by 10 (×10.0)

### Custom Initialization Sequences

```python
def initialize(self) -> bool:
    """Multi-step initialization."""
    steps = [
        ('reset', 'RST'),
        ('set_mode', 'MODE:AUTO'),
        ('set_range', 'RANGE:5V'),
        ('calibrate', 'CAL')
    ]
    
    for name, default_cmd in steps:
        cmd = self.profile.commands.get(name, default_cmd)
        response = self.write_command(cmd)
        
        if not response or 'ERROR' in response:
            print(f"Initialization failed at: {name}")
            return False
    
    return True
```

### Protocol Discovery

Auto-detect device protocol:

```python
def detect_protocol(port: str) -> str:
    """Try to identify device type."""
    test_serial = serial.Serial(port, 115200, timeout=0.5)
    
    # Try common identification commands
    for cmd in ['*IDN?', 'ID?', 'VER?']:
        test_serial.write((cmd + '\n').encode())
        response = test_serial.readline().decode()
        if response:
            test_serial.close()
            return response
    
    test_serial.close()
    return "Unknown"
```

## Migration from Old Code

### Before (Direct SerialDevice)

```python
line = self.serial_device.readLine().decode()
values = line.split(',')
for value_str in values:
    data = float(value_str)
    # process data
```

### After (Driver System)

```python
values = self.current_driver.read_sample()  # Already parsed!
for data in values:
    # process data
```

### Benefits

- ✅ No parsing logic in main app
- ✅ Driver handles protocol details
- ✅ Easy to swap hardware
- ✅ Configuration via JSON
- ✅ Reusable drivers

## Best Practices

### 1. Error Handling

Always handle exceptions in driver methods:

```python
def read_sample(self) -> Optional[List[float]]:
    try:
        # Your code
        return data
    except ValueError as e:
        print(f"Parse error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

### 2. Validation

Validate data before returning:

```python
def read_sample(self) -> Optional[List[float]]:
    values = self._parse_data()
    
    # Validate range
    if any(v < 0 or v > 5.0 for v in values):
        print("Data out of range")
        return None
    
    return values
```

### 3. Logging

Use proper logging instead of print:

```python
import logging

logger = logging.getLogger(__name__)

def read_sample(self):
    logger.debug(f"Reading from {self.profile.name}")
    # ...
```

### 4. Documentation

Document your driver:

```python
class MyDriver(HardwareDriver):
    """
    Driver for XYZ Device Model 123.
    
    Protocol: Binary, 20-byte packets
    Baudrate: 921600
    Data: 4 channels, 16-bit unsigned
    
    Packet format:
        [0xAA][CH1_H][CH1_L]...[CH4_H][CH4_L][CHKSUM][0x55]
    
    Commands:
        INIT - Initialize device
        CAL - Start calibration
    """
```

### 5. Profile Comments

Add comments to JSON (if your parser supports):

```json
{
    "name": "My Device",
    "_comment": "Used for lab setup A",
    "driver": "MyDriver",
    "_baudrate_note": "921600 required for high-speed mode"
}
```

## FAQ

**Q: Can I use multiple devices simultaneously?**
A: Currently, the UI supports one device at a time. For multiple devices, you'd need to create multiple SerialPlotter instances or extend the architecture.

**Q: Do I need to restart after adding a profile?**
A: Yes, profiles are loaded at startup. You can call `driver_manager.reload_profiles()` programmatically.

**Q: Can I use non-serial protocols (USB, Ethernet)?**
A: Yes! Implement a driver that uses the appropriate communication library (socket, usb, etc.) instead of pyserial.

**Q: How do I debug my driver?**
A: Add print statements in `read_sample()` and `write_command()`, or use Python's logging module.

**Q: Can drivers have configuration GUIs?**
A: Not built-in, but you could create a separate configuration widget that modifies the JSON profile and reloads.

**Q: What about binary data over 100KB/s?**
A: The driver system handles this fine. Performance depends on your `read_sample()` implementation and the timer interval (currently 1ms).

## Contributing

To contribute a driver for common hardware:

1. Create driver in `drivers/` directory
2. Add example profile in `hardware_profiles/`
3. Test with real hardware
4. Document protocol and known issues
5. Submit pull request

## License

Same as Pyqt-SerialPlotter main project.

## Support

- Check console output for error messages
- Review `template_driver.py` for implementation examples
- See `ARCHITECTURE.md` for overall design
- Open GitHub issue for bugs
