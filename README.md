# Pyqt-SerialPlotter

Pyqt-SerialPlotter is a Python application for real-time plotting of serial data using PyQt. It is designed to visualize data from microcontrollers, sensors, or any device that outputs data over a serial port.

## Features

- Real-time plotting of serial data
- Configurable serial port settings (baud rate, port, etc.)
- Pause/resume plotting
- Save plot data to file
- Customizable plot appearance

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Pyqt-SerialPlotter.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Connect your device to the serial port.
2. Run the application:
    ```bash
    python main.py
    ```
3. Select the serial port and configure settings.
4. Start plotting!

## Binary Gas Widget (AIR -> N2 Workflow)

The `DensityWidget` in `user_widgets/density_calculator_widget.py` can be used to detect composition changes from AIR to N2 using density-based inversion.

### What it does

- Calibrates an orifice-based gas model at current process conditions.
- Uses measured dP and configured flow to estimate density in real time.
- In dedicated AIR/N2 mode, reports:
    - AIR Fraction [%]
    - N2 Fraction [%]

### Setup and run

1. Load the external widget from the `user_widgets` directory.
2. Set gas pair to `AIR` and `N2` (order does not matter).
3. Configure geometry, manual temperature, manual inlet pressure, and normal flow.
4. Select the dP source channel.
5. Set `Reference Heavy Fraction` to match the calibration state:
     - If Light=`AIR`, Heavy=`N2`: enter N2 percent during calibration.
     - If Light=`N2`, Heavy=`AIR`: enter AIR percent during calibration.
6. With a known baseline stream (for example 100% AIR), click `Calibrate`.
7. Click `Start Live Analysis`.
8. Introduce N2 and monitor `AIR Fraction` and `N2 Fraction` labels.

### Practical notes

- For a pure AIR calibration baseline:
    - Light=`AIR`, Heavy=`N2` -> set `Reference Heavy Fraction` to `0%`.
    - Light=`N2`, Heavy=`AIR` -> set `Reference Heavy Fraction` to `100%`.
- Results are displayed in widget labels only; they are not published back as derived SerialPlotter channels.
- Keep temperature and pressure inputs aligned with actual process conditions for best accuracy.

## Requirements

- Python 3.7+
- PyQt5
- pyserial
- matplotlib

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## Acknowledgements

Inspired by similar serial plotter tools and the PyQt community.