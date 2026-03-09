"""
SerialPlotter - Real-time plotting application for serial data acquisition.

Refactored for improved maintainability with separated concerns.
"""

import sys
import time
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QComboBox, QMessageBox, QFileDialog, QGridLayout, QLineEdit, QLabel
)
from PySide6.QtCore import QTimer, Slot as pyqtSlot, Signal as pyqtSignal
from PySide6.QtGui import QIntValidator

# Import our custom modules
from drivers.serialdevice import SerialDevice
from pyqtgraph.parametertree import Parameter, ParameterTree
from config import ApplicationConfig, Constants
from data_buffer import DataBufferManager
from statistics import StatisticsCalculator, SampleRateTracker
from signal_processing import FFTCalculator, DataProcessor
from plot_manager import PlotManager
from hardware_driver_manager import HardwareDriverManager
from drivers.driver_config_dialog import DriverConfigDialog

class SerialPlotter(QWidget):
    """
    SerialPlotter - Main widget for real-time serial data visualization.
    
    Refactored with separated concerns for better maintainability:
    - Configuration management via ApplicationConfig
    - Data buffering via DataBufferManager
    - Statistics via StatisticsCalculator
    - Signal processing via FFTCalculator and DataProcessor
    - Plot management via PlotManager
    
    Signals:
        portChanged: Emitted when serial port changes
        acquisitionStarted: Emitted when data acquisition starts
        acquisitionStopped: Emitted when data acquisition stops
        sampleLengthChanged: Emitted when buffer size changes
    """
    
    # Qt Signals (fixed typo: acquisition not aquisition)
    portChanged = pyqtSignal(str)
    acquisitionStarted = pyqtSignal(str)
    acquisitionStopped = pyqtSignal(str)
    sampleLengthChanged = pyqtSignal(int)

    def __init__(self, positional=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load configuration
        self.config = ApplicationConfig.from_json_files()
        
        # Initialize hardware driver system
        self.driver_manager = HardwareDriverManager()
        self.current_driver = None
        
        # Keep serial device for backward compatibility (fallback)
        self.serial_device = SerialDevice(
            baudrate=Constants.DEFAULT_BAUDRATE,
            timeout=1
        )
        
        # Initialize data management
        num_channels = len(self.config.datalines)
        self.max_plot_length = Constants.DEFAULT_MAX_PLOT_LENGTH
        
        self.data_buffer = DataBufferManager(num_channels, self.max_plot_length)
        self.statistics_calc = StatisticsCalculator(num_channels)
        self.sample_rate_tracker = SampleRateTracker(Constants.STATS_UPDATE_INTERVAL)
        
        # Initialize signal processing
        self.data_processor = DataProcessor(
            timestep=self.config.plot.timestep,
            smoothing_alpha=self.config.plot.alpha_filter_value
        )
        self.fft_calculator = FFTCalculator()
        
        # Initialize plotting
        self.plot_layout = pg.GraphicsLayoutWidget(border='w')
        self.plot_manager: PlotManager = None  # Will be initialized after UI setup
        
        # State variables
        self.is_acquiring = False
        self.output_file = None
        self.command = "VAL?"
        
        # UI components
        self.param_tree = ParameterTree()
        self.params = None
        
        # Timer for plot updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        
        # Setup UI and plots
        self._setup_ui()
        self._initialize_plots()
        self._setup_parameter_tree()

    # ========================
    # Setup and Initialization
    # ========================
    
    def _setup_ui(self) -> None:
        """Initialize all UI components."""
        # Main layout
        main_layout = QHBoxLayout()
        
        # Right side - plots and controls
        plot_control_layout = QVBoxLayout()
        plot_control_layout.addWidget(self.plot_layout)
        plot_control_layout.addWidget(self._create_control_panel())
        plot_control_layout.addLayout(self._create_serial_controls())
        plot_control_layout.addWidget(self._create_plot_length_control())
        plot_control_layout.addWidget(self._create_command_panel())
        plot_control_layout.addWidget(self._create_action_buttons())
        
        # Add parameter tree and plot area to main layout
        main_layout.addWidget(self.param_tree, stretch=1)
        main_layout.addLayout(plot_control_layout, stretch=4)
        
        self.setLayout(main_layout)
    
    def _create_control_panel(self) -> QWidget:
        """Create the control panel widget."""
        widget = QWidget()
        layout = QGridLayout()
        widget.setLayout(layout)
        return widget
    
    def _create_serial_controls(self) -> QHBoxLayout:
        """Create serial port and baudrate controls."""
        layout = QHBoxLayout()
        
        # Hardware profile dropdown
        profile_label = QLabel("Profile:")
        self.profile_dropdown = QComboBox()
        profile_names = self.driver_manager.get_profile_names()
        if profile_names:
            self.profile_dropdown.addItems(profile_names)
        else:
            self.profile_dropdown.addItem("No profiles found")
        
        # Port dropdown
        port_label = QLabel("Port:")
        self.port_dropdown = QComboBox()
        self.port_dropdown.enterEvent = self._populate_ports
        self._populate_ports(None)
        self.port_dropdown.currentIndexChanged.connect(self._on_port_changed)

        # Connection status indicator button
        self.connection_status_button = QPushButton("●")
        self.connection_status_button.setFixedSize(35, 25)
        self.connection_status_button.setToolTip("Disconnected - Click to reconnect")
        self.connection_status_button.setStyleSheet(
            "QPushButton { background-color: #cc0000; color: white; font-size: 16px; font-weight: bold; border-radius: 3px; }"
            "QPushButton:hover { background-color: #ff0000; }"
        )
        self.connection_status_button.clicked.connect(self._reconnect_device)
        
        # Baudrate dropdown
        baud_label = QLabel("Baud:")
        self.baudrate_dropdown = QComboBox()
        self.baudrate_dropdown.addItems(["921600", "115200", "9600"])
        self.baudrate_dropdown.setCurrentText(str(Constants.DEFAULT_BAUDRATE))
        self.baudrate_dropdown.setToolTip("Baudrate")
        self.baudrate_dropdown.currentTextChanged.connect(self._on_baudrate_changed)

        # Driver configuration button
        self.config_button = QPushButton("Driver Config")
        self.config_button.clicked.connect(self._open_driver_config)
        self.config_button.setToolTip("Open driver configuration and command interface")
        
        layout.addWidget(profile_label)
        layout.addWidget(self.profile_dropdown)
        layout.addWidget(self.config_button)
        layout.addWidget(port_label)
        layout.addWidget(self.port_dropdown)
        layout.addWidget(self.connection_status_button)
        layout.addWidget(baud_label)
        layout.addWidget(self.baudrate_dropdown)
        
        # Connect profile change handler after widgets are created
        self.profile_dropdown.currentTextChanged.connect(self._on_profile_changed)
        
        # Initialize first profile if available
        if profile_names:
            self._on_profile_changed(profile_names[0])
        
        return layout
    
    def _create_plot_length_control(self) -> QWidget:
        """Create plot length input control."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        label = QLabel("Max Plot Length:")
        self.plot_length_input = QLineEdit(str(self.max_plot_length))
        self.plot_length_input.setValidator(QIntValidator(1, 100000))
        self.plot_length_input.editingFinished.connect(self._on_plot_length_changed)
        
        layout.addWidget(label)
        layout.addWidget(self.plot_length_input)
        widget.setLayout(layout)
        
        return widget
    
    def _create_command_panel(self) -> QWidget:
        """Create command input panel."""
        widget = QWidget()
        layout = QHBoxLayout()
        
        self.command_input = QLineEdit(self.command)
        send_button = QPushButton("Send")
        send_button.clicked.connect(self._send_command)
        clear_button = QPushButton("Clear Plot Data")
        clear_button.clicked.connect(self._clear_plots)
        
        layout.addWidget(self.command_input)
        layout.addWidget(send_button)
        layout.addWidget(clear_button)
        widget.setLayout(layout)
        
        return widget
    
    def _create_action_buttons(self) -> QWidget:
        """Create main action buttons."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.start_stop_button = QPushButton("Start DAQ & Plot")
        self.start_stop_button.clicked.connect(self.toggle_acquisition)
        
        self.save_button = QPushButton("Save to CSV")
        self.save_button.clicked.connect(self.save_to_csv)

        layout.addWidget(self.start_stop_button)
        layout.addWidget(self.save_button)

        widget.setLayout(layout)
        
        return widget
    
    def _initialize_plots(self) -> None:
        """Initialize the plot manager and create all data lines."""
        self.plot_manager = PlotManager(self.plot_layout, self.config.datalines)
        self.plot_manager.initialize_datalines(self.max_plot_length)
        
        # Apply initial settings
        self.plot_manager.set_axis_scale('bottom', self.config.plot.timestep, 'time')
        self.plot_manager.set_axis_scale('left', self.config.plot.y_scaling, 'both')
        self.plot_manager.set_axis_label('left', 'Amplitude', self.config.plot.y_unit, 'both')
        self.plot_manager.set_show_grid(self.config.plot.show_grid)
    
    def _setup_parameter_tree(self) -> None:
        """Setup the parameter tree for live configuration."""
        # Create statistics parameters
        stats_params = []
        for line_cfg in self.config.datalines:
            stats_params.append({
                'name': line_cfg.name,
                'type': 'group',
                'children': [
                    {'name': 'Min', 'type': 'float', 'value': float('inf'), 'readonly': True},
                    {'name': 'Max', 'type': 'float', 'value': float('-inf'), 'readonly': True},
                    {'name': 'Mean', 'type': 'float', 'value': 0.0, 'readonly': True, 'decimals': 4},
                    {'name': 'Std', 'type': 'float', 'value': 0.0, 'readonly': True, 'decimals': 4},
                    {'name': 'Slope', 'type': 'float', 'value': 0.0, 'readonly': True, 'decimals': 1},
                ]
            })
        
        # Combine config parameters with statistics
        all_params = self.config.to_parameter_tree_format() + [
            {'name': 'Statistics', 'type': 'group', 'children': stats_params}
        ]
        
        self.params = Parameter.create(name='Parameters', type='group', children=all_params)
        self.params.sigTreeStateChanged.connect(self._on_parameter_changed)
        self.param_tree.setParameters(self.params, showTop=False)

    # =======================
    # Parameter Event Handlers
    # =======================
    
    def _on_parameter_changed(self, param, changes) -> None:
        """Handle parameter changes from the UI."""
        for param_, _,data in changes:
            path = self.params.childPath(param_)
            
            if path is None:
                continue
            
            # Skip statistics (read-only)
            if path[0] == "Statistics":
                continue
            
            param_name = '.'.join(path)
            
            # Dispatch to specific handlers
            if param_name == 'Plot Parameters.show_fft':
                self._handle_fft_toggle(data)
            elif param_name == 'Plot Parameters.show_points':
                self.plot_manager.set_show_points(data)
            elif param_name == 'Plot Parameters.show_grid':
                self.plot_manager.set_show_grid(data)
            elif param_name == 'Plot Parameters.timestep':
                self._handle_timestep_change(data)
            elif param_name == 'Plot Parameters.y-Scaling':
                self.plot_manager.set_axis_scale('left', data, 'both')
                self.config.plot.y_scaling = data
            elif param_name == 'Plot Parameters.y-Unit':
                self.plot_manager.set_axis_label('left', 'Amplitude', data, 'both')
                self.config.plot.y_unit = data
            elif param_name == 'Plot Parameters.Alpha-Filter-Value':
                self.data_processor.set_smoothing(data)
                self.config.plot.alpha_filter_value = data
            elif param_name == 'Export Settings.stream_to_file':
                self._handle_stream_toggle(data)
    
    def _handle_fft_toggle(self, show: bool) -> None:
        """Handle FFT display toggle."""
        self.config.plot.show_fft = show
        
        if show:
            # Calculate and show FFT for all visible channels
            for channel in self.plot_manager.get_visible_channels():
                _, y_data = self.data_buffer.get_channel_data(channel)
                magnitude, freq = self.fft_calculator.calculate_fft(
                    y_data, self.config.plot.timestep
                )
                self.plot_manager.update_frequency_data(channel, freq, magnitude)
        
        self.plot_manager.set_fft_visibility(show)
    
    def _handle_timestep_change(self, timestep: float) -> None:
        """Handle timestep parameter change."""
        self.config.plot.timestep = timestep
        self.config.plot.fs = int(1 / timestep) if timestep > 0 else 0
        self.data_processor.set_timestep(timestep)
        self.plot_manager.set_axis_scale('bottom', timestep, 'time')
        
        # Update fs display
        self.params.child("Plot Parameters").child("fs").setValue(self.config.plot.fs)
    
    def _handle_stream_toggle(self, enable: bool) -> None:
        """Handle file streaming toggle."""
        if enable:
            self._start_file_streaming()
        else:
            self._stop_file_streaming()
    
    def _start_file_streaming(self) -> None:
        """Start streaming data to file."""
        filename = self.config.export.output_filename
        try:
            self.output_file = open(filename, "w+")
            # Write header
            header = "Time"
            for line_cfg in self.config.datalines:
                header += f",{line_cfg.name}"
            self.output_file.write(header + "\n")
            self.config.export.stream_to_file = True
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open file: {e}")
            self.config.export.stream_to_file = False
    
    def _stop_file_streaming(self) -> None:
        """Stop streaming data to file."""
        if self.output_file and not self.output_file.closed:
            self.output_file.flush()
            self.output_file.close()
        self.config.export.stream_to_file = False


    # =====================
    # Serial Port Handling
    # =====================
    
    def _on_profile_changed(self, profile_name: str) -> None:
        """Handle hardware profile selection."""
        if profile_name == "No profiles found":
            return
        
        # Disconnect current driver if any
        if self.current_driver and self.current_driver.is_connected:
            dc_succ, dc_err = self.current_driver.safe_disconnect()
            if dc_err:
                QMessageBox.warning(self, "Error", f"Error disconnecting current driver: {dc_err}")
        
        # Create new driver
        self.current_driver = self.driver_manager.create_driver(profile_name)
        
        if self.current_driver:
            # Update baudrate dropdown to match profile
            self.driver_manager.set_current_driver(self.current_driver)

            self.baudrate_dropdown.setCurrentText(str(self.current_driver.profile.baudrate))
            print(f"Selected profile: {profile_name}")

            self._update_connection_status()
        else:
            QMessageBox.warning(self, "Error", f"Failed to create driver for {profile_name}")
    
    def _populate_ports(self, event) -> None:
        """Populate serial ports dropdown."""
        # Use driver's method if available, otherwise fall back to SerialDevice
        if self.current_driver:
            ports = self.current_driver.list_available_ports()
        else:
            ports = self.serial_device.list_devices()
        
        ports.insert(0, (" ", "Disconnect"))
        
        if len(ports) != self.port_dropdown.count():
            self.port_dropdown.clear()
            self.port_dropdown.addItems([f"{p[0]}-({p[1]})" for p in ports])
    
    @pyqtSlot()
    def _on_port_changed(self) -> None:
        """Handle serial port selection change."""
        port = self.port_dropdown.currentText().split('-')[0]
        
        if port == " ":
            # Disconnect driver or fallback serial device
            if self.current_driver:
                dc_succ, dc_err = self.current_driver.safe_disconnect()
                if dc_err:
                    QMessageBox.warning(self, "Error", f"Error disconnecting current driver: {dc_err}")
            else:
                self.serial_device.close()
            self.portChanged.emit(" ")
            self._update_connection_status()
            return
        
        if not self.current_driver:
            QMessageBox.warning(self, "Error", "Please select a hardware profile first.")
            return
        
        try:
            success = self.current_driver.connect(port)
            if success:
                self.current_driver.initialize()
                self.portChanged.emit(port)
                print(f"Connected to {port}")
            else:
                QMessageBox.warning(self, "Error", "Failed to connect to device")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
        finally:
            self._update_connection_status()
    
    def _on_baudrate_changed(self, baudrate_str: str) -> None:
        """Handle baudrate change."""
        baudrate = int(baudrate_str)
        if self.current_driver:
            self.current_driver.set_baudrate(baudrate)
        else:
            self.serial_device.setBaudrate(baudrate)
    
    def _send_command(self) -> None:
        """Send command to hardware."""
        if not self.current_driver or not self.current_driver.is_connected:
            QMessageBox.warning(self, "Error", "Not connected to hardware")
            return
        
        command = str(self.command_input.text())
        try:
            response = self.current_driver.write_command(command)
            print(f"Command: {command}")
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error sending command: {e}")

    def _update_connection_status(self) -> None:
        """Update the connection status indicator button."""
        if self.current_driver and self.current_driver.is_connected:
            # Connected - Green
            self.connection_status_button.setStyleSheet(
                "QPushButton { background-color: #00cc00; color: white; font-size: 16px; font-weight: bold; border-radius: 3px; }"
                "QPushButton:hover { background-color: #00ff00; }"
            )
            self.connection_status_button.setToolTip("Connected - Click to reconnect")
        else:
            # Disconnected - Red
            self.connection_status_button.setStyleSheet(
                "QPushButton { background-color: #cc0000; color: white; font-size: 16px; font-weight: bold; border-radius: 3px; }"
                "QPushButton:hover { background-color: #ff0000; }"
            )
            self.connection_status_button.setToolTip("Disconnected - Click to reconnect")
    
    def _reconnect_device(self) -> None:
        """Reconnect to the currently selected device."""
        if not self.current_driver:
            QMessageBox.warning(self, "Error", "No hardware profile selected")
            return
        
        current_port = self.port_dropdown.currentText().split('-')[0]
        
        if current_port == " " or not current_port:
            QMessageBox.information(self, "Info", "Please select a COM port first")
            return
        
        # Disconnect if currently connected
        if self.current_driver.is_connected:
            print("Disconnecting...")
            dc_succ, dc_err = self.current_driver.safe_disconnect()
            if dc_err:
                QMessageBox.warning(self, "Error", f"Error disconnecting current driver: {dc_err}")
            self._update_connection_status()
        
        # Attempt to reconnect
        print(f"Reconnecting to {current_port}...")

        success, error = self.current_driver.safe_connect(current_port)
        if not success:
            QMessageBox.warning(self, "Error", f"Failed to reconnect to device: {error}")
            self._update_connection_status()
            return
            
        init_success, init_error = self.current_driver.safe_initialize()
        if init_success:
            self.portChanged.emit(current_port)
            self._update_connection_status()
            return
        
        if init_error:
            QMessageBox.warning(self, "Initialization Error",
                                init_error + "\n\nPossible causes:\n"
                                +"- Wrong baud rate selected\n"
                                +"- Device not responding\n"
                                +"- Device not powered on")
                
        dc_succ, dc_err = self.current_driver.safe_disconnect()
        if dc_err:
            QMessageBox.warning(self, "Error", f"Error disconnecting current driver: {dc_err}")


    # ========================
    # Data Acquisition Control
    # ========================
    
    @pyqtSlot()
    def toggle_acquisition(self) -> None:
        """Start or stop data acquisition and plotting."""
        if not self.is_acquiring:
            self._start_acquisition()
        else:
            self._stop_acquisition()
    
    def _start_acquisition(self) -> None:
        """Start data acquisition."""
        # Check if driver is connected
        if not self.current_driver or not self.current_driver.is_connected:
            QMessageBox.warning(self, "Error", "Please connect to a device first.")
            return
        
        self.start_stop_button.setText("Stop DAQ & Plot")
        self.is_acquiring = True
        self.acquisitionStarted.emit("Started")
        self.sample_rate_tracker.reset()
        
        # Start timer
        self.timer.start(Constants.DEFAULT_TIMER_INTERVAL_MS)
        
        # Open output file if streaming enabled
        if self.config.export.stream_to_file:
            self._start_file_streaming()
    
    def _stop_acquisition(self) -> None:
        """Stop data acquisition."""
        self.start_stop_button.setText("Start DAQ & Plot")
        self.timer.stop()
        
        # Flush driver if connected
        if self.current_driver and self.current_driver.is_connected:
            self.current_driver.flush()
        
        self.is_acquiring = False
        self.acquisitionStopped.emit("Stopped")
        
        # Close output file if open
        if self.config.export.stream_to_file:
            self._stop_file_streaming()
    
    def _on_plot_length_changed(self) -> None:
        """Handle plot length change."""
        try:
            new_length = int(self.plot_length_input.text())
            self.max_plot_length = new_length
            self.data_buffer.resize_buffers(new_length)
            
            # Reinitialize plots with new size
            self.plot_manager.clear_plots()
            self.plot_manager.initialize_datalines(new_length)
            
            self.sampleLengthChanged.emit(new_length)
        except ValueError:
            pass
    
    def _clear_plots(self) -> None:
        """Clear all plot data."""
        self.data_buffer.clear()
        self.plot_manager.clear_plots()
        self.plot_manager.initialize_datalines(self.max_plot_length)
    
    # ===================
    # Data Update Methods
    # ===================
    
    @pyqtSlot()
    def update_plot(self) -> None:
        """Main update loop - read serial data and update plots."""
        # Read all available data from serial port
        samples_processed = self._read_serial_data()
        
        if samples_processed == 0:
            return
        
        # Update plot displays
        self._update_plot_displays()
        
        # Update FFT if enabled
        if self.config.plot.show_fft:
            self._update_fft_displays()
        
        # Update statistics if enabled
        if self.config.plot.show_stats:
            self._update_statistics_display()
    
    def _read_serial_data(self) -> int:
        """Read and process available serial data."""
        if not self.current_driver or not self.current_driver.is_connected:
            return 0
        
        samples_count = 0
        
        while self.current_driver.is_data_available():
            try:
                # Read sample from driver (already parsed!)
                values = self.current_driver.read_sample()
                
                if values is None:
                    continue
                
                # Prepare file output if streaming
                file_line = None
                if self.config.export.stream_to_file:
                    file_line = f"{time.time()}"
                
                # Process each channel
                for channel, data_value in enumerate(values):
                    if channel >= len(self.config.datalines):
                        break
                    
                    # Add to file line if streaming
                    if file_line is not None:
                        file_line += f",{data_value}"
                    
                    # Store in buffer with smoothing
                    self.data_buffer.append_sample(
                        channel, 
                        data_value, 
                        self.config.plot.alpha_filter_value
                    )
                
                # Write to file if streaming
                if file_line is not None and self.output_file:
                    self.output_file.write(file_line + "\n")
                
                # Track sample rate
                if self.sample_rate_tracker.add_sample(time.time()):
                    sample_rate = self.sample_rate_tracker.get_sample_rate()
                    self.params.child("Plot Parameters").child("fs").setValue(sample_rate)
                
                samples_count += 1
                
            except ValueError as e:
                print(f"Data parsing error: {e}")
            except Exception as e:
                print(f"Serial read error: {e}")
        
        return samples_count
    
    def _update_plot_displays(self) -> None:
        """Update time-domain plot displays."""
        for channel in range(len(self.config.datalines)):
            x_data, y_data = self.data_buffer.get_channel_data(channel)
            if x_data is not None and y_data is not None:
                self.plot_manager.update_line_data(channel, x_data, y_data)
    
    def _update_fft_displays(self) -> None:
        """Update frequency-domain (FFT) plot displays."""
        for channel in self.plot_manager.get_visible_channels():
            _, y_data = self.data_buffer.get_channel_data(channel)
            if y_data is not None:
                magnitude, freq = self.fft_calculator.calculate_fft(
                    y_data, self.config.plot.timestep
                )
                self.plot_manager.update_frequency_data(channel, freq, magnitude)
    
    def _update_statistics_display(self) -> None:
        """Update statistics display in parameter tree."""
        all_data = self.data_buffer.get_all_y_data()
        stats_list = self.statistics_calc.compute_all_statistics(
            all_data, self.config.plot.timestep
        )
        
        # Update parameter tree
        stats_group = self.params.child("Statistics")
        for i, stats in enumerate(stats_list):
            if i < len(stats_group.children()):
                channel_stats = stats_group.children()[i]
                channel_stats.child('Min').setValue(stats.min_value)
                channel_stats.child('Max').setValue(stats.max_value)
                channel_stats.child('Mean').setValue(stats.mean)
                channel_stats.child('Std').setValue(stats.std)
                channel_stats.child('Slope').setValue(stats.slope)
        
    # ==================
    # Data Export Methods
    # ==================
    
    def save_to_csv(self) -> None:
        """Save plotted data to CSV file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save to CSV", "", "CSV Files (*.csv)"
        )
        if not filename:
            return
        
        try:
            self._export_to_csv(filename)
            QMessageBox.information(self, "Success", f"Data saved to {filename}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save data: {e}")
    
    def _export_to_csv(self, filename: str) -> None:
        """Export current plot data to CSV file."""
        # Prepare data array
        num_channels = len(self.config.datalines)
        outdata = np.zeros((num_channels + 1, self.max_plot_length))
        
        # Collect data from each visible channel
        for channel in range(num_channels):
            x_data, y_data = self.data_buffer.get_channel_data(channel)
            if x_data is not None and y_data is not None:
                outdata[channel + 1, :len(y_data)] = y_data
                outdata[0, :len(x_data)] = x_data * self.config.plot.timestep
        
        # Convert time to milliseconds
        outdata[0, :] = outdata[0, :] * 1000
        
        # Build header
        header = ["Time[ms]"]
        visible_filter = [True]  # Always include time column
        
        for channel, line_cfg in enumerate(self.config.datalines):
            visible = self.plot_manager.datalines[channel].isVisible()
            visible_filter.append(visible)
            if visible:
                header.append(line_cfg.name)
        
        # Write to file
        with open(filename, 'w') as f:
            # Write header
            f.write(";".join(header) + "\n")
            
            # Write data rows
            filtered_data = outdata[visible_filter, :]
            for row in filtered_data.T:
                # Format with comma as decimal separator (European style)
                row_str = ";".join(str(val).replace(".", ",") for val in row)
                f.write(row_str + "\n")
    
    # ===================
    # Utility Methods
    # ===================
    
    def get_serial_device(self):
        """Get the serial device instance (for external access)."""
        return self.serial_device
    
    def get_max_plot_length(self) -> int:
        """Get current maximum plot length."""
        return self.max_plot_length

    
    def _open_driver_config(self) -> None:
        """Open the driver configuration dialog."""
        dialog = DriverConfigDialog(self.driver_manager, self)
        dialog.exec()
    
    def closeEvent(self, event) -> None:
        """Handle widget close event."""
        # Stop acquisition if running
        if self.is_acquiring:
            self._stop_acquisition()
        
        # Disconnect driver if connected
        if self.current_driver and self.current_driver.is_connected:
            dc_succ, dc_err = self.current_driver.safe_disconnect()
            if dc_err:
                QMessageBox.warning(self, "Error", f"Error disconnecting current driver: {dc_err}")
        
        event.accept()


def main():
    """Main entry point for standalone application."""
    app = QApplication(sys.argv)
    
    window = QWidget()
    layout = QVBoxLayout(window)
    
    # Create serial plotter
    serial_plotter = SerialPlotter()
    
    layout.addWidget(serial_plotter)
    window.setWindowTitle("Serial Plotter")
    window.resize(1280, 1024)
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
