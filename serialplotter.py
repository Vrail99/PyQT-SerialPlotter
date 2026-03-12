"""
SerialPlotter - Real-time plotting application for serial data acquisition.
"""

import sys
from pathlib import Path
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QFileDialog
from PySide6.QtCore import QTimer, Slot as pyqtSlot, Signal as pyqtSignal
from pyqtgraph.parametertree import ParameterTree

from core.config import ApplicationConfig, Constants
from core.models import AcquisitionState
from core.error_handler import ErrorHandler
from hardware.manager import HardwareDriverManager
from acquisition.connection_manager import ConnectionManager
from acquisition.acquisition_engine import AcquisitionEngine
from acquisition.file_stream_manager import FileStreamManager
from visualization.data_buffer import DataBufferManager
from visualization.statistics import StatisticsCalculator
from visualization.signal_processing import FFTCalculator, DataProcessor
from visualization.plot_manager import PlotManager
from ui.ui_builder import UIBuilder
from ui.parameter_manager import ParameterManager
from ui.dialogs.driver_config import DriverConfigDialog
from ui.dialogs.widget_loader import WidgetLoaderDialog, WidgetModuleLoader


class SerialPlotter(QWidget):
    """
    Main widget - acts purely as a coordinator between managers.

    All UI creation is delegated to UIBuilder.
    All parameter handling is delegated to ParameterManager.
    All connection logic is delegated to ConnectionManager.
    All file streaming is delegated to FileStreamManager.
    """

    portChanged = pyqtSignal(str)
    acquisitionStarted = pyqtSignal(str)
    acquisitionStopped = pyqtSignal(str)
    sampleLengthChanged = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = ApplicationConfig.from_json_files()
        self.error_handler = ErrorHandler(parent_widget=self)

        # Managers
        self.driver_manager = HardwareDriverManager()
        self.connection_manager = ConnectionManager(self.driver_manager, self.error_handler)
        self.file_stream_manager = FileStreamManager(self.error_handler)

        # Data layer
        num_channels = len(self.config.datalines)
        self.max_plot_length = Constants.DEFAULT_MAX_PLOT_LENGTH
        self.data_buffer = DataBufferManager(num_channels, self.max_plot_length)
        self.statistics_calc = StatisticsCalculator(num_channels)
        self.data_processor = DataProcessor(self.config.plot.timestep, self.config.plot.alpha_filter_value)
        self.fft_calculator = FFTCalculator()
        self.acquisition_engine = AcquisitionEngine(self.connection_manager, self.error_handler, num_channels)

        # Plot
        self.plot_layout = pg.GraphicsLayoutWidget(border='w')
        self.plot_manager = PlotManager(self.plot_layout, self.config.datalines)
        self.plot_manager.initialize_datalines(self.max_plot_length)
        self.plot_manager.set_axis_scale('bottom', self.config.plot.timestep, 'time')
        self.plot_manager.set_axis_scale('left', self.config.plot.y_scaling, 'both')
        self.plot_manager.set_axis_label('left', 'Amplitude', self.config.plot.y_unit, 'both')
        self.plot_manager.set_show_grid(self.config.plot.show_grid)

        # State
        self.is_acquiring = False
        self.state = AcquisitionState.DISCONNECTED
        self.external_widgets = []

        widgets_dir = Path(__file__).resolve().parent / "user_widgets"
        widgets_dir.mkdir(parents=True, exist_ok=True)
        self.widget_loader = WidgetModuleLoader(widgets_dir)

        self.driver_config_dialog = None

        # UI & parameters
        param_tree = ParameterTree()
        self.ui = UIBuilder(self.driver_manager, self.connection_manager, self.config, self.max_plot_length)
        self.param_manager = ParameterManager(
            self.config, self.plot_manager, self.data_buffer,
            self.fft_calculator, self.data_processor, self.file_stream_manager
        )
        self.param_manager.setup(param_tree)
        self.setLayout(self.ui.build(self.plot_layout, param_tree))

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

        self._connect_signals()
        self.ui.initialize_first_profile()

    # --- Signal wiring -------------------------------------------------------

    def _connect_signals(self) -> None:
        # UI -> managers / actions
        self.ui.profileSelected.connect(self.connection_manager.set_profile)
        self.ui.portSelected.connect(self._on_port_selected)
        self.ui.baudrateChanged.connect(self.connection_manager.set_baudrate)
        self.ui.reconnectRequested.connect(self._reconnect)
        self.ui.commandSendRequested.connect(self._send_command)
        self.ui.plotLengthChanged.connect(self._on_plot_length_changed)
        self.ui.clearPlotsRequested.connect(self._clear_plots)
        self.ui.startStopToggled.connect(self.toggle_acquisition)
        self.ui.saveCsvRequested.connect(self.save_to_csv)
        self.ui.driverConfigRequested.connect(self._open_driver_config)
        self.ui.loadExternalWidgetRequested.connect(self._open_widget_loader)

        # ConnectionManager -> UI status
        self.connection_manager.connectionEstablished.connect(self._on_connected)
        self.connection_manager.connectionLost.connect(self._on_disconnected)
        self.connection_manager.portListUpdated.connect(self.ui.update_ports)
        self.connection_manager.profileBaudrateChanged.connect(self.ui.set_baudrate_display)

        # Acquisition engine -> data handlers
        self.acquisition_engine.sampleReceived.connect(self._on_sample_received)
        self.acquisition_engine.sampleRateUpdated.connect(self.param_manager.update_fs)

    # --- Connection handlers -------------------------------------------------
    
    @pyqtSlot()
    def _open_driver_config(self) -> None:
        if self.driver_config_dialog is None:
            self.driver_config_dialog = DriverConfigDialog(self.driver_manager, self)

        # Keep info in sync if profile/connection changed since last open
        self.driver_config_dialog.refresh()

        # Show as non-modal so main window stays interactive
        self.driver_config_dialog.show()
        self.driver_config_dialog.raise_()
        self.driver_config_dialog.activateWindow()

    def _on_port_selected(self, port: str) -> None:
        self.connection_manager.disconnect()
        if port != ' ':
            self.connection_manager.connect(port)

    def _on_connected(self, port: str) -> None:
        self.state = AcquisitionState.CONNECTED
        self.ui.set_connection_status(True)
        self.portChanged.emit(port)

    def _on_disconnected(self) -> None:
        self.state = AcquisitionState.DISCONNECTED
        self.ui.set_connection_status(False)

    def _send_command(self, command: str) -> None:
        success, response = self.connection_manager.send_command(command)
        if not success:
            self.error_handler.error(f"Command failed: {response}")

    def _reconnect(self) -> None:
        port = self.ui.get_current_port()
        if not port or port == ' ':
            self.error_handler.info("Please select a COM port first")
            return
        success, error = self.connection_manager.reconnect(port)
        if not success:
            self.error_handler.error(f"Failed to reconnect: {error}")

    @pyqtSlot()
    def _open_widget_loader(self) -> None:
        if not self.widget_loader.discover_modules():
            WidgetLoaderDialog.show_empty_hint(self)
            return

        module_name, class_name = WidgetLoaderDialog.prompt_selection(self.widget_loader, self)
        if not module_name or not class_name:
            return

        self._load_external_widget(module_name, class_name)

    def _load_external_widget(self, module_name: str, class_name: str) -> None:
        try:
            widget = self.widget_loader.instantiate(module_name, class_name, parent=None)

            set_channels = getattr(widget, "setAvailableChannels", None)
            if callable(set_channels):
                channel_names = [cfg.name for cfg in self.config.datalines]
                set_channels(channel_names)

            mean_requested = getattr(widget, "meanRequested", None)
            if mean_requested is not None and hasattr(mean_requested, "connect"):
                mean_requested.connect(self._provide_channel_mean)

            widget.show()
            self.external_widgets.append(widget)
        except Exception as e:
            self.error_handler.error(
                f"Failed to load widget '{module_name}.{class_name}' from user_widgets: {e}"
            )

    @pyqtSlot(int)
    def _provide_channel_mean(self, channel: int) -> None:
        if not (0 <= channel < len(self.config.datalines)):
            self.error_handler.warning(f"Invalid channel index requested: {channel}")
            return

        _, y = self.data_buffer.get_channel_data(channel)
        if y is None or len(y) == 0:
            return

        mean_value = float(np.mean(y))
        requester = self.sender()

        if requester is not None:
            set_mean_value = getattr(requester, "setMeanValue", None)
            if callable(set_mean_value):
                set_mean_value(mean_value)
                return

            set_channel_mean = getattr(requester, "setChannelMean", None)
            if callable(set_channel_mean):
                set_channel_mean(channel, mean_value)
                return

        for widget in self.external_widgets:
            if hasattr(widget, "setChannelMean"):
                widget.setChannelMean(channel, mean_value)

    # --- Acquisition control -------------------------------------------------

    @pyqtSlot()
    def toggle_acquisition(self) -> None:
        if not self.is_acquiring:
            self._start_acquisition()
        else:
            self._stop_acquisition()

    def _start_acquisition(self) -> None:
        if not self.connection_manager.is_connected():
            self.error_handler.warning("Please connect to a device first.")
            return
        self.is_acquiring = True
        self.state = AcquisitionState.ACQUIRING
        self.ui.set_start_stop_text("Stop DAQ & Plot")
        self.acquisition_engine.reset_stats()
        if self.config.export.stream_to_file:
            self.file_stream_manager.start_streaming(
                self.config.export.output_filename,
                [l.name for l in self.config.datalines]
            )
        self.timer.start(Constants.DEFAULT_TIMER_INTERVAL_MS)
        self.acquisitionStarted.emit("Started")

    def _stop_acquisition(self) -> None:
        self.timer.stop()
        self.connection_manager.flush()
        self.is_acquiring = False
        self.state = AcquisitionState.CONNECTED
        self.ui.set_start_stop_text("Start DAQ & Plot")
        if self.config.export.stream_to_file:
            self.file_stream_manager.stop_streaming()
        self.acquisitionStopped.emit("Stopped")

    # --- Data update loop ----------------------------------------------------

    @pyqtSlot()
    def update_plot(self) -> None:
        if self.acquisition_engine.read_available_samples() == 0:
            return

        for ch in range(len(self.config.datalines)):
            x, y = self.data_buffer.get_channel_data(ch)
            if x is not None and y is not None:
                self.plot_manager.update_line_data(ch, x, y)

        if self.config.plot.show_fft:
            for ch in self.plot_manager.get_visible_channels():
                _, y = self.data_buffer.get_channel_data(ch)
                if y is not None:
                    mag, freq = self.fft_calculator.calculate_fft(y, self.config.plot.timestep)
                    self.plot_manager.update_frequency_data(ch, freq, mag)

        if self.config.plot.show_stats:
            stats = self.statistics_calc.compute_all_statistics(
                self.data_buffer.get_all_y_data(), self.config.plot.timestep
            )
            self.param_manager.update_statistics(stats)

    @pyqtSlot(float, list)
    def _on_sample_received(self, timestamp: float, values: list) -> None:
        if self.file_stream_manager.is_active():
            self.file_stream_manager.write_sample(timestamp, values)
        for ch, value in enumerate(values[:len(self.config.datalines)]):
            self.data_buffer.append_sample(ch, value, self.config.plot.alpha_filter_value)

    def _on_plot_length_changed(self, new_length: int) -> None:
        self.max_plot_length = new_length
        self.data_buffer.resize_buffers(new_length)
        self.plot_manager.clear_plots()
        self.plot_manager.initialize_datalines(new_length)
        self.sampleLengthChanged.emit(new_length)

    def _clear_plots(self) -> None:
        self.data_buffer.clear()
        self.plot_manager.clear_plots()
        self.plot_manager.initialize_datalines(self.max_plot_length)

    # --- CSV export ----------------------------------------------------------

    def save_to_csv(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Save to CSV", "", "CSV Files (*.csv)")
        if not filename:
            return
        try:
            num_channels = len(self.config.datalines)
            outdata = np.zeros((num_channels + 1, self.max_plot_length))
            for ch in range(num_channels):
                x, y = self.data_buffer.get_channel_data(ch)
                if x is not None and y is not None:
                    outdata[ch + 1, :len(y)] = y
                    outdata[0, :len(x)] = x * self.config.plot.timestep
            outdata[0] *= 1000  # -> milliseconds

            visible = [True] + [self.plot_manager.datalines[ch].isVisible() for ch in range(num_channels)]
            header = ["Time[ms]"] + [cfg.name for ch, cfg in enumerate(self.config.datalines) if visible[ch + 1]]
            with open(filename, 'w') as f:
                f.write(";".join(header) + "\n")
                for row in outdata[visible].T:
                    f.write(";".join(str(v).replace(".", ",") for v in row) + "\n")
        except Exception as e:
            self.error_handler.error(f"Failed to save data: {e}")

    # --- Lifecycle -----------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self.is_acquiring:
            self._stop_acquisition()
        self.file_stream_manager.close()
        self.connection_manager.disconnect()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = QWidget()
    layout = QVBoxLayout(window)
    layout.addWidget(SerialPlotter())
    window.setWindowTitle("Serial Plotter")
    window.resize(1280, 1024)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
