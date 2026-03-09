"""
Plot management for data visualization.

This module handles creation and updating of pyqtgraph plots.
"""

import pyqtgraph as pg
import numpy as np
from typing import List, Optional, Tuple
from config import DatalineConfig


class PlotManager:
    """
    Manages pyqtgraph plots for multi-channel data visualization.
    """
    
    def __init__(self, plot_layout: pg.GraphicsLayoutWidget, dataline_configs: List[DatalineConfig]):
        """
        Initialize plot manager.
        
        Args:
            plot_layout: PyQtGraph graphics layout widget
            dataline_configs: Configuration for each data line
        """
        self.plot_layout = plot_layout
        self.dataline_configs = dataline_configs
        
        # Create plot items
        self.live_plot: Optional[pg.PlotItem] = None
        self.frequency_plot: Optional[pg.PlotItem] = None
        
        # Data line items
        self.datalines: List[pg.PlotDataItem] = []
        self.frequency_datalines: List[pg.PlotDataItem] = []
        
        self._setup_plots()
    
    def _setup_plots(self) -> None:
        """Initialize the plot items and data lines."""
        # Create time-domain plot
        self.live_plot = self.plot_layout.addPlot(title="Live Data", row=1, col=0)
        self.live_plot.addLegend(offset=(0, 1))
        
        # Create frequency-domain plot
        self.frequency_plot = self.plot_layout.addPlot(title="FFT", row=2, col=0)
        self.frequency_plot.addLegend(offset=(0, 1))
        
        # Style the plots
        self._style_axes()
    
    def _style_axes(self) -> None:
        """Apply styling to plot axes."""
        tick_font = pg.QtGui.QFont('Arial', 14)
        label_style = {'font-size': '14pt', 'color': '#FFF'}
        
        # Style time-domain plot
        self.live_plot.getAxis('bottom').setStyle(tickFont=tick_font)
        self.live_plot.getAxis('left').setStyle(tickFont=tick_font)
        self.live_plot.getAxis('bottom').setLabel('Time', 's', **label_style)
        self.live_plot.getAxis('left').setLabel('Amplitude', 'mV', **label_style)
        
        # Style frequency-domain plot
        self.frequency_plot.getAxis('bottom').setStyle(tickFont=tick_font)
        self.frequency_plot.getAxis('left').setStyle(tickFont=tick_font)
        self.frequency_plot.getAxis('bottom').setLabel('Frequency', 'Hz', **label_style)
        self.frequency_plot.getAxis('left').setLabel('FFT', 'mV', **label_style)
    
    def initialize_datalines(self, buffer_size: int) -> None:
        """
        Create plot data items for all channels.
        
        Args:
            buffer_size: Size of data buffers
        """
        self.datalines.clear()
        self.frequency_datalines.clear()
        
        for i, line_cfg in enumerate(self.dataline_configs):
            # Initialize with dummy data
            x = np.arange(buffer_size)
            y = np.ones(buffer_size)
            
            # Create time-domain line
            line = self.live_plot.plot(
                x, y,
                pen=(i, len(self.dataline_configs)),
                width=1,
                name=line_cfg.name
            )
            line.setVisible(line_cfg.visible)
            line.setSkipFiniteCheck(True)
            
            # Configure hover tooltip
            line.scatter.opts.update(
                hoverable=True,
                tip='x: {x:.2f}, y: {y:.5f}'.format,
                hoverSymbol='+',
                hoverSize=10
            )
            
            self.datalines.append(line)
            
            # Create frequency-domain line
            freq_line = self.frequency_plot.plot(
                x, y,
                pen=(i, len(self.dataline_configs)),
                width=1,
                name=line_cfg.name
            )
            freq_line.setVisible(line_cfg.visible)
            
            freq_line.scatter.opts.update(
                hoverable=True,
                tip='x: {x:.5f}, y: {y:.5f}'.format,
                hoverSymbol='+',
                hoverSize=10
            )
            
            self.frequency_datalines.append(freq_line)
        
        self.update_legends()
    
    def update_line_data(self, channel: int, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """
        Update data for a specific channel.
        
        Args:
            channel: Channel index
            x_data: X-axis data
            y_data: Y-axis data
        """
        if 0 <= channel < len(self.datalines):
            self.datalines[channel].setData(x_data, y_data)
    
    def update_frequency_data(self, channel: int, freq: np.ndarray, magnitude: np.ndarray) -> None:
        """
        Update FFT data for a specific channel.
        
        Args:
            channel: Channel index
            freq: Frequency array
            magnitude: FFT magnitude array
        """
        if 0 <= channel < len(self.frequency_datalines):
            self.frequency_datalines[channel].setData(freq, magnitude)
    
    def set_line_visibility(self, channel: int, visible: bool) -> None:
        """
        Show or hide a data line.
        
        Args:
            channel: Channel index
            visible: True to show, False to hide
        """
        if 0 <= channel < len(self.datalines):
            self.datalines[channel].setVisible(visible)
            self.frequency_datalines[channel].setVisible(visible)
            self.update_legends()
    
    def set_show_points(self, show: bool) -> None:
        """
        Toggle display of data points.
        
        Args:
            show: True to show points, False to hide
        """
        symbol = 'o' if show else None
        for line in self.datalines:
            line.setSymbol(symbol)
        for line in self.frequency_datalines:
            line.setSymbol(symbol)
    
    def set_show_grid(self, show: bool) -> None:
        """
        Toggle grid display.
        
        Args:
            show: True to show grid, False to hide
        """
        self.live_plot.showGrid(y=show)
        self.frequency_plot.showGrid(x=show)
    
    def set_axis_scale(self, axis: str, scale: float, plot_type: str = 'both') -> None:
        """
        Set axis scale factor.
        
        Args:
            axis: 'left' or 'bottom'
            scale: Scale factor
            plot_type: 'time', 'frequency', or 'both'
        """
        if plot_type in ['time', 'both']:
            self.live_plot.getAxis(axis).setScale(scale)
        if plot_type in ['frequency', 'both']:
            self.frequency_plot.getAxis(axis).setScale(scale)
    
    def set_axis_label(self, axis: str, label: str, units: str, plot_type: str = 'time') -> None:
        """
        Set axis label and units.
        
        Args:
            axis: 'left' or 'bottom'
            label: Label text
            units: Unit string
            plot_type: 'time', 'frequency', or 'both'
        """
        if plot_type in ['time', 'both']:
            self.live_plot.getAxis(axis).setLabel(text=label, units=units)
        if plot_type in ['frequency', 'both']:
            self.frequency_plot.getAxis(axis).setLabel(text=label, units=units)
    
    def update_legends(self) -> None:
        """Update legends to show only visible lines."""
        # Clear legends
        self.live_plot.legend.clear()
        self.frequency_plot.legend.clear()
        
        # Add visible lines to legends
        for line in self.datalines:
            if line.isVisible():
                self.live_plot.legend.addItem(line, line.name())
        
        for line in self.frequency_datalines:
            if line.isVisible():
                self.frequency_plot.legend.addItem(line, line.name())
    
    def clear_plots(self) -> None:
        """Clear all plot data."""
        self.live_plot.clear()
        self.frequency_plot.clear()
    
    def get_visible_channels(self) -> List[int]:
        """
        Get list of visible channel indices.
        
        Returns:
            List of channel indices that are currently visible
        """
        return [i for i, line in enumerate(self.datalines) if line.isVisible()]
    
    def set_fft_visibility(self, show: bool) -> None:
        """
        Show or hide all FFT plots.
        
        Args:
            show: True to show, False to hide
        """
        for i, line in enumerate(self.datalines):
            if line.isVisible():
                self.frequency_datalines[i].setVisible(show)
            else:
                self.frequency_datalines[i].setVisible(False)
