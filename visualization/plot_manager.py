"""
Plot manager: creates and updates pyqtgraph plots for multi-channel data.
"""

import pyqtgraph as pg
import numpy as np
from typing import List, Optional

from core.config import DatalineConfig


class PlotManager:
    """Manages pyqtgraph time-domain and frequency-domain plots."""

    def __init__(self, plot_layout: pg.GraphicsLayoutWidget,
                 dataline_configs: List[DatalineConfig]) -> None:
        self.plot_layout = plot_layout
        self.dataline_configs = dataline_configs

        self.live_plot: Optional[pg.PlotItem] = None
        self.frequency_plot: Optional[pg.PlotItem] = None
        self.datalines: List[pg.PlotDataItem] = []
        self.frequency_datalines: List[pg.PlotDataItem] = []

        self._setup_plots()

    # ─── Initialisation ───────────────────────────────────────────────────

    def _setup_plots(self) -> None:
        self.live_plot = self.plot_layout.addPlot(title="Live Data", row=1, col=0)
        self.live_plot.addLegend(offset=(0, 1))
        self.frequency_plot = self.plot_layout.addPlot(title="FFT", row=2, col=0)
        self.frequency_plot.addLegend(offset=(0, 1))
        self._style_axes()

    def _style_axes(self) -> None:
        tick_font = pg.QtGui.QFont("DejaVu Sans", 14)
        label_style = {"font-size": "14pt", "color": "#FFF"}
        for plot, x_label, x_unit in [
            (self.live_plot, "Time", "s"),
            (self.frequency_plot, "Frequency", "Hz"),
        ]:
            for axis in ("bottom", "left"):
                plot.getAxis(axis).setStyle(tickFont=tick_font)
            plot.getAxis("bottom").setLabel(x_label, x_unit, **label_style)
            plot.getAxis("left").setLabel("Amplitude" if plot is self.live_plot else "FFT",
                                          "mV", **label_style)

    def initialize_datalines(self, buffer_size: int) -> None:
        """Create or re-create plot data items for all channels."""
        self.datalines.clear()
        self.frequency_datalines.clear()

        n = len(self.dataline_configs)
        x = np.array([])
        y = np.array([])
        hover_opts = dict(hoverable=True, hoverSymbol="+", hoverSize=10)

        for i, cfg in enumerate(self.dataline_configs):
            line = self.live_plot.plot(x, y, pen=(i, n), width=1, name=cfg.name)
            line.setVisible(cfg.visible)
            line.setSkipFiniteCheck(True)
            line.scatter.opts.update(**hover_opts,
                                     tip="x: {x:.2f}, y: {y:.5f}".format)
            self.datalines.append(line)

            freq_line = self.frequency_plot.plot(x, y, pen=(i, n), width=1, name=cfg.name)
            freq_line.setVisible(cfg.visible)
            freq_line.scatter.opts.update(**hover_opts,
                                          tip="x: {x:.5f}, y: {y:.5f}".format)
            self.frequency_datalines.append(freq_line)

        self.update_legends()

    # ─── Data updates ─────────────────────────────────────────────────────

    def update_line_data(self, channel: int, x_data: np.ndarray, y_data: np.ndarray) -> None:
        if 0 <= channel < len(self.datalines):
            self.datalines[channel].setData(x_data, y_data)

    def update_frequency_data(self, channel: int, freq: np.ndarray, magnitude: np.ndarray) -> None:
        if 0 <= channel < len(self.frequency_datalines):
            self.frequency_datalines[channel].setData(freq, magnitude)

    # ─── Appearance controls ──────────────────────────────────────────────

    def set_line_visibility(self, channel: int, visible: bool) -> None:
        if 0 <= channel < len(self.datalines):
            self.datalines[channel].setVisible(visible)
            self.frequency_datalines[channel].setVisible(visible)
            self.update_legends()

    def set_show_points(self, show: bool) -> None:
        symbol = "o" if show else None
        for line in self.datalines + self.frequency_datalines:
            line.setSymbol(symbol)

    def set_show_grid(self, show: bool) -> None:
        self.live_plot.showGrid(y=show)
        self.frequency_plot.showGrid(x=show)

    def set_axis_scale(self, axis: str, scale: float, plot_type: str = "both") -> None:
        if plot_type in ("time", "both"):
            self.live_plot.getAxis(axis).setScale(scale)
        if plot_type in ("frequency", "both"):
            self.frequency_plot.getAxis(axis).setScale(scale)

    def set_axis_label(self, axis: str, label: str, units: str, plot_type: str = "time") -> None:
        if plot_type in ("time", "both"):
            self.live_plot.getAxis(axis).setLabel(text=label, units=units)
        if plot_type in ("frequency", "both"):
            self.frequency_plot.getAxis(axis).setLabel(text=label, units=units)

    def set_fft_visibility(self, show: bool) -> None:
        for i, line in enumerate(self.datalines):
            self.frequency_datalines[i].setVisible(show and line.isVisible())

    def update_legends(self) -> None:
        self.live_plot.legend.clear()
        self.frequency_plot.legend.clear()
        for line in self.datalines:
            if line.isVisible():
                self.live_plot.legend.addItem(line, line.name())
        for line in self.frequency_datalines:
            if line.isVisible():
                self.frequency_plot.legend.addItem(line, line.name())

    def clear_plots(self) -> None:
        self.live_plot.clear()
        self.frequency_plot.clear()

    def clear_plot_data(self) -> None:
        empty = np.array([])
        for line in self.datalines:
            line.setData(empty, empty)
        for line in self.frequency_datalines:
            line.setData(empty, empty)

    def get_visible_channels(self) -> List[int]:
        return [i for i, line in enumerate(self.datalines) if line.isVisible()]
