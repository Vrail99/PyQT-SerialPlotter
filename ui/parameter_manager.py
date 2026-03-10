"""
Parameter tree manager: owns the ParameterTree widget and dispatches all changes.
"""

from pyqtgraph.parametertree import Parameter, ParameterTree
from PySide6.QtCore import QObject


class ParameterManager(QObject):
    """
    Manages the pyqtgraph ParameterTree.

    Internalises all parameter-change dispatch so SerialPlotter does not
    need to know about individual parameter names.
    """

    def __init__(self, config, plot_manager, data_buffer,
                 fft_calculator, data_processor, file_stream_manager) -> None:
        super().__init__()
        self.config = config
        self.plot_manager = plot_manager
        self.data_buffer = data_buffer
        self.fft_calculator = fft_calculator
        self.data_processor = data_processor
        self.file_stream_manager = file_stream_manager
        self.params: Parameter = None

    def setup(self, param_tree: ParameterTree) -> Parameter:
        """Build the parameter tree, attach it to the widget, and return the root."""
        stats_params = [
            {
                "name": cfg.name,
                "type": "group",
                "children": [
                    {"name": "Min",   "type": "float", "value": float("inf"),  "readonly": True},
                    {"name": "Max",   "type": "float", "value": float("-inf"), "readonly": True},
                    {"name": "Mean",  "type": "float", "value": 0.0, "readonly": True, "decimals": 4},
                    {"name": "Std",   "type": "float", "value": 0.0, "readonly": True, "decimals": 4},
                    {"name": "Slope", "type": "float", "value": 0.0, "readonly": True, "decimals": 1},
                ],
            }
            for cfg in self.config.datalines
        ]

        all_params = self.config.to_parameter_tree_format() + [
            {"name": "Statistics", "type": "group", "children": stats_params}
        ]

        self.params = Parameter.create(name="Parameters", type="group", children=all_params)
        self.params.sigTreeStateChanged.connect(self._on_changed)
        param_tree.setParameters(self.params, showTop=False)
        return self.params

    # ─── Change dispatch ──────────────────────────────────────────────────

    def _on_changed(self, _param, changes) -> None:
        for param_, _, data in changes:
            path = self.params.childPath(param_)
            if path is None or path[0] == "Statistics":
                continue
            self._dispatch(".".join(path), data)

    def _dispatch(self, name: str, data) -> None:
        pm = self.plot_manager
        cfg = self.config.plot

        if name == "Plot Parameters.show_fft":
            cfg.show_fft = data
            if data:
                for ch in pm.get_visible_channels():
                    _, y = self.data_buffer.get_channel_data(ch)
                    mag, freq = self.fft_calculator.calculate_fft(y, cfg.timestep)
                    pm.update_frequency_data(ch, freq, mag)
            pm.set_fft_visibility(data)

        elif name == "Plot Parameters.show_points":
            pm.set_show_points(data)

        elif name == "Plot Parameters.show_grid":
            pm.set_show_grid(data)

        elif name == "Plot Parameters.timestep":
            cfg.timestep = data
            cfg.fs = int(1 / data) if data > 0 else 0
            self.data_processor.set_timestep(data)
            pm.set_axis_scale("bottom", data, "time")
            self.params.child("Plot Parameters").child("fs").setValue(cfg.fs)

        elif name == "Plot Parameters.y-Scaling":
            pm.set_axis_scale("left", data, "both")
            cfg.y_scaling = data

        elif name == "Plot Parameters.y-Unit":
            pm.set_axis_label("left", "Amplitude", data, "both")
            cfg.y_unit = data

        elif name == "Plot Parameters.Alpha-Filter-Value":
            self.data_processor.set_smoothing(data)
            cfg.alpha_filter_value = data

        elif name == "Export Settings.stream_to_file":
            if data:
                names = [l.name for l in self.config.datalines]
                self.file_stream_manager.start_streaming(self.config.export.output_filename, names)
            else:
                self.file_stream_manager.stop_streaming()

    # ─── Live update API ──────────────────────────────────────────────────

    def update_statistics(self, stats_list) -> None:
        stats_group = self.params.child("Statistics")
        for i, stats in enumerate(stats_list):
            if i >= len(stats_group.children()):
                break
            ch = stats_group.children()[i]
            ch.child("Min").setValue(stats.min_value)
            ch.child("Max").setValue(stats.max_value)
            ch.child("Mean").setValue(stats.mean)
            ch.child("Std").setValue(stats.std)
            ch.child("Slope").setValue(stats.slope)

    def update_fs(self, rate: float) -> None:
        if self.params:
            self.params.child("Plot Parameters").child("fs").setValue(int(rate))
