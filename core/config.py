"""
Application configuration dataclasses and constants.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import json


@dataclass
class PlotConfig:
    """Configuration for plot display settings."""
    show_fft: bool = True
    show_points: bool = False
    show_stats: bool = True
    show_grid: bool = True
    timestep: float = 1e-3
    fs: int = 1000
    time_window: float = 10.0
    y_scaling: float = 1.0
    y_unit: str = "mV"
    alpha_filter_value: float = 1.0

    @property
    def sample_rate(self) -> float:
        return 1.0 / self.timestep if self.timestep > 0 else 0


@dataclass
class ExportConfig:
    """Configuration for data export settings."""
    output_filename: str = "output.csv"
    stream_to_file: bool = False


@dataclass
class DatalineConfig:
    """Configuration for a single data line."""
    name: str
    index: int
    visible: bool = True


@dataclass
class ApplicationConfig:
    """Complete application configuration."""
    plot: PlotConfig = field(default_factory=PlotConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    datalines: List[DatalineConfig] = field(default_factory=list)

    @classmethod
    def from_json_files(cls, param_file: str = "parameter_config.json",
                        dataline_file: str = "dataline_config.json") -> "ApplicationConfig":
        try:
            with open(param_file) as f:
                param_data = json.load(f)
            plot_params = cls._extract_plot_params(param_data)
            export_params = cls._extract_export_params(param_data)
        except Exception as e:
            print(f"Error loading {param_file}: {e}")
            plot_params = PlotConfig()
            export_params = ExportConfig()

        try:
            with open(dataline_file) as f:
                dataline_data = json.load(f)
            datalines = cls._extract_datalines(dataline_data)
        except Exception as e:
            print(f"Error loading {dataline_file}: {e}")
            datalines = []

        return cls(plot=plot_params, export=export_params, datalines=datalines)

    @staticmethod
    def _extract_plot_params(config_data: Dict[str, Any]) -> PlotConfig:
        params = {}
        for group in config_data.get("parameters", []):
            if group["name"] == "Plot Parameters":
                for child in group["children"]:
                    key = child["name"].lower().replace("-", "_")
                    params[key] = child["value"]
        return PlotConfig(**params) if params else PlotConfig()

    @staticmethod
    def _extract_export_params(config_data: Dict[str, Any]) -> ExportConfig:
        params = {}
        for group in config_data.get("parameters", []):
            if group["name"] == "Export Settings":
                for child in group["children"]:
                    key = child["name"].lower().replace("-", "_")
                    params[key] = child["value"]
        return ExportConfig(**params) if params else ExportConfig()

    @staticmethod
    def _extract_datalines(config_data: Dict[str, Any]) -> List[DatalineConfig]:
        return [
            DatalineConfig(
                name=d.get("name", f"Line {d.get('index', 0)}"),
                index=d.get("index", 0),
                visible=d.get("visible", True),
            )
            for d in config_data.get("datalines", [])
        ]

    def to_parameter_tree_format(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Plot Parameters",
                "type": "group",
                "children": [
                    {"name": "show_fft",   "type": "bool",  "value": self.plot.show_fft},
                    {"name": "show_points","type": "bool",  "value": self.plot.show_points},
                    {"name": "show_stats", "type": "bool",  "value": self.plot.show_stats},
                    {"name": "show_grid",  "type": "bool",  "value": self.plot.show_grid},
                    {"name": "timestep",   "type": "float", "value": self.plot.timestep},
                    {"name": "fs",         "type": "int",   "value": self.plot.fs, "readonly": True},
                    {"name": "time_window","type": "float", "value": self.plot.time_window},
                    {"name": "y-Scaling",  "type": "float", "value": self.plot.y_scaling},
                    {"name": "y-Unit",     "type": "str",   "value": self.plot.y_unit},
                    {"name": "Alpha-Filter-Value", "type": "float", "value": self.plot.alpha_filter_value,
                     "limits": (0.0, 1.0), "step": 0.01},
                ],
            },
            {
                "name": "Export Settings",
                "type": "group",
                "children": [
                    {"name": "output_filename", "type": "str",  "value": self.export.output_filename},
                    {"name": "stream_to_file",  "type": "bool", "value": self.export.stream_to_file},
                ],
            },
        ]


class Constants:
    """Application-wide constants."""
    DEFAULT_BAUDRATE = 921600
    DEFAULT_MAX_PLOT_LENGTH = 1000
    DEFAULT_TIMER_INTERVAL_MS = 1
    STATS_UPDATE_INTERVAL = 1000
    COMMAND_TERMINATOR = "\n"

    MBAR_TO_MMHG = 0.750061683 / 1000
    MBAR_TO_BAR = 1 / 1000
