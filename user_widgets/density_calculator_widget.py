"""Density calculator widget based on local DensityCalculators package."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PySide6.QtCore import QTimer, Signal as pyqtSignal, Slot as pyqtSlot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
    QGroupBox,
)

# Make local package importable when loaded as a standalone plugin module.
_WIDGET_DIR = Path(__file__).resolve().parent
_DENSITY_SRC = _WIDGET_DIR / "density_calculator" / "src"
if str(_DENSITY_SRC) not in sys.path:
    sys.path.insert(0, str(_DENSITY_SRC))

from DensityCalculators import (  # noqa: E402
    AIR,
    GasMixture,
    GasComponent,
    ConcentricOrifice,
    GasMeasurementSystem,
)


def _pure(species: str) -> GasMixture:
    return GasMixture([GasComponent(species, 1.0)])


def _clone_mix(mix: GasMixture) -> GasMixture:
    return GasMixture([GasComponent(c.name, c.mole_fraction) for c in mix.components])


class DensityWidget(QWidget):
    meanRequested = pyqtSignal(int)
    derivedSampleReady = pyqtSignal(int, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gas Measurement System")

        self._gas_factories = [
            ("AIR", lambda: _clone_mix(AIR)),
            ("N2", lambda: _pure("N2")),
            ("O2", lambda: _pure("O2")),
            ("CO2", lambda: _pure("CO2")),
            ("CH4", lambda: _pure("CH4")),
            ("Ar", lambda: _pure("Ar")),
            ("He", lambda: _pure("He")),
            ("H2", lambda: _pure("H2")),
        ]

        self._channel_count = 0
        self._last_dp_mean_mbar = float("nan")
        self._last_output_mean = float("nan")
        self._is_publish_mode = False
        self._system: GasMeasurementSystem | None = None

        self.gas_combo = QComboBox(self)
        for name, _ in self._gas_factories:
            self.gas_combo.addItem(name)

        self.orifice_combo = QComboBox(self)
        self.orifice_combo.addItems(["Concentric orifice", "Rectangular slit"])
        self.orifice_combo.currentTextChanged.connect(self._on_orifice_changed)

        self.geom_a_spin = QDoubleSpinBox(self)
        self.geom_a_spin.setRange(0.1, 500.0)
        self.geom_a_spin.setDecimals(3)
        self.geom_a_spin.setSuffix(" mm")

        self.geom_b_spin = QDoubleSpinBox(self)
        self.geom_b_spin.setRange(0.1, 500.0)
        self.geom_b_spin.setDecimals(3)
        self.geom_b_spin.setSuffix(" mm")

        self.temp_spin = QDoubleSpinBox(self)
        self.temp_spin.setRange(-80.0, 300.0)
        self.temp_spin.setDecimals(2)
        self.temp_spin.setValue(20.0)
        self.temp_spin.setSuffix(" °C")

        self.pressure_spin = QDoubleSpinBox(self)
        self.pressure_spin.setRange(100.0, 5000.0)
        self.pressure_spin.setDecimals(2)
        self.pressure_spin.setValue(1013.25)
        self.pressure_spin.setSuffix(" mbar(abs)")

        self.flow_spin = QDoubleSpinBox(self)
        self.flow_spin.setRange(0.0, 100000.0)
        self.flow_spin.setDecimals(3)
        self.flow_spin.setValue(0.0)
        self.flow_spin.setSuffix(" L/min")

        self.dp_channel_spin = QSpinBox(self)
        self.dp_channel_spin.setMinimum(0)
        self.dp_channel_spin.setMaximum(0)

        self.output_channel_spin = QSpinBox(self)
        self.output_channel_spin.setMinimum(0)
        self.output_channel_spin.setMaximum(0)

        self.calibrate_btn = QPushButton("Calibrate", self)
        self.calibrate_btn.clicked.connect(self._calibrate)

        self.publish_btn = QPushButton("Start Back-Propagation", self)
        self.publish_btn.clicked.connect(self._toggle_publish_mode)

        self.geom_a_label = QLabel("Orifice diameter d", self)
        self.geom_b_label = QLabel("Pipe diameter D", self)

        form = QFormLayout()
        form.addRow("Gas", self.gas_combo)
        form.addRow("Orifice type", self.orifice_combo)
        form.addRow(self.geom_a_label, self.geom_a_spin)
        form.addRow(self.geom_b_label, self.geom_b_spin)
        form.addRow("Temperature", self.temp_spin)
        form.addRow("Inlet Pressure", self.pressure_spin)
        form.addRow("Volumetric Flow", self.flow_spin)
        form.addRow("dP Source Channel", self.dp_channel_spin)
        form.addRow("Output Channel", self.output_channel_spin)

        input_group = QGroupBox("Inputs", self)
        input_group.setLayout(form)

        self.status_label = QLabel("Status: idle", self)
        self.dp_mean_label = QLabel("dP mean: -- mbar", self)
        self.density_label = QLabel("Density: -- kg/m^3", self)
        self.mass_flow_label = QLabel("Mass Flow: -- g/s", self)
        self.molar_mass_label = QLabel("Molar Mass: -- g/mol", self)
        self.output_mean_label = QLabel("Output mean: -- kg/m^3", self)
        self.k_factor_label = QLabel("K-Factor: --", self)

        out_layout = QVBoxLayout()
        out_layout.addWidget(self.status_label)
        out_layout.addWidget(self.dp_mean_label)
        out_layout.addWidget(self.density_label)
        out_layout.addWidget(self.mass_flow_label)
        out_layout.addWidget(self.molar_mass_label)
        out_layout.addWidget(self.output_mean_label)
        out_layout.addWidget(self.k_factor_label)

        output_group = QGroupBox("Results", self)
        output_group.setLayout(out_layout)

        layout = QVBoxLayout(self)
        layout.addWidget(input_group)
        layout.addWidget(self.calibrate_btn)
        layout.addWidget(self.publish_btn)
        layout.addWidget(output_group)

        self._on_orifice_changed(self.orifice_combo.currentText())

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(500)
        self._poll_timer.timeout.connect(self._poll_means_and_update_idle_result)
        self._poll_timer.start()

    @pyqtSlot(str)
    def _on_orifice_changed(self, text: str) -> None:
        if text == "Concentric orifice":
            self.geom_a_label.setText("Orifice diameter d")
            self.geom_b_label.setText("Pipe diameter D")
            self.geom_a_spin.setValue(20.0)
            self.geom_b_spin.setValue(50.0)
        else:
            self.geom_a_label.setText("Slit width w")
            self.geom_b_label.setText("Pipe diameter D")
            self.geom_a_spin.setValue(8.0)
            self.geom_b_spin.setValue(50.0)

    def setAvailableChannels(self, channel_names: list[str]) -> None:
        self._channel_count = len(channel_names)
        max_idx = max(0, self._channel_count - 1)
        self.dp_channel_spin.setMaximum(max_idx)
        self.output_channel_spin.setMaximum(max_idx)
        if self._channel_count > 0:
            self.dp_channel_spin.setValue(0)
            self.output_channel_spin.setValue(max_idx)

    @pyqtSlot(int, float)
    def setChannelMean(self, channel: int, value: float) -> None:
        if channel == self.dp_channel_spin.value():
            self._last_dp_mean_mbar = float(value)
            self.dp_mean_label.setText(f"dP mean: {value:.4f} mbar")
        if channel == self.output_channel_spin.value():
            self._last_output_mean = float(value)
            self.output_mean_label.setText(f"Output mean: {value:.6f} kg/m^3")

    @pyqtSlot()
    def _poll_means_and_update_idle_result(self) -> None:
        if self._channel_count <= 0:
            return

        self.meanRequested.emit(self.dp_channel_spin.value())
        self.meanRequested.emit(self.output_channel_spin.value())

        if self._is_publish_mode:
            return

        self._update_result_from_dp(self._last_dp_mean_mbar, source="mean")

    @pyqtSlot()
    def _calibrate(self) -> None:
        dp_mbar = self._last_dp_mean_mbar
        dp_pa = dp_mbar * 100.0
        flow = self._flow_m3_s()

        if not self._is_valid_positive(dp_pa) or flow <= 0.0:
            self.status_label.setText("Status: calibration failed (need dP mean > 0 and flow > 0)")
            return

        try:
            self._system = self._build_system()
            system = self._system
            if system is None:
                return
            k_factor = system.calibrate(delta_p=dp_pa, V_dot=flow)
            self.k_factor_label.setText(f"K-Factor: {k_factor:.8f}")
            self.status_label.setText("Status: calibrated")
        except Exception as exc:
            self.status_label.setText(f"Status: calibration error ({exc})")

    @pyqtSlot()
    def _toggle_publish_mode(self) -> None:
        self._is_publish_mode = not self._is_publish_mode
        if self._is_publish_mode:
            if self._system is None:
                self._system = self._build_system()
            self.publish_btn.setText("Stop Back-Propagation")
            self.status_label.setText("Status: publish mode active")
        else:
            self.publish_btn.setText("Start Back-Propagation")
            self.status_label.setText("Status: idle")

    def processSample(self, timestamp: float, values: list[float]) -> None:
        if not self._is_publish_mode:
            return
        if not values:
            return

        dp_idx = self.dp_channel_spin.value()
        if dp_idx < 0 or dp_idx >= len(values):
            return

        dp_mbar = float(values[dp_idx])
        self._update_result_from_dp(dp_mbar, source="sample", emit_output=True)

    def _update_result_from_dp(self, dp_mbar: float, source: str, emit_output: bool = False) -> None:
        flow = self._flow_m3_s()
        dp_pa = dp_mbar * 100.0
        if not self._is_valid_positive(dp_pa) or flow <= 0.0:
            return

        try:
            if self._system is None:
                self._system = self._build_system()
            system = self._system
            if system is None:
                return

            result = system.measure(delta_p=dp_pa, V_dot=flow)
            rho = float(result.density)
            m_dot_g_s = rho * flow * 1000.0

            self.density_label.setText(f"Density: {rho:.6f} kg/m^3")
            self.mass_flow_label.setText(f"Mass Flow: {m_dot_g_s:.6f} g/s")
            self.molar_mass_label.setText(f"Molar Mass: {result.molar_mass:.6f} g/mol")
            self.status_label.setText(f"Status: {source} update")

            if emit_output:
                self.derivedSampleReady.emit(self.output_channel_spin.value(), rho)
        except Exception as exc:
            self.status_label.setText(f"Status: calculation error ({exc})")

    def _build_system(self) -> GasMeasurementSystem:
        d_or_w_m = self.geom_a_spin.value() / 1000.0
        pipe_d_m = self.geom_b_spin.value() / 1000.0

        if self.orifice_combo.currentText() == "Concentric orifice":
            geometry = ConcentricOrifice(d=d_or_w_m, pipe_diameter=pipe_d_m)
        else:
            geometry = RectangularSlitOrifice(width=d_or_w_m, pipe_diameter=pipe_d_m)

        gas = self._selected_gas()
        temp_k = self.temp_spin.value() + 273.15
        p1_pa = self.pressure_spin.value() * 100.0
        return GasMeasurementSystem(geometry=geometry, gas=gas, T=temp_k, p1=p1_pa)

    def _selected_gas(self) -> GasMixture:
        idx = self.gas_combo.currentIndex()
        idx = max(0, min(idx, len(self._gas_factories) - 1))
        return self._gas_factories[idx][1]()

    def _flow_m3_s(self) -> float:
        return self.flow_spin.value() / 60000.0

    @staticmethod
    def _is_valid_positive(value: float) -> bool:
        return isinstance(value, float) and math.isfinite(value) and value > 0.0
