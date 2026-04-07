"""Binary gas composition widget based on local DensityCalculators package.

This widget is designed as a runtime plugin for SerialPlotter. It supports:

1. Generic binary gas estimation for two pure components.
2. Dedicated AIR/N2 mode where composition is reported as AIR% and N2%.
3. Calibration with a known reference composition at current process conditions.

All outputs are shown in widget labels; no derived channel publication is required.
"""

from __future__ import annotations

from dataclasses import asdict
import csv
import math
import sys
import time
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
    QCheckBox,
)

# Make local package importable when loaded as a standalone plugin module.
_WIDGET_DIR = Path(__file__).resolve().parent
_DENSITY_SRC = _WIDGET_DIR / "density_calculator" / "src"
if str(_DENSITY_SRC) not in sys.path:
    sys.path.insert(0, str(_DENSITY_SRC))

from DensityCalculators import (
    AIR,
    GasMixture,
    GasComponent,
    ConcentricOrifice,
    RectangularSlitOrifice,
    GasMeasurementSystem,
)


class DensityWidget(QWidget):
    """Binary gas analysis widget using ``GasMeasurementSystem``.

    Host integration points used by SerialPlotter:
    - ``setAvailableChannels(channel_names)``
    - ``setChannelMean(channel, value)``
    - ``processSample(timestamp, values)``
    - ``meanRequested`` signal

    The widget continuously converts measured dP and configured flow into density,
    then infers composition based on the selected gas model.
    """

    meanRequested = pyqtSignal(int)
    derivedSampleReady = pyqtSignal(int, float)
    written_once = False

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Binary Gas Measurement System")

        self._gas_species = ["AIR", "N2", "O2", "CO2", "CH4", "Ar", "He", "H2"]
        self._channel_count = 0
        self._last_dp_mean_mbar = float("nan")
        self._is_live_mode = False
        self._system: GasMeasurementSystem | None = None
        self._last_live_update_time = 0.0
        self._dp_sample_buffer: list[float] = []

        self.light_gas_combo = QComboBox(self)
        self.light_gas_combo.addItems(self._gas_species)
        self.light_gas_combo.setCurrentText("AIR")

        self.heavy_gas_combo = QComboBox(self)
        self.heavy_gas_combo.addItems(self._gas_species)
        self.heavy_gas_combo.setCurrentText("N2")

        self.ref_heavy_fraction_spin = QDoubleSpinBox(self)
        self.ref_heavy_fraction_spin.setRange(0.0, 100.0)
        self.ref_heavy_fraction_spin.setDecimals(2)
        self.ref_heavy_fraction_spin.setValue(50.0)
        self.ref_heavy_fraction_spin.setSuffix(" %")

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
        self.temp_spin.setSuffix(" degC")

        self.pressure_spin = QDoubleSpinBox(self)
        self.pressure_spin.setRange(100.0, 5000.0)
        self.pressure_spin.setDecimals(2)
        self.pressure_spin.setValue(1013.25)
        self.pressure_spin.setSuffix(" mbar(abs)")

        self.flow_spin = QDoubleSpinBox(self)
        self.flow_spin.setRange(0.0, 100000.0)
        self.flow_spin.setDecimals(3)
        self.flow_spin.setValue(0.0)
        self.flow_spin.setSuffix(" NL/min")

        self.dp_channel_spin = QSpinBox(self)
        self.dp_channel_spin.setMinimum(0)
        self.dp_channel_spin.setMaximum(0)

        self.apply_state_checkbox = QCheckBox("Apply measured composition state", self)
        self.apply_state_checkbox.setChecked(False)

        self.calibrate_btn = QPushButton("Calibrate", self)
        self.calibrate_btn.clicked.connect(self._calibrate)

        self.live_btn = QPushButton("Start Live Analysis", self)
        self.live_btn.clicked.connect(self._toggle_live_mode)

        self.geom_a_label = QLabel("Orifice diameter d", self)
        self.geom_b_label = QLabel("Pipe diameter D", self)

        form = QFormLayout()
        form.addRow("Light Gas", self.light_gas_combo)
        form.addRow("Heavy Gas", self.heavy_gas_combo)
        form.addRow("Reference Heavy Fraction", self.ref_heavy_fraction_spin)
        form.addRow("Orifice type", self.orifice_combo)
        form.addRow(self.geom_a_label, self.geom_a_spin)
        form.addRow(self.geom_b_label, self.geom_b_spin)
        form.addRow("Temperature", self.temp_spin)
        form.addRow("Inlet Pressure", self.pressure_spin)
        form.addRow("Volumetric Flow (Normal)", self.flow_spin)
        form.addRow("dP Source Channel", self.dp_channel_spin)
        form.addRow(self.apply_state_checkbox)

        input_group = QGroupBox("Inputs", self)
        input_group.setLayout(form)

        self.status_label = QLabel("Status: idle", self)
        self.dp_mean_label = QLabel("dP mean: -- mbar", self)
        self.density_label = QLabel("Density: -- kg/m^3", self)
        self.std_density_label = QLabel("Standard Density: -- kg/m^3", self)
        self.mass_flow_label = QLabel("Mass Flow: -- g/s", self)
        self.molar_mass_label = QLabel("Molar Mass: -- g/mol", self)
        self.light_fraction_label = QLabel("Light Fraction: -- %", self)
        self.heavy_fraction_label = QLabel("Heavy Fraction: -- %", self)
        self.iterations_label = QLabel("Composition Iterations: --", self)
        self.k_factor_label = QLabel("K-Factor: --", self)

        out_layout = QVBoxLayout()
        out_layout.addWidget(self.status_label)
        out_layout.addWidget(self.dp_mean_label)
        out_layout.addWidget(self.density_label)
        out_layout.addWidget(self.std_density_label)
        out_layout.addWidget(self.mass_flow_label)
        out_layout.addWidget(self.molar_mass_label)
        out_layout.addWidget(self.light_fraction_label)
        out_layout.addWidget(self.heavy_fraction_label)
        out_layout.addWidget(self.iterations_label)
        out_layout.addWidget(self.k_factor_label)

        output_group = QGroupBox("Results", self)
        output_group.setLayout(out_layout)

        layout = QVBoxLayout(self)
        layout.addWidget(input_group)
        layout.addWidget(self.calibrate_btn)
        layout.addWidget(self.live_btn)
        layout.addWidget(output_group)

        self._on_orifice_changed(self.orifice_combo.currentText())

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(500)
        self._poll_timer.timeout.connect(self._poll_means_and_update_idle_result)
        self._poll_timer.start()

    @pyqtSlot(str)
    def _on_orifice_changed(self, text: str) -> None:
        """Update geometry labels/defaults when the orifice type changes."""
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
        """Receive available channel names from host and clamp channel selectors."""
        self._channel_count = len(channel_names)
        max_idx = max(0, self._channel_count - 1)
        self.dp_channel_spin.setMaximum(max_idx)
        if self._channel_count > 0:
            self.dp_channel_spin.setValue(0)

    @pyqtSlot(int, float)
    def setChannelMean(self, channel: int, value: float) -> None:
        """Receive channel means requested via ``meanRequested`` and cache dP mean."""
        if channel == self.dp_channel_spin.value():
            self._last_dp_mean_mbar = float(value)
            self.dp_mean_label.setText(f"dP mean: {value:.4f} mbar")

    @pyqtSlot()
    def _poll_means_and_update_idle_result(self) -> None:
        """Poll host for dP mean and refresh labels while live mode is inactive."""
        if self._channel_count <= 0:
            return

        self.meanRequested.emit(self.dp_channel_spin.value())

        if self._is_live_mode:
            return

        self._update_result_from_dp(self._last_dp_mean_mbar, source="mean")

    @pyqtSlot()
    def _calibrate(self) -> None:
        """Calibrate system K-factor using current conditions and reference mixture.

        Reference density is computed from the currently configured gas composition
        (including AIR/N2 reference blend in AIR/N2 mode), temperature, and pressure.
        """
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

            rho_reference = system.gas.get_properties(system.T, system.p1).density
            k_factor = system.calibrate(delta_p=dp_pa, V_dot=flow, rho_reference=rho_reference)
            result = system.measure(
                delta_p=dp_pa,
                V_dot=flow,
                apply_state=(self.apply_state_checkbox.isChecked() and not self._is_air_n2_mode()),
            )

            rho = float(result.density)
            std_rho = system.calc_standard_density(
                rho_current=rho,
                T_current=self.temp_spin.value() + 273.15,
                p_current=self.pressure_spin.value() * 100.0,
            )

            self.k_factor_label.setText(f"K-Factor: {k_factor:.8f}")
            self.status_label.setText("Status: calibrated")
            self._update_result_labels(result, rho, std_rho, flow)
            self.save_info_to_file(system, result, dp_pa, flow, std_rho)
        except Exception as exc:
            self.status_label.setText(f"Status: calibration error ({exc})")

    def save_info_to_file(self, system, result, delta_p, flow, std_rho) -> None:
        """Append calibration/measurement snapshot to ``result_info.csv``."""
        output_dict = {
            "Flow (m3/s)": flow,
            "Delta P (Pa)": delta_p,
            "K-Factor": system.K_factor,
            "Inlet Pressure (Pa)": system.p1,
            "Temperature (K)": system.T,
            "Standard density (kg/m^3)": std_rho,
            **asdict(result),
        }
        with open("result_info.csv", mode="a+", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=output_dict.keys())
            if not self.written_once:
                writer.writeheader()
                self.written_once = True
            writer.writerow(output_dict)

    @pyqtSlot()
    def _toggle_live_mode(self) -> None:
        """Enable/disable live analysis mode and keep UI status in sync."""
        self._is_live_mode = not self._is_live_mode
        if self._is_live_mode:
            if self._system is None:
                try:
                    self._system = self._build_system()
                except Exception as exc:
                    self._is_live_mode = False
                    self.status_label.setText(f"Status: setup error ({exc})")
                    return
            self.live_btn.setText("Stop Live Analysis")
            self.status_label.setText("Status: live analysis active")
        else:
            self.live_btn.setText("Start Live Analysis")
            self.status_label.setText("Status: idle")

    def processSample(self, timestamp: float, values: list[float]) -> None:
        """Consume raw samples from host and update result at a throttled rate.

        The widget averages dP samples over approximately one-second windows to
        reduce jitter in displayed composition.
        """
        if not self._is_live_mode:
            return
        if not values:
            return

        dp_idx = self.dp_channel_spin.value()
        if dp_idx < 0 or dp_idx >= len(values):
            return

        dp_mbar = float(values[dp_idx])
        self._dp_sample_buffer.append(dp_mbar)

        current_time = time.time()
        if current_time - self._last_live_update_time < 1.0:
            return
        self._last_live_update_time = current_time

        if self._dp_sample_buffer:
            avg_dp_mbar = sum(self._dp_sample_buffer) / len(self._dp_sample_buffer)
            self._dp_sample_buffer = []
            self._update_result_from_dp(avg_dp_mbar, source="sample")

    def _update_result_from_dp(self, dp_mbar: float, source: str) -> None:
        """Run one measurement update from a dP value in mbar."""
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

            result = system.measure(
                delta_p=dp_pa,
                V_dot=flow,
                apply_state=(self.apply_state_checkbox.isChecked() and not self._is_air_n2_mode()),
            )
            rho = float(result.density)
            std_rho = system.calc_standard_density(
                rho_current=rho,
                T_current=self.temp_spin.value() + 273.15,
                p_current=self.pressure_spin.value() * 100.0,
            )

            self._update_result_labels(result, rho, std_rho, flow)
            self.status_label.setText(f"Status: {source} update")
        except Exception as exc:
            self.status_label.setText(f"Status: calculation error ({exc})")

    def _update_result_labels(self, result, rho: float, std_rho: float, flow_m3_s: float) -> None:
        """Refresh all result labels for density and composition.

        AIR/N2 mode uses a dedicated inversion from measured density to report
        AIR% and N2%. Other gas pairs use ``result.composition`` directly.
        """
        m_dot_g_s = rho * flow_m3_s * 1000.0
        self.density_label.setText(f"Density: {rho:.6f} kg/m^3")
        self.std_density_label.setText(f"Standard Density: {std_rho:.6f} kg/m^3")
        self.mass_flow_label.setText(f"Mass Flow: {m_dot_g_s:.6f} g/s")
        self.molar_mass_label.setText(f"Molar Mass: {result.molar_mass:.6f} g/mol")

        if self._is_air_n2_mode():
            x_air, x_n2 = self._estimate_air_n2_blend_from_density(
                rho,
                self.temp_spin.value() + 273.15,
                self.pressure_spin.value() * 100.0,
            )
            self.light_fraction_label.setText(f"AIR Fraction: {100.0 * x_air:.3f} %")
            self.heavy_fraction_label.setText(f"N2 Fraction: {100.0 * x_n2:.3f} %")
            self.iterations_label.setText("Composition Iterations: AIR/N2 inversion")
            return

        if not result.composition or self._system is None or not self._system.gas.is_binary:
            self.light_fraction_label.setText("Light Fraction: -- %")
            self.heavy_fraction_label.setText("Heavy Fraction: -- %")
            self.iterations_label.setText("Composition Iterations: 0")
            return

        light_name = self._system.gas.light.name
        heavy_name = self._system.gas.heavy.name
        x_light = float(result.composition.get(light_name, 0.0))
        x_heavy = float(result.composition.get(heavy_name, 0.0))

        self.light_fraction_label.setText(f"Light Fraction ({light_name}): {100.0 * x_light:.3f} %")
        self.heavy_fraction_label.setText(f"Heavy Fraction ({heavy_name}): {100.0 * x_heavy:.3f} %")
        self.iterations_label.setText(f"Composition Iterations: {result.composition_iterations}")

    def _build_system(self) -> GasMeasurementSystem:
        """Create a configured ``GasMeasurementSystem`` from current UI values."""
        d_or_w_m = self.geom_a_spin.value() / 1000.0
        pipe_d_m = self.geom_b_spin.value() / 1000.0

        if self.orifice_combo.currentText() == "Concentric orifice":
            geometry = ConcentricOrifice(d=d_or_w_m, pipe_diameter=pipe_d_m)
        else:
            geometry = RectangularSlitOrifice(width=d_or_w_m, pipe_diameter=pipe_d_m)

        gas = self._selected_measurement_gas()
        temp_k = self.temp_spin.value() + 273.15
        p1_pa = self.pressure_spin.value() * 100.0
        return GasMeasurementSystem(geometry=geometry, gas=gas, T=temp_k, p1=p1_pa)

    def _selected_measurement_gas(self) -> GasMixture:
        """Return gas model used by the solver for current mode.

        In AIR/N2 mode, the returned gas is a synthetic blend built from the
        AIR reference mixture and pure N2 using the UI reference fraction.
        """
        if self._is_air_n2_mode():
            x_air = self._reference_air_fraction_from_ui()
            return self._air_n2_blend_mixture(x_air)
        return self._selected_binary_gas()

    def _is_air_n2_mode(self) -> bool:
        """True when selected pair is exactly AIR and N2 (any order)."""
        light = self.light_gas_combo.currentText().strip()
        heavy = self.heavy_gas_combo.currentText().strip()
        return light != heavy and {light, heavy} == {"AIR", "N2"}

    def _reference_air_fraction_from_ui(self) -> float:
        """Map UI heavy-fraction input to AIR fraction for AIR/N2 mode.

        If heavy gas is N2, AIR fraction is ``1 - x_heavy``.
        If heavy gas is AIR, AIR fraction is ``x_heavy``.
        """
        light = self.light_gas_combo.currentText().strip()
        heavy = self.heavy_gas_combo.currentText().strip()
        x_heavy = self.ref_heavy_fraction_spin.value() / 100.0

        if light == "AIR" and heavy == "N2":
            return 1.0 - x_heavy
        if light == "N2" and heavy == "AIR":
            return x_heavy
        raise ValueError("AIR/N2 mode requires Light/Heavy gases to be AIR and N2")

    @staticmethod
    def _air_n2_blend_mixture(x_air: float) -> GasMixture:
        """Construct AIR/N2 blend as explicit components compatible with GasMixture."""
        x_air = max(0.0, min(1.0, x_air))
        fractions = {c.name: x_air * c.mole_fraction for c in AIR.components}
        fractions["N2"] = fractions.get("N2", 0.0) + (1.0 - x_air)
        components = [GasComponent(name, frac) for name, frac in fractions.items() if frac > 0.0]
        return GasMixture(components)

    def _estimate_air_n2_blend_from_density(
        self,
        rho_target: float,
        temp_k: float,
        pressure_pa_abs: float,
    ) -> tuple[float, float]:
        """Estimate AIR and N2 mole fractions from measured density.

        Uses a monotonic bisection on AIR fraction over [0, 1] while evaluating
        density of the synthetic AIR/N2 blend at current temperature and pressure.
        """
        if rho_target <= 0.0:
            return 0.0, 1.0

        def rho_from_air_fraction(x_air: float) -> float:
            mix = self._air_n2_blend_mixture(x_air)
            return float(mix.get_properties(temp_k, pressure_pa_abs).density)

        lo = 0.0
        hi = 1.0
        rho_lo = rho_from_air_fraction(lo)
        rho_hi = rho_from_air_fraction(hi)

        rho_min = min(rho_lo, rho_hi)
        rho_max = max(rho_lo, rho_hi)
        if rho_target <= rho_min:
            x_air = lo if rho_lo <= rho_hi else hi
            return x_air, 1.0 - x_air
        if rho_target >= rho_max:
            x_air = lo if rho_lo >= rho_hi else hi
            return x_air, 1.0 - x_air

        increasing = rho_hi > rho_lo
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            rho_mid = rho_from_air_fraction(mid)
            if increasing:
                if rho_mid < rho_target:
                    lo = mid
                else:
                    hi = mid
            else:
                if rho_mid > rho_target:
                    lo = mid
                else:
                    hi = mid

        x_air = 0.5 * (lo + hi)
        return x_air, 1.0 - x_air

    def _selected_binary_gas(self) -> GasMixture:
        """Build a generic binary pure-component mixture from UI controls."""
        light = self.light_gas_combo.currentText().strip()
        heavy = self.heavy_gas_combo.currentText().strip()
        if not light or not heavy or light == heavy:
            raise ValueError("Select two different gas components for binary analysis")
        if "AIR" in {light, heavy}:
            raise ValueError("AIR is currently only supported in dedicated AIR/N2 mode")

        x_heavy = self.ref_heavy_fraction_spin.value() / 100.0
        return GasMixture([
            GasComponent(light, 1.0 - x_heavy),
            GasComponent(heavy, x_heavy),
        ])

    def _flow_m3_s(self) -> float:
        """Return operating volumetric flow in m^3/s from normal L/min input."""
        flow_normal_l_min = self.flow_spin.value()
        flow_operating_l_min = self._normal_l_min_to_operating_l_min(
            flow_normal_l_min,
            self.temp_spin.value() + 273.15,
            self.pressure_spin.value() * 100.0,
        )
        return flow_operating_l_min / 60000.0

    @staticmethod
    def _normal_l_min_to_operating_l_min(
        flow_normal_l_min: float,
        temp_k: float,
        pressure_pa_abs: float,
    ) -> float:
        """Convert normal flow (0 C, 101325 Pa) to operating flow."""
        if flow_normal_l_min < 0.0:
            raise ValueError("flow_normal_l_min must be >= 0")
        if temp_k <= 0.0 or pressure_pa_abs <= 0.0:
            raise ValueError("temp_k and pressure_pa_abs must be > 0")

        t_normal_k = 273.15
        p_normal_pa = 101325.0
        return flow_normal_l_min * (p_normal_pa / pressure_pa_abs) * (temp_k / t_normal_k)

    @staticmethod
    def _is_valid_positive(value: float) -> bool:
        """Validate finite positive float values for process calculations."""
        return isinstance(value, float) and math.isfinite(value) and value > 0.0
