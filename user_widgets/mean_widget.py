"""Example user widget plugin that requests and displays channel mean values."""

from PySide6.QtCore import Signal as pyqtSignal, Slot as pyqtSlot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QPushButton,
)


class MeanWidget(QWidget):
    """Simple plugin that requests mean value for a selected channel."""

    meanRequested = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mean Monitor")

        self.channel_spin = QSpinBox(self)
        self.channel_spin.setMinimum(0)
        self.channel_spin.setMaximum(0)

        self.channel_label = QLabel("Channel", self)
        self.value_label = QLabel("Mean: --", self)

        request_btn = QPushButton("Request Mean", self)
        request_btn.clicked.connect(self._request_mean)

        row = QHBoxLayout()
        row.addWidget(self.channel_label)
        row.addWidget(self.channel_spin)

        layout = QVBoxLayout(self)
        layout.addLayout(row)
        layout.addWidget(request_btn)
        layout.addWidget(self.value_label)

    def setAvailableChannels(self, channel_names: list[str]) -> None:
        max_index = max(0, len(channel_names) - 1)
        self.channel_spin.setMaximum(max_index)

    @pyqtSlot()
    def _request_mean(self) -> None:
        self.meanRequested.emit(self.channel_spin.value())

    @pyqtSlot(float)
    def setMeanValue(self, value: float) -> None:
        self.value_label.setText(f"Mean: {value:.6f}")
