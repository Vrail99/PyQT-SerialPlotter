# User Widget API

This document describes how to write plugins for the `Load Widget` feature.

## Discovery And Loading

`SerialPlotter` uses `WidgetModuleLoader` to scan this folder:

- `user_widgets/*.py`

Rules:

- Files named `__init__.py` are ignored.
- Files starting with `_` are ignored.
- The selected class must be a subclass of `QWidget`.
- The class must be defined in the selected module file.

## Minimum Requirements

A plugin is usable if it provides:

1. A Python module in `user_widgets/`.
2. A class deriving from `QWidget`.
3. A constructor compatible with:
   - `__init__(self, parent=None)`

That is enough to be loadable and shown.

## Optional Integration Hooks

`SerialPlotter` checks these members dynamically. Implement any of them if needed.

### Optional method: `setAvailableChannels(channel_names: list[str])`

- Called once after widget creation.
- `channel_names` is built from configured dataline names.
- Use this to update channel dropdowns/spinboxes.

### Optional signal: `meanRequested(int)`

- Emit with channel index when your widget wants the current mean value.
- Connected automatically to `SerialPlotter`.

### Optional method: `processSample(timestamp: float, values: list[float])`

- Called for each acquired sample after the raw hardware values have been written into
    the application's normal data buffer.
- Use this to derive calculated values from the live sample stream.

### Optional signal: `derivedSampleReady(int, float)`

- Emit with `(channel_index, value)` when your widget produces a derived scalar.
- `channel_index` must point to a reserved derived channel declared in
    `dataline_config.json`.
- Connected automatically to `SerialPlotter`.

### Optional callback methods for mean response

When a mean request is handled, `SerialPlotter` calls one of these on the requester:

- `setMeanValue(value: float)`
- `setChannelMean(channel: int, value: float)`

If neither is available on the requester, `SerialPlotter` falls back to broadcasting
`setChannelMean(channel, value)` to loaded widgets that implement it.

The existing mean request flow remains unchanged when using the optional sample
processing hooks above.

## Minimal Template

```python
from PySide6.QtWidgets import QWidget


class MyWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
```

## Mean-Request Template

```python
from PySide6.QtCore import Signal as pyqtSignal, Slot as pyqtSlot
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel


class MyMeanWidget(QWidget):
    meanRequested = pyqtSignal(int)
    derivedSampleReady = pyqtSignal(int, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.value = QLabel("Mean: --", self)
        btn = QPushButton("Request Ch0 Mean", self)
        btn.clicked.connect(lambda: self.meanRequested.emit(0))

        layout = QVBoxLayout(self)
        layout.addWidget(self.value)
        layout.addWidget(btn)

    def setAvailableChannels(self, channel_names: list[str]) -> None:
        # Optional: configure your channel selectors here
        pass

    def processSample(self, timestamp: float, values: list[float]) -> None:
        if values:
            self.derivedSampleReady.emit(len(values), values[0])

    @pyqtSlot(float)
    def setMeanValue(self, value: float) -> None:
        self.value.setText(f"Mean: {value:.6f}")
```

## Quick Test

1. Add your plugin file under `user_widgets/`.
2. Start the app.
3. Click `Load Widget`.
4. Select module and class from the dropdown dialog.
