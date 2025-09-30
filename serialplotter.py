import sys, time
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QWidget, QCheckBox, QPushButton, QComboBox, QMessageBox, QFileDialog, QGridLayout, QLineEdit, QLabel
from PySide6.QtCore import QTimer, Slot as pyqtSlot, Signal as pyqtSignal
from PySide6.QtGui import QIntValidator
from serialdevice import SerialDevice

class SerialPlotter(QWidget):
    """
    SerialPlotter is a PyQt-based widget for visualizing real-time data from a serial device. 
    It provides functionalities for plotting, managing multiple data lines, and saving data to a CSV file.
    Attributes:
        portChanged (pyqtSignal): Signal emitted when the serial port is changed.
        aquisitionStarted (pyqtSignal): Signal emitted when data acquisition starts.
        aquisitionStopped (pyqtSignal): Signal emitted when data acquisition stops.
        SampleLengthChanged (pyqtSignal): Signal emitted when the sample length is updated.
        serialDevice (SerialDevice): Instance of the SerialDevice class for managing serial communication.
        data (dict): Dictionary to store data for each line.
        x_data (dict): Dictionary to store x-axis data for each line.
        timer (QTimer): Timer for updating the plot at regular intervals.
        plot_running (bool): Flag indicating whether the plot is running.
        maxplotlength (int): Maximum number of data points to display on the plot.
        timestep (float): Time step between data points.
        sendCommands (bool): Flag indicating whether commands are being sent to the serial device.
        command (str): Command to be sent to the serial device.
    Methods:
        __init__(self, positional, *args, **kwargs): Initializes the SerialPlotter widget.
        getSerialDevice(self): Returns the SerialDevice instance.
        init_ui(self): Initializes the user interface components.
        test(self, event): Populates the list of available serial ports.
        isPlotting(self): Returns whether the plot is currently running.
        toggle_plot(self): Starts or stops the data acquisition and plotting.
        update_maxplotlength(self): Updates the maximum plot length and reinitializes data arrays.
        activateLine(self, line_index, label=None): Activates a specific data line for plotting.
        deactivateLine(self, line_index): Deactivates a specific data line.
        setTimestep(self, timestep): Sets the time step between data points.
        setXAxisTicks(self, timestep): Configures the x-axis ticks based on the time step.
        setSecondXAxisTicks(self, xtick_data, index=1): Configures a secondary x-axis with custom tick data.
        setYAxisTicks(self, min=0, max=1): Configures the y-axis ticks and labels.
        initCanvas(self): Initializes the plot canvas and data structures.
        plotData(self, data, index=0, **kwargs): Plots data on a specific line.
        populate_ports(self): Populates the dropdown with available serial ports.
        selected_port_changed(self): Handles changes in the selected serial port.
        setSendCommand(self): Toggles the sending of commands to the serial device.
        toggle_line(self, line_index, state): Toggles the visibility of a specific data line.
        update_plot(self): Updates the plot with new data from the serial device.
        save_to_csv(self): Saves the plotted data to a CSV file.
        closeEvent(self, event): Handles the widget's close event and closes the serial connection.
    """

    portChanged = pyqtSignal(str)
    aquisitionStarted = pyqtSignal(str)
    aquisitionStopped = pyqtSignal(str)
    SampleLengthChanged = pyqtSignal(int)
    filesizeupdate = pyqtSignal(float)
    SigPlotUpdated = pyqtSignal()

    Samples_per_second = time.time()
    samplecount = 0
    SPS = 0

    def __init__(self, positional, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.oldtime = time.time_ns()
        self.serialDevice = SerialDevice(baudrate=921600, timeout=1)

        self.maxNrLines = 5
        self.datalines : list[pg.PlotDataItem] = []
        self.secondary_datalines : list[pg.PlotDataItem] = []
        self.xData : list[np.ndarray] = []
        self.secondary_xData : list[np.ndarray] = []
        self.yData : list[np.ndarray] = []
        self.secondary_yData : list[np.ndarray] = []

        self.outfile = None
        self.outfilename = "output.csv"
        self.streamToFile = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.plot_running = False

        self.maxplotlength = 1000
        #Data rate is 1 kHz
        self.timestep = 1e-3

        self.incomingDataScaling = 1/1000 # Incoming Data is in mbar
        self.yscale = 0.75  # conversion factor to mmHg
        self.yUnit = "mHg"  # Shown data unit

        self.sendCommands = False
        self.command = "VAL?"
        
        self.plotLayout = pg.GraphicsLayoutWidget(border='w')

        self.init_ui()
        
        self.livePlotItem = self.plotLayout.addPlot(title="Live Data", row=1, col=0)
        self.livePlotItem.addLegend(offset=(0,1))
        self.livePlotItem.setTitle("Live Data")
        
        self.frequency_PlotItem = self.plotLayout.addPlot(title="FFT", row=2, col=0)
        self.frequency_PlotItem.addLegend(offset=(0,1))
        self.frequency_PlotItem.setTitle("FFT")

        self.initCanvas()

        # self.setOutputToFile("output.csv")

        t = self.printstats(0)
        self.statlabel = pg.LabelItem(text=t)
        self.plotLayout.addItem(self.statlabel, row=0, col=0)


    def setOutputToFile(self, filename):
        """Outputs the data to a file"""
        self.outfile = open(filename, "w+")
        self.streamToFile = True
        self.timer.timeout.disconnect()
        self.timer.timeout.connect(self.update_file)

    def setOutputToPlot(self):
        """Outputs the data to the plot"""
        if not self.outfile.closed:
            self.outfile.flush()
            self.outfile.close()
        
        self.streamToFile = False
        self.timer.timeout.disconnect()
        self.timer.timeout.connect(self.update_plot)
        
    def toggle_points(self, checked):
        for i in range(self.maxNrLines):
            self.datalines[i].setSymbol('o' if checked else None)
            self.secondary_datalines[i].setSymbol('o' if checked else None)

    def toggle_stats(self, checked):
        self.statlabel.setVisible(checked)

    def toggle_grid(self, checked):
        self.livePlotItem.showGrid(y=checked)
        self.frequency_PlotItem.showGrid(x=checked)

    def toggleUnits(self, checked):
        #Default Units are in mbar
        if checked:
            self.yUnit = "mHg"
            self.yscale = (1/1000)*0.750061683
            self.livePlotItem.getAxis('left').setLabel(text="Amplitude", units=self.yUnit)
            # self.livePlotItem.getAxis('left').enableAutoSIPrefix(False)
            self.livePlotItem.getAxis('left').setScale(self.yscale)
            self.frequency_PlotItem.getAxis('left').setScale(self.yscale)
        else:
            self.yscale = 1/1000
            self.yUnit = "Bar"
            self.livePlotItem.getAxis('left').setLabel(text="Amplitude", units=self.yUnit)
            self.livePlotItem.getAxis('left').enableAutoSIPrefix(True)
            self.livePlotItem.getAxis('left').setScale(self.yscale)
            self.frequency_PlotItem.getAxis('left').setScale(self.yscale)

    def setAxis(self, settings: dict, axis='left'):
        """Sets the axis settings for the plot"""
        if 'Label' in settings and 'Units' in settings:
            self.livePlotItem.getAxis(axis).setLabel(text=settings['Label'], units=settings['Units'])
        if 'Scale' in settings:
            self.livePlotItem.getAxis(axis).setScale(settings['Scale'])
        if 'Font' in settings:
            self.livePlotItem.getAxis(axis).setStyle(tickFont=settings['Font'])
        if 'SIPrefix' in settings:
            self.livePlotItem.getAxis(axis).enableAutoSIPrefix(settings['SIPrefix'])


    def initCanvas(self):
        # self.livePlotItem.plotItem.addLegend()
        for i in range(self.maxNrLines):  # Assuming there are at most 5 lines
            x = np.arange(self.maxplotlength)
            y = np.ones(self.maxplotlength)
            self.datalines.append(self.livePlotItem.plot(x, y*i, pen=(i, 5), width=1, name=f"Line {i}"))
            self.secondary_datalines.append(self.frequency_PlotItem.plot(x, y*i, pen=(i, 5), width=1, name=f"Line {i}"))
            self.xData.append(x)
            self.yData.append(y)
            self.secondary_xData.append(x)
            self.secondary_yData.append(y)
            self.secondary_datalines[i].setVisible(False)
            self.datalines[i].scatter.opts.update(hoverable=True, tip='x: {x:.2f}, y: {y:.5f}(mBar)'.format, hoverSymbol='+', hoverSize=10)
            self.secondary_datalines[i].scatter.opts.update(hoverable=True, tip='x: {x:.5f}, y: {y:.5f}(mBar)'.format, hoverSymbol='+', hoverSize=10)
            self.datalines[i].setSkipFiniteCheck(True)
            if i > 0:
                self.datalines[i].setVisible(False)

        # Update statistics for the pressure line
        # self.SigPlotUpdated.connect(self.displaySignalInfo)
        self.livePlotItem.getAxis('bottom').setScale(self.timestep)
        #Scale according to incoming data
        self.livePlotItem.getAxis('left').setScale(self.yscale*self.incomingDataScaling)
        tickfont = pg.Qt.QtGui.QFont('Arial', 14)
        self.livePlotItem.getAxis('bottom').setStyle(tickFont=tickfont)
        self.livePlotItem.getAxis('left').setStyle(tickFont=tickfont)
        labelStyle = {'font-size': '14pt', 'color': '#FFF'}
        self.livePlotItem.getAxis('bottom').setLabel('Time', 's', **labelStyle)#, siPrefixEnableRanges((1e-12,1)))
        self.livePlotItem.getAxis('left').setLabel('Amplitude', self.yUnit, **labelStyle)
        self.frequency_PlotItem.getAxis('bottom').setLabel('Frequency', 'Hz',**labelStyle)
        self.frequency_PlotItem.getAxis('left').setLabel('FFT', self.yUnit,**labelStyle)
        self.frequency_PlotItem.getAxis('bottom').setStyle(tickFont=tickfont)
        self.frequency_PlotItem.getAxis('left').setStyle(tickFont=tickfont)

        self.frequency_PlotItem.showGrid(x=True)
        self.livePlotItem.showGrid(y=True)

    def getSerialDevice(self):
        return self.serialDevice
    
    def printstats(self, index:int):
        avg = self.yData[index].mean()
        std = self.yData[index].std()
        # min = self.yData[index].min()
        # max = self.yData[index].max()
        # median = np.median(self.yData[index])
        return f"Avg: {avg:.5f}\tStd: {std:.5f} (mBar)"

    def getAverage(self):
        return float(self.yData[1].mean())

    def displaySignalInfo(self, evt):
        stattext = ""
        for idx, line in enumerate(self.datalines):
            if line.isVisible():
                stattext += str(line.name()) + ": " + self.printstats(idx) + "\t"

        if self.samplecount > self.maxplotlength:
            deltatime = time.time() - self.Samples_per_second
            if deltatime > 0:
                self.SPS = self.samplecount / deltatime

            self.Samples_per_second = time.time()
            self.samplecount = 0

        stattext += str(int(self.SPS)) + " Samples/s"

        self.statlabel.setText(stattext)

    def init_ui(self):
        self.line_select_layout = QGridLayout()
        self.line_select_widget = QWidget()
        self.line_select_widget.setLayout(self.line_select_layout)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.plotLayout)
        #self.layout.addWidget(self.line_select_widget)
        
        self.start_stop_button = QPushButton("Start DAQ & Plot")
        self.start_stop_button.clicked.connect(self.toggle_plot)

        self.save_button = QPushButton("Save to CSV")
        self.save_button.clicked.connect(self.save_to_csv)
        
        self.serialDevice_layout = QHBoxLayout()
        self.serialDevice_Baudrate = QComboBox()
        self.serialDevice_Baudrate.addItems(["921600", "115200","9600"])
        self.serialDevice_Baudrate.setCurrentText("921600")
        self.serialDevice_Baudrate.setToolTip("Baudrate")
        self.serialDevice_Baudrate.currentTextChanged.connect(self.on_baudrate_changed)
        self.port_dropdown = QComboBox()
        self.port_dropdown.enterEvent = self.populate_ports
        self.populate_ports(None)
        self.port_dropdown.currentIndexChanged.connect(self.selected_port_changed)

        self.serialDevice_layout.addWidget(self.port_dropdown)
        self.serialDevice_layout.addWidget(self.serialDevice_Baudrate)
        
        self.layout.addLayout(self.serialDevice_layout)

        self.maxplotlen_label = QLabel("Max Plot Length:")
        self.maxplotlen_input = QLineEdit(str(self.maxplotlength))
        self.maxplotlen_input.setValidator(QIntValidator(1, 100000))
        self.maxplotlen_input.editingFinished.connect(self.update_maxplotlength)
        
        self.layout.addWidget(self.maxplotlen_label)
        self.layout.addWidget(self.maxplotlen_input)
        self.checkboxes = []

        # for i in range(self.maxNrLines):  # Assuming there are at most 5 lines
        #     checkbox = QCheckBox(f'Line {i+1}')
        #     self.checkboxes.append(checkbox)
        #     checkbox.setChecked(False)
        #     checkbox.stateChanged.connect(self.on_state_changed)
        #     self.line_select_layout.addWidget(checkbox, 0, i)
        
        self.label_pressure = QLabel(f"Command: {self.command}")
        self.line_select_layout.addWidget(self.label_pressure, 1, 0)

        self.edit_pressure = QLineEdit(self.command)
        self.line_select_layout.addWidget(self.edit_pressure, 1, 1, 1, 3)

        self.acceptButton = QPushButton("OK")
        self.acceptButton.clicked.connect(self.setSendCommand)
        self.line_select_layout.addWidget(self.acceptButton, 1, 4)

        self.clearButton = QPushButton("Clear Plot Data")
        self.clearButton.clicked.connect(self.clearPlot)
        self.line_select_layout.addWidget(self.clearButton, 1, 5)
        
        self.layout.addWidget(self.start_stop_button)
        self.layout.addWidget(self.save_button)
        
        self.setLayout(self.layout)

    def isPlotting(self):
        return self.plot_running
    
    @pyqtSlot()
    def toggle_plot(self):
        if not self.plot_running:
            if not self.serialDevice.is_open:
                QMessageBox.warning(self, "Error", "Please select a COM port.")
                return
            self.start_stop_button.setText("Stop DAQ & Plot")
            self.plot_running = True
            self.aquisitionStarted.emit("Started")
            self.timer.start(1)  # Update plot every ms
            if self.streamToFile:
                self.outfile = open(self.outfilename, "a+")
        else:
            self.start_stop_button.setText("Start DAQ & Plot")
            self.timer.stop()
            self.serialDevice.flush()
            self.plot_running = False
            self.aquisitionStopped.emit("Stopped")
            if self.streamToFile:
                self.outfile.flush()
                self.outfile.close()

    def getMaxPlotLength(self):
        return self.maxplotlength
    
    def setLineNames(self, names:list[str]):
        for i in range(len(names)):
            if i < len(self.datalines):
                self.activateLine(i, names[i])
                self.datalines[i].setName(names[i])
            else:
                break

    @pyqtSlot()
    def update_maxplotlength(self, length=None):
        self.maxplotlength = int(self.maxplotlen_input.text()) if length is None else length
        for idx, line in enumerate(self.datalines):
            xData = self.xData[idx]
            yData = self.yData[idx]
            N = len(yData)
            if N > self.maxplotlength:
                N = self.maxplotlength
            newX = np.arange(self.maxplotlength)
            newY = np.zeros(self.maxplotlength)
            newY[:N] = yData[:N]
            line.setData(newX, newY)
            self.xData[idx] = newX
            self.yData[idx] = newY

        self.SampleLengthChanged.emit(self.maxplotlength)

    def changeLineName(self, line_index, new_name):
        if line_index >= len(self.datalines) or line_index < 0:
            raise ValueError("Index out of range")
        self.datalines[line_index].name = new_name

        self.livePlotItem.legend.removeItem(self.datalines[line_index])
        self.livePlotItem.legend.addItem(self.datalines[line_index], new_name)

    def activateLine(self, line_index, label=None):
        if label is None:
            label = f"Line {line_index}"
        self.datalines[line_index].setData(self.datalines[line_index].getData()[1], name=label)
        self.datalines[line_index].setVisible(True)

        self.livePlotItem.legend.removeItem(self.datalines[line_index])
        self.livePlotItem.legend.addItem(self.datalines[line_index], label)
    
    def deactivateLine(self, line_index):
        self.datalines[line_index].setVisible(False)
        #self.checkboxes[line_index].setChecked(False)

    def setTimestep(self, timestep):
        self.timestep = timestep
        self.livePlotItem.getAxis('bottom').setScale(self.timestep)

    def plotData(self, yData, /, xData=None, index=0, **kwargs):
        if index >= len(self.datalines) or index < 0:
            raise ValueError("Index out of range")

        label = f"Line {index}"
        if "label" in kwargs:
            label = kwargs["label"]

        self.datalines[index].setVisible(True)
        #self.checkboxes[index].setChecked(True)
        #self.checkboxes[index].setText(label)
        if xData is not None:
            self.datalines[index].setData(xData, yData, name=label)
            self.xData[index] = xData
        else:
            self.datalines[index].setData(yData, name=label)

        self.yData[index] = yData

        self.livePlotItem.legend.removeItem(self.datalines[index])
        self.livePlotItem.legend.addItem(self.datalines[index], label)

    def plotSecondaryData(self, yData, xData=None, index=1, **kwargs):
        if index >= len(self.datalines) or index < 0:
            raise ValueError("Index out of range")
        
        label = f"Line {index}"
        if "label" in kwargs:
            label = kwargs["label"]

        self.secondary_datalines[index].setVisible(True)

        if xData is not None:
            self.secondary_datalines[index].setData(xData, yData, name=label)
            self.secondary_xData[index] = xData
        else:
            self.secondary_datalines[index].setData(yData, name=label)
        self.secondary_yData[index] = yData

        self.frequency_PlotItem.legend.removeItem(self.secondary_datalines[index])
        self.frequency_PlotItem.legend.addItem(self.secondary_datalines[index], label)

    def populate_ports(self, evt):
        ports = self.serialDevice.list_devices()  # Use SerialDevice to list ports
        ports.insert(0, (" ","Disconnect"))
        if len(ports) != self.port_dropdown.count():
            self.port_dropdown.clear()
            self.port_dropdown.addItems([f"{p[0]}-({p[1]})" for p in ports])

    @pyqtSlot()
    def selected_port_changed(self):
        port = self.port_dropdown.currentText().split('-')[0]
        if port == " ":
            self.serialDevice.close()  # Use SerialDevice to close the connection
            self.portChanged.emit(" ")
            return
        try:
            self.serialDevice.open_connection(port)  # Use SerialDevice to open the connection
            self.portChanged.emit(port)

        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            return

    def setSendCommand(self):
        self.command = self.edit_pressure.text()
        self.label_pressure.setText(f"Command: {self.command}")
        self.sendCommands = not self.sendCommands
        self.acceptButton.setText("Stop Sending" if self.sendCommands else "OK")

    def toggle_line(self, line_index, state):
        #selff.datalines[line_index].setVisible(self.checkboxes[line_index].isChecked())
        pass


    def update_plot(self):
        if self.sendCommands:
            self.serialDevice.writeCommand(self.command)  # Use SerialDevice to send commands
        
        while self.serialDevice.is_open and self.serialDevice.getInWaiting() > 0:
            try:
                line = self.serialDevice.readLine().decode()  # Use SerialDevice to read line
                values = line.split(',')
                for i, val in enumerate(values):
                    if i >= self.maxNrLines:
                        break
                    data_point = float(val)
                    self.yData[i] = np.roll(self.yData[i], 1)  # Roll data to the left
                    self.yData[i][0] = data_point  # Update the last value with new data
                self.samplecount += 1
            except ValueError as e:
                print("UPDATE_PLOT:", e)
            except ConnectionError as e:
                print("UPDATE_PLOT:", e)


        for i in range(self.maxNrLines):
            if self.datalines[i].isVisible():
                self.datalines[i].setData(self.xData[i], self.yData[i])

        # Calculate and plot FFT for each visible line
        for i in range(self.maxNrLines):
            if self.datalines[i].isVisible():
                y = self.yData[i]
                # Remove mean to avoid DC offset
                y = y - np.mean(y)
                fft_result = np.fft.fft(y)/len(y)
                fft_freq = np.fft.fftfreq(len(y), d=self.timestep)
                # Only plot the positive frequencies
                pos_mask = fft_freq >= 0
                self.secondary_datalines[i].setData(fft_freq[pos_mask], np.abs(fft_result[pos_mask]))
                self.secondary_datalines[i].setVisible(True)
            else:
                self.secondary_datalines[i].setVisible(False)

        self.displaySignalInfo(None)
        

    def update_file(self):
        while self.serialDevice.is_open and self.serialDevice.getInWaiting() > 0:
            try:
                line = self.serialDevice.readLine().decode()  # Use SerialDevice to read line
                values = line.split(',')
                #Init String with timestamp
                stri = f"{time.time()}"
                for i, val in enumerate(values):
                    data_point = float(val)
                    stri += f",{data_point}"
                    self.yData[i] = np.roll(self.yData[i], 1)  # Roll data to the left
                    self.yData[i][0] = data_point  # Update the last value with new data
                self.outfile.write(stri + "\n")
                self.samplecount += 1
            except ValueError as e:
                print("UPDATE_FILE:", e)
            except ConnectionError as e:
                print("UPDATE_FILE:", e)
                
            filesize = self.outfile.tell()/(1024*1024)  # Get file size in MB
            # Emit signal once per MB of filesize
            if not hasattr(self, "_last_emitted_mb"):
                self._last_emitted_mb = -1
            current_mb = int(filesize)
            if current_mb != self._last_emitted_mb:
                self.filesizeupdate.emit(filesize)
                self._last_emitted_mb = current_mb

        for i in range(self.maxNrLines):
            if self.datalines[i].isVisible():
                self.datalines[i].setData(self.xData[i], self.yData[i])

        self.displaySignalInfo(None)            

    def save_to_csv(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save to CSV", "", "CSV Files (*.csv)")
        if not filename:
            return
        # filename_sec = filename.replace(".csv", "_secondary.csv")
        outdata = np.zeros((self.maxNrLines+1, self.maxplotlength))
        outdata_sec = np.zeros((self.maxNrLines+1, self.maxplotlength))

        for i in range(self.maxNrLines):
                if self.datalines[i].isVisible():
                    x, y = self.datalines[i].getData()
                    outdata[i+1,:len(y)] = y
                    outdata[0,:len(x)] = x * self.timestep

        for i in range(self.maxNrLines):
            if self.secondary_datalines[i].isVisible():
                x, y = self.secondary_datalines[i].getData()
                outdata_sec[i+1,:len(y)] = y
                outdata_sec[0,:len(x)] = x * self.timestep

        header = []
        header_sec = []
        #Create time axis
        units = self.livePlotItem.getAxis('bottom').labelUnits
        if units == 's':
            outdata[0,:] = outdata[0,:] * 1000 # Convert to ms
        elif units == 'ms':
            pass # Already in ms
        else:
            outdata[0,:] = outdata[0,:] / 1000 # Convert to ms

        
        liveplotfilter = [line.isVisible() for line in self.datalines]
        liveplotfilter.insert(0, True)  # Include the time axis in the filter

        #Construct headers
        #Add time axis
        header.append(str(self.livePlotItem.getAxis('bottom').labelText) + '[ms]')
        for i in range(self.maxNrLines):
            if self.datalines[i].isVisible():
                header.append(self.datalines[i].name())
        
        #Write data to file
        with open(filename, 'w') as f:
            # Write header
            f.write(";".join(header) + "\n")
            # Write data

            for data in outdata[liveplotfilter, :].T:
                f.write(";".join(map(lambda x: str(x).replace(".",","), data))+ "\n")


    def clearPlot(self):
        for line in range(self.maxNrLines):
            self.xData[line] = np.arange(self.maxplotlength)
            self.yData[line] = np.zeros(self.maxplotlength)
            self.secondary_xData[line] = np.arange(self.maxplotlength)
            self.secondary_yData[line] = np.zeros(self.maxplotlength)
            self.datalines[line].setData(self.xData[line], self.yData[line])
            self.secondary_datalines[line].setData(self.secondary_xData[line], self.secondary_yData[line])

    def on_state_changed(self, state, i):
        self.toggle_line(i, state == 2)

    def on_baudrate_changed(self, text):
        self.serialDevice.setBaudrate(int(text))

    def closeEvent(self, event):
        self.serialDevice.close_connection()  # Use SerialDevice to close the connection
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = QWidget()
    layout = QVBoxLayout(window)
    serial_plotter = SerialPlotter(None)
#    serial_plotter.livePlotItem.getViewBox().enableAutoRange(True)

    serial_plotter.update_maxplotlength(1000)

    serial_plotter.plotData(np.arange(100,100+1000)/1000, index=0, label="Time")
    serial_plotter.setTimestep(1/1000)
    # Create two sine waves with different frequencies
    t = np.linspace(0, 1, 1000, endpoint=False)  # Time vector
    #serial_plotter.setTimestep(1/1000)
    freq1, freq2 = 5, 20  # Frequencies in Hz
    sine_wave1 = np.sin(2 * np.pi * freq1 * t)
    sine_wave2 = np.sin(2 * np.pi * freq2 * t)

    # Combine the sine waves
    combined_wave = sine_wave1 + sine_wave2

    # Plot the combined wave
    serial_plotter.plotData(combined_wave, index=1, label="Combined Wave")

    # Calculate FFT
    fft_result = np.fft.fft(combined_wave)
    fft_freq = np.fft.fftfreq(len(combined_wave), d=(t[1] - t[0]))

    # Plot the FFT (magnitude spectrum)
    #serial_plotter.plotData(np.abs(fft_result[:len(fft_result)//2]), xData=fft_freq[:len(fft_freq)//2], index=2, label="FFT Magnitude")

    #serial_plotter.setSecondXAxisTicks(fft_freq[:len(fft_freq)//2], index=2)

    layout.addWidget(serial_plotter)
    window.setWindowTitle("Serial Plotter")
    window.resize(1280, 1024)
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
