# import
import numpy as np
import os
import time
from time import perf_counter
import argparse
import tty
import sys
import termios
import threading

import ComHandler
import SignalProcessor
from flowChart_analyzer import colorGenerator

#Params
timeToKeep = 5
updatesPerSecond = 10
darkMode = False
restTime = 5
EngageTime = 10
postRestTime = 5
totalPrepTime = restTime + EngageTime + postRestTime

class EventListener(threading.Thread):
    def __init__(self, startTime):
        threading.Thread.__init__(self)
        self.events = {}
        self.startTime = startTime
        self.terminated = False
        self.terminationState = "Ok" # Ok, Cancel
        self.start()

    def run(self):
        print("Press Enter to exit and save data")
        print("      ESC for exit without saving")

        print("\n\nPress any alpha-numeric key to add event named after the pressed key\n")

        orig_settings = termios.tcgetattr(sys.stdin)

        # Disable line buffering so we can read keys immediately
        tty.setcbreak(sys.stdin)
        while not self.terminated:
            x=sys.stdin.read(1)[0]
            
            if x == '\n': # Enter -> exit and save
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)   
                self.terminated = True

            elif x == '\x1b': # ESC -> exit without saving
                self.terminationState = "Cancel"
                self.terminated = True

            # only add alpha-numeric keys as events
            elif 0x21 <= ord(x) <= 0x7e:
                self.addEvent(x)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)

    def getEventArrays(self):
        eventDict = {}
        endTime = time.time() - self.startTime

        for eventName, eventTimes in self.events.items():
            eventArray = np.array([[]],ndmin=2)

            # Events allways starts with 0 at time 0
            eventArray = np.append(eventArray, [0,0])

            eventState = False
            for eventTime in eventTimes:
                eventState = not eventState
                eventArray = np.append(eventArray, [eventTime, eventState])

            # Events allways ends with 0 at endTime
            eventArray = np.append(eventArray, [endTime, 0])

            eventDict[eventName] = eventArray
        return eventDict

    def addEvent(self, event):
        # get time
        relTime = time.time() - self.startTime
        # add event to group
        if not event in self.events:
            self.events[event] = []
        self.events[event].append(relTime)
        enabledEvent = len(self.events[event]) % 2 == 1
        print("event " + event + " " + ("enabled" if enabledEvent else "disabled"))

class CLI_Handler:
    def __init__(self, filename, sampleRate, gain, useGui):

        self.dataHandler = SignalProcessor.DataHandler()
        self.comHandler = ComHandler.ComHandler(updatesPerSecond)
        self.comHandler.setCallback(self.addData)
        self.signalProcessor = SignalProcessor.SignalProcessor(self.dataHandler)

        self.samplerate = sampleRate
        self.gain = gain
        self.filename = filename

        self.useGui = useGui
        self.initDone = False

        self.tFirstMeasurement = None
        self.tLastMeasurement = None

        self.latestSampleTime = 0
        self.startSampleTime = 0
        self.tStart = 0 # time when the first sample was received in seconds
        self.isLearning = False
        self.createLabelData()

    def createLabelData(self):
        # Labeltimes in the trapezoid format
        labelData = []
        labelData.append([(2,0), (4,1), (8,1), (10,0), (12,0), (14,-1), (18,-1), (20,0)])
        labelData.append([(22,0), (24,1), (28,1), (30,0), (32,0), (34,-1), (38,-1), (40,0)])
        labelData.append([(42,0), (44,1), (48,1), (50,0), (52,0), (54,-1), (58,-1), (60,0)])
        labelData.append([(62,0), (64,1), (68,1), (70,0), (72,0), (74,-1), (78,-1), (80,0)])
        labelData.append([(82,0), (84,1), (88,1), (90,0), (92,0), (94,-1), (98,-1), (100,0)])
        labelData.append([(102,0), (104,1), (108,1), (110,0), (112,0), (114,-1), (118,-1), (120,0)])
        self.nLabels = len(labelData)

        dataPoints = 0
        for data in labelData:
            dataPoints += len(data)

        labelTimes = np.zeros(dataPoints)
        labelValues = np.zeros((len(labelData), dataPoints))

        for i, data in enumerate(labelData):
            for j, (time, value) in enumerate(data):
                labelTimes[i*len(data) + j] = time
                labelValues[i, i*len(data) + j] = value
        labelTimes += totalPrepTime
        self.dataHandler.addData(labelTimes, labelValues, "labelData")

    def startLearning(self):
        self.isLearning = True
        self.startSampleTime = self.latestSampleTime

    def startNormal(self):
        self.isLearning = False
        self.startSampleTime = self.latestSampleTime

    def stop(self):
        from PySide6 import QtWidgets  # Should work with PyQt5 / PySide2 / PySide6 as well
        import pyqtgraph as pg  
        from pyqtgraph.Qt import QtCore, QtWidgets

        currentDir = os.getcwd()
        emgTimes, emgValues = self.dataHandler.getDataInRange(self.startSampleTime, self.latestSampleTime, "raw")
        labelTimes, labelValues = self.dataHandler.getDataInRange(0, self.latestSampleTime - self.startSampleTime, "labelData")

        emgTimesCorrected = np.copy(emgTimes) - self.startSampleTime

        # Get filename from text input
        filename = self.nameInput.text()
        filename = os.path.join(currentDir, filename)


        np.savez(filename, emgTimes = emgTimesCorrected, emgValues = emgValues, labelTimes = labelTimes, labelValues = labelValues)

    def addData(self, times, values):
        if self.initDone:
            self.dataHandler.addData(times, values)

            self.latestSampleTime = times[-1]

            if self.tFirstMeasurement is None:
                self.tFirstMeasurement = time.time()
            self.tLastMeasurement = time.time()

        # calculate fps
        if hasattr(self, "lastTime") and hasattr(self, "avgFps"):
            self.fps = 1 / (perf_counter() - self.lastTime)
            self.avgFps = self.avgFps * 0.95 + self.fps * 0.05
        self.lastTime = perf_counter()

    def start(self):
        self.comHandler.initCommunication()

        print("\n\n*** FreeThetics Data Logger ***\n")
        print("File will be saved to " + self.filename)
        print()

        # configure adc
        while not self.comHandler.comminucationIsInitialized():
            time.sleep(0.1)
            self.run()

        # Force updatePerSecond 
        self.comHandler.forceUpdatesPerSecond(updatesPerSecond)

        while not self.comHandler.validateConfig(self.samplerate, self.gain):
            time.sleep(0.1)
            self.comHandler.sampleRate = self.samplerate
            self.comHandler.gain = self.gain
            self.run()

        # print current config
        print("Samplingrate: {}Hz".format(self.comHandler.sampleRate), end=' | ')
        print("Gain: {}".format(self.comHandler.gain), end='\n\n')

        self.eventListener = EventListener(time.time())

        self.initDone = True

        # start gui
        if self.useGui:
            self.setupUI()

        while not self.eventListener.terminated:
            self.run()  
        
        # get events
        self.events = self.eventListener.getEventArrays()

        self.terminate(self.eventListener.terminationState)

    def run(self):
        t1Start = perf_counter()
        self.comHandler.run()
        t1End = perf_counter()
        self.signalProcessor.run()

        if hasattr(self, "avgT"):
            self.avgT[0] = self.avgT[0] * 0.8 + (t1End - t1Start) * 0.2
        

    def updateUi(self):
        self.run()

        # Display user commands
        tNow = self.latestSampleTime - self.startSampleTime
        if self.startSampleTime > 0:
            if tNow < restTime:
                self.userCommandLabel.setText("Rest")
            elif tNow < (restTime + EngageTime):
                self.userCommandLabel.setText("Engage all muscles")
            elif tNow < (totalPrepTime):
                self.userCommandLabel.setText("Rest")
            else:
                self.userCommandLabel.setText("Start")

        rawTimes, rawValues = self.dataHandler.getData(timeToKeep, "raw")


        if rawTimes is None:
            return []

        rawTimes = rawTimes[:rawValues.shape[1]]
        t2Start = perf_counter()

        for i, plotItem in enumerate(self.plotItems):
            plotItem.setData(rawTimes, rawValues[i])

        if self.eventListener.terminated:
            self.w.close()

        
        if self.isLearning:
            startTime = tNow - 15/2
            endTime = startTime + 15
            labelTimes, labelValues = self.dataHandler.getDataInRange(startTime, endTime, "labelData")

            if labelTimes is not None and np.shape(labelTimes)[0] != 0:

                #prepend the first value to the labelValues
                labelTimes = np.insert(labelTimes, 0, startTime)
                labelValues = np.insert(labelValues, 0, labelValues[:,0], axis=1)

                # append the last value to the labelValues
                labelTimes = np.append(labelTimes, endTime)
                labelValues = np.append(labelValues, labelValues[:,-1,None], axis=1)

                labelTimes = labelTimes - tNow 

                if labelTimes is not None:
                    for i, plotItem in enumerate(self.outputPlotItems):
                        if i < len(labelValues):
                            plotItem.setData(labelTimes, labelValues[i])

        t2End = perf_counter()
        
        self.avgT[1] = self.avgT[1] * 0.8 + (t2End - t2Start) * 0.2
        self.t1Label.setText("T1: {:.2f} ms".format(self.avgT[0]*1000))
        self.t2Label.setText("T2: {:.2f} ms".format(self.avgT[1]*1000))
        self.fpsLabel.setText("FPS: {:.2f}".format(self.fps))

    def setupUI(self):
        from PySide6 import QtWidgets  # Should work with PyQt5 / PySide2 / PySide6 as well
        import pyqtgraph as pg  
        from pyqtgraph.Qt import QtCore, QtWidgets

        ## Always start by initializing Qt (only once per application)
        self.app = QtWidgets.QApplication([])

        ## Define a top-level widget to hold everything
        self.w = QtWidgets.QWidget()
        self.w.setWindowTitle('FreeThetics Data Logger')
        self.w.resize(800,800)
        pg.setConfigOptions(antialias=True)
        if not darkMode:
            pg.setConfigOption('background', 'w')

        ## Create some widgets to be placed inside
        self.plots = [pg.PlotWidget() for i in range(6)]
        self.plotItems = [plot.plot([0]) for plot in self.plots]

        self.outputPlots = [pg.PlotWidget() for i in range(self.nLabels)]
        self.outputPlotItems = [plot.plot([0]) for plot in self.outputPlots]

        ## Create a grid layout to manage the widgets size and position
        self.layout = QtWidgets.QVBoxLayout()
        self.w.setLayout(self.layout)

        # Add labels
        self.headerLayout = QtWidgets.QHBoxLayout()
        self.layout.addLayout(self.headerLayout)
        self.t1Label = QtWidgets.QLabel("T1: ")
        self.t2Label = QtWidgets.QLabel("T2: ")
        self.fpsLabel = QtWidgets.QLabel("FPS: ")
        self.headerLayout.addWidget(self.t1Label)
        self.headerLayout.addWidget(self.t2Label)
        self.headerLayout.addWidget(self.fpsLabel)
        self.lastDuration = 0.0
        self.lastUpdate = perf_counter()
        self.avgFps = 0.0
        self.avgT = [0.0, 0.0]

        # Add text input for name
        self.nameInput = QtWidgets.QLineEdit()
        self.nameInput.setText(self.filename)
        self.layout.addWidget(self.nameInput)

        # Add Buttons
        self.startRawButton = QtWidgets.QPushButton("Start")
        self.startRawButton.clicked.connect(lambda: self.startNormal())
        self.startLearningButton = QtWidgets.QPushButton("Start Learning")
        self.startLearningButton.clicked.connect(lambda: self.startLearning())
        self.stopButton = QtWidgets.QPushButton("Stop")
        self.stopButton.clicked.connect(lambda: self.stop())

        self.headerLayout.addWidget(self.startRawButton)
        self.headerLayout.addWidget(self.startLearningButton)
        self.headerLayout.addWidget(self.stopButton)

        # Add a label for commands to the user
        self.userCommandLabel = QtWidgets.QLabel()
        self.userCommandLabel.setStyleSheet("font-size: 30pt")
        self.userCommandLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.userCommandLabel)

        # Add a two-column layout for the plots
        self.hLayout = QtWidgets.QHBoxLayout()
        self.layout.addLayout(self.hLayout)

        self.leftLayout = QtWidgets.QVBoxLayout()
        self.rightLayout = QtWidgets.QVBoxLayout()
        self.hLayout.addLayout(self.leftLayout)
        self.hLayout.addLayout(self.rightLayout)

        ## Add realdata plots to the left layout 
        for i, plot in enumerate(self.plots):
            self.leftLayout.addWidget(plot)
            plot.getPlotItem().listDataItems()[0].setPen(pg.mkPen(colorGenerator(i, darkMode), width=1))
            plot.getPlotItem().getAxis('left').setLabel(units='V')

        ## Fill the right layout with widgets for learning/state

        for i, plot in enumerate(self.outputPlots):
            self.rightLayout.addWidget(plot)
            plot.getPlotItem().listDataItems()[0].setPen(pg.mkPen(colorGenerator(0, darkMode), width=2))
            plot.setYRange(-1.1, 1.1)
            plot.setXRange(-5, 5)

            # Add a red line at 0
            plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=2)))
            
        ## Display the widget as a new window
        self.w.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateUi)
        self.timer.start(0)

        ## Start the Qt event loop
        self.app.exec_()  # or app.exec_() for PyQt5 / PySide2


    def terminate(self, terminationState):
        # close USB connection
        self.comHandler.closeCommunication()

        # close GUI
        if self.useGui:
            self.app.quit()

        if terminationState == "Ok":
            print("terminate and save data")
            self.saveValues()
            exit(0)

        elif terminationState == "Cancel":
            print("terminate without saving")
            exit(0)

        else:
            print("Unknown termination state")
            exit(1)

    def saveValues(self):
        currentDir = os.getcwd()
        emgTimes, emgValues = self.dataHandler.getData(0, "raw")

        # compare mcu time with pc time
        if self.tFirstMeasurement is not None:
            timeDiff = self.tLastMeasurement - self.tFirstMeasurement
            print("Time difference on pc: " + str(timeDiff) + "s")
            mcuTime = emgTimes[-1] - emgTimes[0]
            print("Time difference on mcu: " + str(mcuTime) + "s")

            # correct time
            emgTimes = np.linspace(0, timeDiff, len(emgTimes))


        filename = os.path.join(currentDir, self.filename)
        np.savez(filename, emgTimes = emgTimes, emgValues = emgValues, **self.events)

    def updateSampleRate(self, sampleRate):
        print("set sample rate to " + str(sampleRate))
        self.comHandler.sampleRate = sampleRate

    def updateGain(self, val):
        print("set gain to " + str(val))
        self.comHandler.gain = val

    def updateChannel(self, val):
        channel = int(val[-1]) - 1
        print("set channel to " + str(channel))
        self.comHandler.channels = 1 << channel

if __name__ == "__main__":

    possibleContSampleRates = [1.9, 3.9, 7.8, 15.6, 31.2, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    possibleSingleSampleRates = [50, 62.5, 100, 125, 200, 250, 400, 500, 800, 1000, 1600, 2000, 3200, 4000, 6400, 12800]

    parser = argparse.ArgumentParser(description='FreeThetics Data Logger', epilog='Terminate with Ctrl+C saves data to file')

    parser.add_argument('filename', help='filename to save data to')
    parser.add_argument('-g', '--gain', choices=[2**g for g in range(8)], help='gain to use', type=int)
    parser.add_argument('-s', '--samplerate', help='sample rate to use', type=int, choices=possibleContSampleRates + possibleSingleSampleRates, default=2000)
    parser.add_argument('-a', '--auto', help='automaticly add config to filename', action='store_true', default=False)
    parser.add_argument('-gui', '--gui', help='use gui', action='store_true', default=False)
    args = parser.parse_args()

    filename = args.filename.strip()

    if args.auto:
        filename = filename + "_s=" + str(args.samplerate) + "_g=" + str(args.gain)

    while True:
        cliHandler = CLI_Handler(filename, args.samplerate, args.gain, args.gui)
        cliHandler.start()
        del cliHandler
