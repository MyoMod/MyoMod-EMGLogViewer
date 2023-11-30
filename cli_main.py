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

    def addData(self, times, values):
        if self.initDone:
            self.dataHandler.addData(times, values)

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

        rawTimes, rawValues = self.dataHandler.getData(timeToKeep, "raw")

        if rawTimes is None:
            return []

        rawTimes = rawTimes[:rawValues.shape[1]]
        t2Start = perf_counter()

        for i, plotItem in enumerate(self.plotItems):
            plotItem.setData(rawTimes, rawValues[i])

        if self.eventListener.terminated:
            self.w.close()

        t2End = perf_counter()
        
        self.avgT[1] = self.avgT[1] * 0.8 + (t2End - t2Start) * 0.2
        self.t1Label.setText("T1: {}".format(self.avgT[0]))
        self.t2Label.setText("T2: {}".format(self.avgT[1]))
        self.fpsLabel.setText("FPS: {}".format(self.fps))

    def setupUI(self):
        from PySide2 import QtWidgets  # Should work with PyQt5 / PySide2 / PySide6 as well
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

        ## Create a grid layout to manage the widgets size and position
        self.layout = QtWidgets.QVBoxLayout()
        self.w.setLayout(self.layout)

        # Add labels
        self.labelLayout = QtWidgets.QHBoxLayout()
        self.layout.addLayout(self.labelLayout)
        self.t1Label = QtWidgets.QLabel("T1: ")
        self.t2Label = QtWidgets.QLabel("T2: ")
        self.fpsLabel = QtWidgets.QLabel("FPS: ")
        self.labelLayout.addWidget(self.t1Label)
        self.labelLayout.addWidget(self.t2Label)
        self.labelLayout.addWidget(self.fpsLabel)
        self.lastDuration = 0.0
        self.lastUpdate = perf_counter()
        self.avgFps = 0.0
        self.avgT = [0.0, 0.0]


        ## Add widgets to the layout in their proper positions
        for i, plot in enumerate(self.plots):
            self.layout.addWidget(plot)
            plot.getPlotItem().listDataItems()[0].setPen(pg.mkPen(colorGenerator(i, darkMode), width=1))
            plot.getPlotItem().getAxis('left').setLabel(units='V')
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
