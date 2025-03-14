"""
This example demonstrates writing a custom Node subclass for use with flowcharts.

We implement a couple of simple image processing nodes.
"""

import numpy as np
import os

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart import Flowchart, Node
from pyqtgraph.flowchart.library.common import CtrlNode
import scipy as sp
import scipy.signal as signal
from pyqtgraph.metaarray import MetaArray

import emg_flowChart.nodes as EMG_Nodes

def colorGenerator(index, darkMode=False):  
    if darkMode:
        color = pg.intColor(index, hues=7, alpha=255)
    else:   
        tab10 = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
                (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
                (188, 189, 34), (23, 190, 207)]
        color = pg.mkColor(tab10[index%10])
    return color

class EMG_FlowChart():

    def __init__(self, win, darkMode=False):
        self.darkMode = darkMode

        self.layout = None
        self.widgets = []
        self.channelCheckboxes = []
        self.eventPlot = None
        self.region = None
        self.fc = None

        self.cmpltTimes = None
        self.cmpltValues = None
        self.eventData = None
        self.labelData = None
        self.labelTimes = None

        self.timeRange = None
        self.currentFile = None

        self.setupUi(win)

    def loadFromFile(self, filename):
        self.currentFile = filename
        data = np.load(filename, allow_pickle=True)

        labelData = None
        labelTimes = None

        if "timeArray" in data:
            emgTimes = data["timeArray"]
            emgValues = data["valueArray"]
        elif "eventTimes" in data:
            emgTimes = data["emgTimes"]
            emgValues = data["emgValues"]

            eventData = dict()
            if "eventTimes" in data:
                eventTimes = data["eventTimes"]
                eventValues = data["eventValues"]
            
            eventData["UI-Event"] = np.array([eventTimes, eventValues]).T
        else:
            emgTimes = data["emgTimes"]
            emgValues = data["emgValues"]

            if "labelValues" in data:
                labelData = data["labelValues"]
                labelTimes = data["labelTimes"]

            eventData = dict( data)
            if "labelValues" in eventData:
                del eventData["labelTimes"]
                del eventData["labelValues"]
            del eventData["emgTimes"]
            del eventData["emgValues"]

        # make sure that emgValues is a 2D array
        if len(emgValues.shape) == 1:
            emgValues = emgValues.reshape(1, -1)

        # close file
        data.close()
        self.cmpltTimes = emgTimes
        self.cmpltValues = emgValues
        self.eventData = eventData
        self.labelData = labelData
        self.labelTimes = labelTimes

        # Add events on the up-most row
        # add events
        self.eventPlot.clear()
        for i, eventGroup in enumerate(self.eventData.items()):
            eventName = eventGroup[0]
            eventData = eventGroup[1].reshape(-1,2)
            eventTimes = eventData[:,0]
            eventValues = eventData[:,1]

            self.eventPlot.plot(eventTimes, eventValues, stepMode='right', name=eventName, pen=pg.mkPen(i, width=2))

        if self.labelData is not None:
            for i in range(self.labelData.shape[0]):
                labelName = "Label Channel {}".format(i+1)
                labelValues = labelData[i,:]

                self.eventPlot.plot(self.labelTimes, labelValues, name=labelName, pen=pg.mkPen(i, width=2))
            
            #display range -1 to 1
            self.eventPlot.setYRange(-1, 1)
        else:
            self.eventPlot.setYRange(0, 1)
        
        self.eventPlot.addItem(self.region, ignoreBounds=True)
        self.eventPlot.addItem(self.eventVLine, ignoreBounds=True)
        self.updateInput()

        # set event plot x-axis range to the same as the raw data
        completeTimes = self.cmpltTimes
        self.eventPlot.setXRange(self.cmpltTimes[0], self.cmpltTimes[-1])
        startRegion = self.cmpltTimes[0] + (self.cmpltTimes[-1] - self.cmpltTimes[0]) * 0.0
        endRegion = self.cmpltTimes[0] + (self.cmpltTimes[-1] - self.cmpltTimes[0]) * 1.0
        self.region.setRegion([startRegion, endRegion])

    def getChannelSelection(self):
        return [checkbox.isChecked() for checkbox in self.channelCheckboxes]

    def updateInput(self):
        nChannels = self.cmpltValues.shape[0]

        colors = [colorGenerator(i, self.darkMode).name() for i in range(nChannels)]
        cols = [{"name": "Channel {}".format(i+1), "units": "V"} for i in range(nChannels)]

        # select channels
        channelSelection = self.getChannelSelection()[:nChannels]
        values = self.cmpltValues[channelSelection,:]
        colors = [colors[i] for i in range(len(channelSelection)) if channelSelection[i]]
        cols = [cols[i] for i in range(len(channelSelection)) if channelSelection[i]]

        # select time
        if self.timeRange is not None and 0:
            minIndex = np.searchsorted(self.cmpltTimes, self.timeRange[0])
            maxIndex = np.searchsorted(self.cmpltTimes, self.timeRange[1])
            time = self.cmpltTimes[minIndex:maxIndex]
            values = values[:,minIndex:maxIndex]
        else:
            time = self.cmpltTimes

        info = [{"name": "Signal", "units": "V", "cols": cols, "colors": colors},
                {"name": "Time", "units": "sec", "values":time }]
        multiChannelData = MetaArray(values, info=info)

        if self.labelData is not None:
            labelData = MetaArray(self.labelData, info=[{"name": "Label", "cols": [{"name": "Label Channel {}".format(i+1), "units": "V"} for i in range(self.labelData.shape[0])]},
                                                    {"name": "Time", "units": "sec", "values": self.labelTimes}])
        else:
            labelData = MetaArray(np.zeros((1,1)), info=[{"name": "Label", "cols": [{"name": "Label Channel 1", "units": "V"}]},
                                                    {"name": "Time", "units": "sec", "values": [0]}])

        ## Set the raw data as the input value to the flowchart
        self.fc.setInput(dataIn=multiChannelData,labelIn=labelData)

    def updateChannels(self):
        self.updateInput()

    def updatePlotRanges(self):
        timeRange = self.region.getRegion()
        [widget.setXRange(*timeRange, padding = 0) for widget in self.widgets]
        #self.updateInput()

    def selectFile(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select data file", "", "Data Files (*.npz)")
        if filename:
            self.loadFromFile(filename)

    def setupUi(self, win):
        cw = QtWidgets.QWidget()
        win.setCentralWidget(cw)
        self.layout = QtWidgets.QGridLayout()
        cw.setLayout(self.layout)

        # Add a button for loading data
        loadButton = QtWidgets.QPushButton("Load Data")
        loadButton.clicked.connect(self.selectFile)
        self.layout.addWidget(loadButton, 0, 0, alignment=QtCore.Qt.AlignTop)

        ## Create an empty flowchart with a single input and output
        self.fc = Flowchart(terminals={
            'dataIn': {'io': 'in'},
            'labelIn': {'io': 'in'},
            'dataOut': {'io': 'out'}    
        })
        self.layout.addWidget(self.fc.widget(), 1, 0, -1, 1)
        self.layout.setColumnMinimumWidth(0, 400)

        # Add events on the up-most row
        self.eventPlot = pg.PlotWidget(title="Events")
        self.eventPlot.addLegend()
        self.eventPlot.showGrid(x=True, y=False)
        self.eventPlot.setMouseEnabled(x=True, y=False)
        self.eventPlot.setYRange(0, 1.1)
        self.eventPlot.hideAxis('left')
        self.eventPlot.scene().sigMouseMoved.connect(self.eventPlotMouseMoved)

        self.eventVLine = pg.InfiniteLine(angle=90, movable=False)
        self.eventPlot.addItem(self.eventVLine, ignoreBounds=True)

        # Add region selection to the event plot
        self.region = pg.LinearRegionItem()
        self.region.sigRegionChanged.connect(self.updatePlotRanges)
        # Set color that matches the dark mode
        if self.darkMode:
            self.region.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255, 50)))
        else:
            self.region.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255, 20)))

        self.layout.addWidget(self.eventPlot, 1, 1)

        # Add Channel selection on the second row
        chnContainer = QtWidgets.QWidget()
        if self.darkMode:
            chnContainer.setStyleSheet("QWidget {background-color: #888;}")
        channelSelectionLayout = QtWidgets.QHBoxLayout(chnContainer)
        for i in range(6):
            channelCheckbox = QtWidgets.QCheckBox("Channel {}".format(i+1), checked= (i == 0))
            color = colorGenerator(i, self.darkMode).name()
            channelCheckbox.setStyleSheet('QCheckBox {color: '+color+';}')
            channelCheckbox.stateChanged.connect(self.updateChannels)

            channelSelectionLayout.addWidget(channelCheckbox)
            self.channelCheckboxes.append(channelCheckbox)
        self.layout.addWidget(chnContainer, 0, 1)

        win.show()

    def eventPlotMouseMoved(self, pos):
        if self.eventPlot.sceneBoundingRect().contains(pos):
            mousePoint = self.eventPlot.plotItem.vb.mapSceneToView(pos)
            self.updateVLines(mousePoint)

    def updateVLines(self, pos):
        for widget in self.widgets:
            widget.VLine.setPos(pos.x())
        self.eventVLine.setPos(pos.x())

    def setupFlowChart(self, defaultPath = None):

        library = fclib.LIBRARY.copy() # start with the default node set
        library.addNodeType(Spectrogram, [('Display',)])
        library.addNodeType(MultiLineView, [('Display',)])
        library.addNodeType(EMG_Nodes.NotchFilterNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.ButterBandpassFilterNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.DirectFFTFilterNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.DirectFFTFilterCMSISNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.HysteresisNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.RootMeanSquareNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.MovingAvgConvFilterNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.SquareNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.SquareRootNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.MinTrackerNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.MaxTrackerNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.ChannelJoinNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.minMaxScaleNode, [('EMG_Filter',)])
        library.addNodeType(SaveNode, [('Data',)])
        library.addNodeType(TimeSliceNode, [('Data',)])


        self.fc.setLibrary(library)

        if defaultPath is None:
            self.fc.loadFile("default.fc")
        else:
            self.fc.loadFile(defaultPath)

class MultiLineView(CtrlNode):
    """Node that displays image data in an ImageView widget"""
    nodeName = 'MultiLineView'
    uiTemplate = [
        ('percentile',  'spin', {'value': 99.0, 'step': 0.1, 'bounds': [0.0, 100.0]}),
        ('scaleToVisible',  'check', {'checked': True}),
    ]
    
    
    def __init__(self, name):
        global emg_flowChart

        self.emg_flowChart = emg_flowChart
        # Create a PlotWidget for displaying our data
        self.widget = pg.PlotWidget(title=name)
        self.widget.showGrid(x=True, y=True)
        self.widget.getPlotItem().getAxis('left').setLabel(units='V')
        self.widget.getPlotItem().getAxis('left').setStyle(tickTextWidth = 30, autoExpandTextSpace=False)

        emg_flowChart.layout.addWidget(self.widget, len(emg_flowChart.widgets)+2, 1)
        emg_flowChart.widgets.append(self)

        if len(emg_flowChart.widgets) != 1:
            self.setXLink(emg_flowChart.widgets[0].getWidget())

        self.widget.scene().sigMouseMoved.connect(self.mouseMoved)

        # Add a VLine to the widget to show the current position
        self.VLine = pg.InfiniteLine(angle=90, movable=False)
        self.widget.addItem(self.VLine, ignoreBounds=True)

        ## Initialize node with only a single input terminal
        CtrlNode.__init__(self, name, terminals={'data': {'io':'in'}})

        # config
        self._allowRemove = True
    
    def printValues(self, pos):
        print(pos)

    def close(self):
        emg_flowChart.layout.removeWidget(self.widget)
        # if we are the first widget, then we need to update the x-link
        if self.emg_flowChart.widgets[0] == self and len(self.emg_flowChart.widgets) > 1:
            self.emg_flowChart.widgets[1].setXLink(None)
            for i in range(2, len(self.emg_flowChart.widgets)):
                self.emg_flowChart.widgets[i].setXLink(self.emg_flowChart.widgets[1].getWidget())

        emg_flowChart.widgets.remove(self)
        self.widget.deleteLater()
        return super().close()

    def mouseMoved(self, pos):
        if self.widget.sceneBoundingRect().contains(pos) and self.data is not None:
            mousePoint = self.widget.plotItem.vb.mapSceneToView(pos)
            index = np.searchsorted(self.data.xvals('Time'), mousePoint.x())
            if index > 0 and index < len(self.data.xvals('Time')):
                data = self.data[:,index]
                unit = self.data._info[0]['units']
                dataStrings = [pg.siFormat(data[i], suffix=unit) for i in range(len(data))]
                string = "Time: {:.2f}, ".format(mousePoint.x())
                for i in range(len(dataStrings)):
                    color = self.data._info[0]['colors'][i]
                    string += "<span style='color: {}'>{}</span>, ".format(color, dataStrings[i])
                self.widget.setTitle(string)

            # hide label for other plots
            for widget in emg_flowChart.widgets:
                if widget != self:
                    widget.setTitle(widget.name())

            # Set vertical line position
            emg_flowChart.updateVLines(mousePoint)

    def setTitle(self, title):
        self.widget.setTitle(title)

    def setXLink(self, widget):
        self.widget.setXLink(widget)

    def setXRange(self, *args, **kwargs):
        self.widget.setXRange(*args, **kwargs)

    def getWidget(self):
        return self.widget                

    def process(self, data, display=True):
        ## if process is called with display=False, then the flowchart is being operated
        ## in batch processing mode, so we should skip displaying to improve performance.
        
        if display and self.widget is not None:
            # clear the plot
            self.data = data
            self.widget.clear()
            s = self.stateGroup.state()
            self.widget.addItem(self.VLine, ignoreBounds=True)
            percentile = s['percentile'] / 100.0
            self.widget.enableAutoRange('y', percentile)
            self.widget.setAutoVisible(y=s['scaleToVisible'])

            ## the 'data' argument is the value given to the 'data' terminal
            if data is None:
                self.widget.plot(np.zeros((1))) # give a blank array to clear the view
            else:
                
                signalValues = data[:,:]
                timeValues = data.axisValues('Time')

                for i in range(signalValues.shape[0]):
                    color = data._info[0]['colors'][i]
                    self.widget.plot(timeValues, signalValues[i,:], pen=pg.mkPen(color, width=1))
        else:
            self.data = None


class Spectrogram(CtrlNode):
    """Node that displays the spectogram of the selected channel in a ImageView widget"""
    nodeName = 'Spectrogram'
    uiTemplate = [
        ('channel', 'combo', {'values': ["Channel {}".format(i+1) for i in range(6)], 'value': 'Channel 1'}),
    ]
    
    
    def __init__(self, name):
        global emg_flowChart

        self.emg_flowChart = emg_flowChart

        self.win = pg.GraphicsLayoutWidget()
        # A plot area (ViewBox + axes) for displaying the image
        self.p1 = self.win.addPlot()
        self.p1.setDefaultPadding(0)
        self.p1.hideAxis('bottom')
        self.p1.getAxis('left').setStyle(tickTextWidth = 30, autoExpandTextSpace=False)


        # Item for displaying image data
        self.img = pg.ImageItem()
        self.img.setOpts(axisOrder='row-major')
        self.p1.addItem(self.img)
        # Add a histogram with which to control the gradient of the image
        self.hist = pg.HistogramLUTItem()
        # Link the histogram to the image
        self.hist.setImageItem(self.img)

        self.hist.gradient.restoreState(
            {'mode': 'rgb',
            'ticks': [(0.5, (0, 182, 188, 255)),
                    (1.0, (246, 111, 0, 255)),
                    (0.0, (75, 0, 113, 255))]})
        
        # If you don't add the histogram to the window, it stays invisible, but I find it useful.
        self.win.addItem(self.hist)

        emg_flowChart.layout.addWidget(self.win, len(emg_flowChart.widgets)+2, 1)
        emg_flowChart.widgets.append(self)

        if len(emg_flowChart.widgets) != 1:
            self.setXLink(emg_flowChart.widgets[0].getWidget())

        self.p1.scene().sigMouseMoved.connect(self.mouseMoved)

        # Add a VLine to the widget to show the current position
        self.VLine = pg.InfiniteLine(angle=90, movable=False)
        self.p1.addItem(self.VLine)

        ## Initialize node with only a single input terminal
        CtrlNode.__init__(self, name, terminals={   'timeSeries': {'io':'in'},
                                                    'Sxx': {'io':'in'}})

        self.setTitle(name)

        # config
        self._allowRemove = True

        self.Sxx = None
        self.f = None
        self.t = None
    
    def printValues(self, pos):
        print(pos)

    def setTitle(self, title):
        self.p1.setTitle(title)

    def setXLink(self, widget):
        self.p1.setXLink(widget)

    def setXRange(self, *args, **kwargs):
        self.p1.setXRange(*args, **kwargs)

    def getWidget(self):
        return self.p1    

    def close(self):
        emg_flowChart.layout.removeWidget(self.win)
        # if we are the first widget, then we need to update the x-link
        if self.emg_flowChart.widgets[0] == self and len(self.emg_flowChart.widgets) > 1:
            self.emg_flowChart.widgets[1].setXLink(None)
            for i in range(2, len(self.emg_flowChart.widgets)):
                self.emg_flowChart.widgets[i].setXLink(self.emg_flowChart.widgets[1].getWidget())

        emg_flowChart.widgets.remove(self)
        self.win.deleteLater()
        return super().close()

    def mouseMoved(self, pos):
        if self.p1.sceneBoundingRect().contains(pos) and self.Sxx is not None:
            mousePoint = self.p1.vb.mapSceneToView(pos)
            tIndex = np.searchsorted(self.t, mousePoint.x())
            fIndex = np.searchsorted(self.f, mousePoint.y())
            if tIndex > 0 and tIndex < len(self.t) and fIndex > 0 and fIndex < len(self.f):
                value = self.Sxx[fIndex, tIndex]
                unit = self.unit
                valueString = pg.siFormat(value, suffix=unit)
                string = "Time: {:.2f}s, ".format(mousePoint.x())
                string += "Freq: {:.2f}Hz, ".format(mousePoint.y())
                color = self.color
                string += "<span style='color: {}'>{}</span>".format(color, valueString)
                self.setTitle(string)

            # hide label for other plots
            for widget in emg_flowChart.widgets:
                if widget != self:
                    widget.setTitle(widget.name())

            # Set vertical line position
            emg_flowChart.updateVLines(mousePoint)
                

    def process(self, timeSeries, Sxx, display=True):
        ## if process is called with display=False, then the flowchart is being operated
        ## in batch processing mode, so we should skip displaying to improve performance.

        if display and self.win is not None:
            # calculate the spectrogram
            s = self.stateGroup.state()
            channelName = s['channel']

            if timeSeries is None and Sxx is None:
                self.img.setImage(np.zeros((1,1))) # give a blank array to clear the view
            else:
                if timeSeries is not None:
                    for i, channelInfo in enumerate(timeSeries._info[0]['cols']):
                        if channelInfo['name'] == channelName:
                            channelIndex = i
                    if channelIndex == -1:
                        self.setTitle("Channel {} does not exist".format(channelName))
                        return

                    signalValues = timeSeries[channelIndex,:]
                    timeValues = timeSeries.axisValues('Time')

                    self.unit = timeSeries._info[0]['units']
                    self.color = timeSeries._info[0]['colors'][channelIndex]

                    self.f, self.t, self.Sxx = signal.spectrogram(signalValues, fs=1/(timeValues[1]-timeValues[0]), nperseg=256, noverlap=256*0.75, nfft=512)
                else:
                    channelIndex = -1
                    for i, channelInfo in enumerate(Sxx._info[0]['cols']):
                        if channelInfo['name'] == channelName:
                            channelIndex = i
                    if channelIndex == -1:
                        self.setTitle("Channel {} does not exist".format(channelName))
                        return
                    
                    self.f = Sxx._info[1]['values']
                    self.t = Sxx._info[2]['values']
                    self.Sxx = Sxx[channelIndex].asarray().copy()

                    self.unit = Sxx._info[0]['units']
                    self.color = Sxx._info[0]['colors'][channelIndex]

                self.img.setImage(self.Sxx)

                maxPercentile = np.percentile(self.Sxx, 99)
                self.hist.setHistogramRange(np.min(self.Sxx), maxPercentile)
                self.hist.setLevels(min = np.min(self.Sxx), max = maxPercentile)
                
                #scale the X and Y Axis to time and frequency (standard is pixels)
                tr = QtGui.QTransform()
                tr.scale(self.t[-1]/np.size(self.Sxx, axis=1),
                        (self.f[-1])/np.size(self.Sxx, axis=0))
                self.img.setTransform(tr)

                self.p1.setLimits(yMin=self.f[0], yMax=self.f[-1])
                self.p1.setLabel('left', 'f', units='Hz')

                self.label = pg.TextItem("this is a nice label")
                self.p1.addItem(self.label)


        else:
            self.Sxx = None
            self.f = None
            self.t = None

class SaveNode(CtrlNode):
    """Node for saving data to file"""
    nodeName = "Save"
    
    def __init__(self, name):
        terminals = {
            'In': {'io': 'in'},
        }
        CtrlNode.__init__(self, name, terminals=terminals)

        global emg_flowChart

        self.emg_flowChart = emg_flowChart
    
    def process(self, In, display=True):
        """save data to file"""

        if In is not None:
            customName = self.name().split(".")[1]
            inFilename = self.emg_flowChart.currentFile.split("/")[-1][:-4]
            inPath = self.emg_flowChart.currentFile[:-len(inFilename)-4]
            filename = inPath + inFilename + "/" + customName

            # create directory if it does not exist
            if not os.path.exists(inPath + inFilename):
                os.makedirs(inPath + inFilename)
            np.savez(filename, In = In)
            return
        

class TimeSliceNode(CtrlNode):
    """Node that slices the data according to the region selected in the event plot"""
    nodeName = 'TimeSlice'
    
    def __init__(self, name):
        global emg_flowChart

        self.emg_flowChart = emg_flowChart

        ## Initialize node with only a single input terminal
        CtrlNode.__init__(self, name, terminals=
                        {
                            'In': {'io':'in'},
                            'Out': {'io':'out'}
                        })

        # config
        self._allowRemove = True

    def process(self, In, display=True):
        """slice data"""

        if In is not None:
            #Get region
            region = self.emg_flowChart.region.getRegion()

            infoIn = In.infoCopy()

            timeValues = In.xvals('Time')
            minIndex = np.searchsorted(timeValues, region[0])
            maxIndex = np.searchsorted(timeValues, region[1])
            timeValues = timeValues[minIndex:maxIndex]

            
            infoIn[1]['values'] = timeValues

            Out = MetaArray(In[:,minIndex:maxIndex], info=infoIn)
            
            return {'Out': Out}

if __name__ == '__main__':
    app = pg.mkQApp("Flowchart Custom Node Example")

    darkMode = False

    pg.setConfigOptions(antialias=True)
    if not darkMode:
        pg.setConfigOption('background', 'w')
    ## Create main window with a grid layout inside
    win = QtWidgets.QMainWindow()
    win.setWindowTitle('pyqtgraph example: FlowchartCustomNode')
    win.resize(1000,1000)

    emg_flowChart = EMG_FlowChart(win, darkMode)
    emg_flowChart.setupFlowChart()
    win.show()

    emg_flowChart.loadFromFile("dataSamples/laptopAtDesktopWithoutPower_s=2000_g=1_c=5.npz")

    pg.exec()
