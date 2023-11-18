"""
This example demonstrates writing a custom Node subclass for use with flowcharts.

We implement a couple of simple image processing nodes.
"""

import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart import Flowchart, Node
from pyqtgraph.flowchart.library.common import CtrlNode
import numpy as np
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

    def __init__(self, filename, win, darkMode=False):
        self.filename = filename
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

        self.timeRange = None

        self.setupUi(win)

    def loadFromFile(self, filename):
        data = np.load(filename, allow_pickle=True)

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

            eventData = dict( data)
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

        # Add events on the up-most row
        # add events
        self.eventPlot.clear()
        for i, eventGroup in enumerate(self.eventData.items()):
            eventName = eventGroup[0]
            eventData = eventGroup[1].reshape(-1,2)
            eventTimes = eventData[:,0]
            eventValues = eventData[:,1]

            self.eventPlot.plot(eventTimes, eventValues, stepMode='right', name=eventName, pen=pg.mkPen(i, width=2))
        
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

        ## Set the raw data as the input value to the flowchart
        self.fc.setInput(dataIn=multiChannelData)

    def updateChannels(self):
        self.updateInput()

    def updatePlotRanges(self):
        timeRange = self.region.getRegion()
        [widget.widget.setXRange(*timeRange, padding = 0) for widget in self.widgets]
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
            channelCheckbox = QtWidgets.QCheckBox("Channel {}".format(i+1), checked=True)
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
        library.addNodeType(MultiLineView, [('Display',)])
        library.addNodeType(EMG_Nodes.NotchFilterNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.ButterBandpassFilterNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.DirectFFTFilterNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.HysteresisNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.RootMeanSquareNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.MovingAvgConvFilterNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.SquareNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.SquareRootNode, [('EMG_Filter',)])


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

        emg_flowChart.layout.addWidget(self.widget, len(emg_flowChart.widgets)+2, 1)
        emg_flowChart.widgets.append(self)

        if len(emg_flowChart.widgets) != 1:
            self.widget.setXLink(emg_flowChart.widgets[0].widget)

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
            self.emg_flowChart.widgets[1].widget.setXLink(None)
            for i in range(2, len(self.emg_flowChart.widgets)):
                self.emg_flowChart.widgets[i].widget.setXLink(self.emg_flowChart.widgets[1].widget)

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
                    widget.widget.setTitle(widget.name())

            # Set vertical line position
            emg_flowChart.updateVLines(mousePoint)
                

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

    filename = "dataSamples/Liana_test_s=12800_g=1.npz"
    emg_flowChart = EMG_FlowChart(filename, win, darkMode)
    emg_flowChart.setupFlowChart()
    emg_flowChart.loadFromFile(filename)
    win.show()

    pg.exec()
