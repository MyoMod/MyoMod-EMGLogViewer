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
import flowChart.nodes as EMG_Nodes

class EMG_FlowChart():

    def __init__(self, filename, win):
        self.filename = filename

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
            if "eventTimes" in data:
                eventTimes = data["eventTimes"]
                eventValues = data["eventValues"]
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

        colors = [pg.intColor(i, hues=7, alpha=255).name() for i in range(nChannels)]
        cols = [{"name": "Channel {}".format(i+1), "units": "V"} for i in range(nChannels)]

        # select channels
        channelSelection = self.getChannelSelection()
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

    def setupUi(self, win):
        cw = QtWidgets.QWidget()
        win.setCentralWidget(cw)
        self.layout = QtWidgets.QGridLayout()
        cw.setLayout(self.layout)


        ## Create an empty flowchart with a single input and output
        self.fc = Flowchart(terminals={
            'dataIn': {'io': 'in'},
            'dataOut': {'io': 'out'}    
        })
        w = self.fc.widget()
        self.layout.addWidget(self.fc.widget(), 0, 0, -1, 1)

        # Add events on the up-most row
        self.eventPlot = pg.PlotWidget(title="Events")
        self.eventPlot.addLegend()
        self.eventPlot.showGrid(x=True, y=False)
        self.eventPlot.setMouseEnabled(x=True, y=False)
        self.eventPlot.setYRange(0, 1.1)
        self.eventPlot.hideAxis('left')

        # Add region selection to the event plot
        self.region = pg.LinearRegionItem()
        self.region.sigRegionChanged.connect(self.updatePlotRanges)

        self.layout.addWidget(self.eventPlot, 0, 1)

        # Add Channel selection on the second row
        chnContainer = QtWidgets.QWidget()
        chnContainer.setStyleSheet("QWidget {background-color: #888;}")
        channelSelectionLayout = QtWidgets.QHBoxLayout(chnContainer)
        for i in range(6):
            channelCheckbox = QtWidgets.QCheckBox("Channel {}".format(i+1), checked=True)
            color = pg.intColor(i, hues=7, alpha=255).name()
            channelCheckbox.setStyleSheet('QCheckBox {color: '+color+';}')
            channelCheckbox.stateChanged.connect(self.updateChannels)

            channelSelectionLayout.addWidget(channelCheckbox)
            self.channelCheckboxes.append(channelCheckbox)
        self.layout.addWidget(chnContainer, 1, 1)

        win.show()

    def setupFlowChart(self):

        library = fclib.LIBRARY.copy() # start with the default node set
        library.addNodeType(MultiLineView, [('Display',)])
        library.addNodeType(EMG_Nodes.NotchFilterNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.ButterBandpassFilterNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.DirectFFTFilterNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.HysteresisNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.RootMeanSquareNode, [('EMG_Filter',)])
        library.addNodeType(EMG_Nodes.MovingAvgConvFilterNode, [('EMG_Filter',)])


        self.fc.setLibrary(library)


        ## Now we will programmatically add nodes to define the function of the flowchart.
        ## Normally, the user will do this manually or by loading a pre-generated
        ## flowchart file.

        v1Node = self.fc.createNode('MultiLineView', pos=(0, -150), name="Raw Data")

        v2Node = self.fc.createNode('MultiLineView', pos=(150, -0), name="Notch Filtered Data")

        v3Node = self.fc.createNode('MultiLineView', pos=(150, 150), name="Direct FFT Filtered Data")

        f1Node = self.fc.createNode('NotchFilter', pos=(100, 0))
        f1Node.ctrls['cutoff'].setValue(50)
        f2Node = self.fc.createNode('DirectFFTFilter', pos=(100, 150))


        self.fc.connectTerminals(self.fc['dataIn'], v1Node['data'])
        self.fc.connectTerminals(self.fc['dataIn'], f1Node['In'])
        self.fc.connectTerminals(self.fc['dataIn'], f2Node['In'])
        self.fc.connectTerminals(f1Node['Out'], v2Node['data'])
        self.fc.connectTerminals(f2Node['Out'], v3Node['data'])
        self.fc.connectTerminals(self.fc['dataIn'], self.fc['dataOut'])
        
app = pg.mkQApp("Flowchart Custom Node Example")

## Create main window with a grid layout inside
win = QtWidgets.QMainWindow()
win.setWindowTitle('pyqtgraph example: FlowchartCustomNode')
win.resize(1000,1000)

#enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

class MultiLineView(CtrlNode):
    """Node that displays image data in an ImageView widget"""
    nodeName = 'MultiLineView'
    uiTemplate = [
        ('percentile',  'spin', {'value': 98.0, 'step': 0.5, 'bounds': [0.0, 100.0]}),
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

        emg_flowChart.eventPlot.showGrid(x=True, y=True)

        self.widget.scene().sigMouseMoved.connect(self.mouseMoved)

        ## Initialize node with only a single input terminal
        CtrlNode.__init__(self, name, terminals={'data': {'io':'in'}})
    
    def printValues(self, pos):
        print(pos)

    def mouseMoved(self, pos):
        if self.widget.sceneBoundingRect().contains(pos):
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
                

    def process(self, data, display=True):
        ## if process is called with display=False, then the flowchart is being operated
        ## in batch processing mode, so we should skip displaying to improve performance.
        
        if display and self.widget is not None:
            # clear the plot
            self.data = data
            self.widget.clear()
            s = self.stateGroup.state()


            self.widget.enableAutoRange('y', s['percentile'])
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


filename = "dataSamples/simTest1_s=12800_g=1.npz"
filename = "dataSamples/Liana_test_s=12800_g=1.npz"
emg_flowChart = EMG_FlowChart(filename, win)
emg_flowChart.setupFlowChart()
emg_flowChart.loadFromFile(filename)
win.show()
if __name__ == '__main__':
    pg.exec()
