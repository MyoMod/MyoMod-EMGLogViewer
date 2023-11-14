"""
This example demonstrates writing a custom Node subclass for use with flowcharts.

We implement a couple of simple image processing nodes.
"""

import numpy as np

import pyqtgraph as pg
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart import Flowchart, Node
from pyqtgraph.flowchart.library.common import CtrlNode
from pyqtgraph.Qt import QtWidgets
import numpy as np
from MetaArray import MetaArray

def loadFile(filename):

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
    return emgTimes, emgValues, eventData

app = pg.mkQApp("Flowchart Custom Node Example")

## Create main window with a grid layout inside
win = QtWidgets.QMainWindow()
win.setWindowTitle('pyqtgraph example: FlowchartCustomNode')
win.resize(1000,600)

cw = QtWidgets.QWidget()
win.setCentralWidget(cw)
layout = QtWidgets.QGridLayout()
cw.setLayout(layout)

#enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

## Create an empty flowchart with a single input and output
fc = Flowchart(terminals={
    'dataIn': {'io': 'in'},
    'dataOut': {'io': 'out'}    
})
w = fc.widget()
layout.addWidget(fc.widget(), 0, 0, -1, 1)


## Create two ImageView widgets to display the raw and processed data with contrast
## and color control.


win.show()

## generate random input data
time, values, events = loadFile("dataSamples/simTest1_s=12800_g=1.npz")
#values = values.T
cols = ["Channel {}".format(i+1) for i in range(values.shape[0])]
info = [{"name": "Signal", "units": "mV"},
        {"name": "Time", "units": "sec", "values":time }]
multiChannelData = MetaArray(values, info=info)

## Set the raw data as the input value to the flowchart
fc.setInput(dataIn=multiChannelData)


widgets = {}

## At this point, we need some custom Node classes since those provided in the library
## are not sufficient. Each node will define a set of input/output terminals, a 
## processing function, and optionally a control widget (to be displayed in the 
## flowchart control panel)

class MultiLineView(CtrlNode):
    """Node that displays image data in an ImageView widget"""
    nodeName = 'MultiLineView'
    uiTemplate = [
        ('signalName',  'spin', {'value': 1.0, 'step': 1.0, 'bounds': [0.0, None]}),
    ]
    
    def __init__(self, name):
        global layout, widgets

        # Create a PlotWidget for displaying our data
        self.widget = pg.PlotWidget(title=name)
        layout.addWidget(self.widget, len(widgets)+2, 1)
        widgets[name] = self.widget

        if name != "MultiLineView.0":
            self.widget.setXLink(widgets["MultiLineView.0"])

        ## Initialize node with only a single input terminal
        CtrlNode.__init__(self, name, terminals={'data': {'io':'in'}})
        
    def setWidget(self, widget):  ## setView must be called by the program
        self.widget = widget
        
    def process(self, data, display=True):
        ## if process is called with display=False, then the flowchart is being operated
        ## in batch processing mode, so we should skip displaying to improve performance.
        
        if display and self.widget is not None:
            ## the 'data' argument is the value given to the 'data' terminal
            if data is None:
                self.widget.plot(np.zeros((1))) # give a blank array to clear the view
            else:
                
                signalValues = data[:,:]
                timeValues = data.axisValues('Time')

                for i in range(signalValues.shape[0]):
                    self.widget.plot(timeValues, signalValues[i,:], pen=pg.mkPen(i, width=1))


        
## We will define an unsharp masking filter node as a subclass of CtrlNode.
## CtrlNode is just a convenience class that automatically creates its
## control widget based on a simple data structure.
class FFT_Filter(CtrlNode):
    """Return the input data passed through an unsharp mask."""
    nodeName = "UnsharpMask"
    uiTemplate = [
        ('sigma',  'spin', {'value': 1.0, 'step': 1.0, 'bounds': [0.0, None]}),
        ('strength', 'spin', {'value': 1.0, 'dec': True, 'step': 0.5, 'minStep': 0.01, 'bounds': [0.0, None]}),
    ]
    
    def __init__(self, name):
        ## Define the input / output terminals available on this node
        terminals = {
            'dataIn': dict(io='in'),    # each terminal needs at least a name and
            'dataOut': dict(io='out'),  # to specify whether it is input or output
        }                              # other more advanced options are available
                                       # as well..
        
        CtrlNode.__init__(self, name, terminals=terminals)
        
    def process(self, dataIn, display=True):
        # CtrlNode has created self.ctrls, which is a dict containing {ctrlName: widget}
        sigma = self.ctrls['sigma'].value()
        strength = self.ctrls['strength'].value()
        output = dataIn - (strength * pg.gaussianFilter(dataIn, (sigma,sigma)))
        return {'dataOut': output}
    



## To make our custom node classes available in the flowchart context menu,
## we can either register them with the default node library or make a
## new library.

        
## Method 1: Register to global default library:
#fclib.registerNodeType(ImageViewNode, [('Display',)])
#fclib.registerNodeType(UnsharpMaskNode, [('Image',)])

## Method 2: If we want to make our custom node available only to this flowchart,
## then instead of registering the node type globally, we can create a new 
## NodeLibrary:
library = fclib.LIBRARY.copy() # start with the default node set
library.addNodeType(MultiLineView, [('Display',)])
fc.setLibrary(library)


## Now we will programmatically add nodes to define the function of the flowchart.
## Normally, the user will do this manually or by loading a pre-generated
## flowchart file.

v1Node = fc.createNode('MultiLineView', pos=(0, -150))

v2Node = fc.createNode('MultiLineView', pos=(150, -150))

fc.connectTerminals(fc['dataIn'], v1Node['data'])
fc.connectTerminals(fc['dataIn'], v2Node['data'])
fc.connectTerminals(fc['dataIn'], fc['dataOut'])

if __name__ == '__main__':
    pg.exec()
