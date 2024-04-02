import pyqtgraph as pg
import numpy as np
from pyqtgraph.flowchart.library.common import CtrlNode, Node
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import functions

class NotchFilterNode(CtrlNode):
    """Node for applying notch filter to data"""
    nodeName = "NotchFilter"
    uiTemplate = [
        ('cutoff', 'spin', {'value': 50., 'step': 0.5, 'dec': False, 'bounds': [0.0, 1000.0], 'suffix': 'Hz', 'siPrefix': True}),
        ('bidir', 'check', {'checked': False}),
        ('quality', 'spin', {'value': 30.0, 'step': 1.0, 'dec': False, 'bounds': [0.0, 100.0]}),
    ]
    
    def process(self, In, display=True):
        """apply notch filter to data"""

        if In is not None:
            s = self.stateGroup.state()
            cutoff = s['cutoff']
            fs = None
            quality = s['quality']
            bidir = s['bidir']
            return {'Out':functions.notchFilter(In, cutoff, fs, quality, bidir)}
    
class ButterBandpassFilterNode(CtrlNode):
    """Node for applying butterworth bandpass filter to data"""
    nodeName = "ButterBandpassFilter"
    uiTemplate = [
        ('lowcut', 'spin', {'value': 50., 'step': 0.5, 'dec': False, 'bounds': [0.0, 1000.0], 'suffix': 'Hz', 'siPrefix': True}),
        ('highcut', 'spin', {'value': 300., 'step': 0.5, 'dec': False, 'bounds': [0.0, 1000.0], 'suffix': 'Hz', 'siPrefix': True}),
        ('bidir', 'check', {'checked': False}),
        ('order', 'intSpin', {'value': 4, 'min': 1, 'max': 16}),
    ]
    
    def process(self, In, display=True):
        """apply butterworth bandpass filter to data"""

        if In is not None:
            s = self.stateGroup.state()
            lowcut = s['lowcut']
            highcut = s['highcut']
            fs = None
            order = s['order']
            bidir = s['bidir']
            return {'Out':functions.butterBandpassFilter(In, lowcut, highcut, fs, order, bidir)}

class DirectFFTFilterNode(CtrlNode):
    """Node for applying FFT filter to data"""
    nodeName = "DirectFFTFilter"
    uiTemplate = [
        ('f lower', 'spin', {'value': 50., 'step': 0.5, 'dec': False, 'bounds': [0.0, 1000.0], 'suffix': 'Hz', 'siPrefix': True}),
        ('f upper', 'spin', {'value': 300., 'step': 0.5, 'dec': False, 'bounds': [0.0, 1000.0], 'suffix': 'Hz', 'siPrefix': True}),
        ('normStart', 'spin', {'value': 0.5, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0], 'suffix': 's', 'siPrefix': True}),
        ('normEnd', 'spin', {'value': 2.5, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0], 'suffix': 's', 'siPrefix': True}),
        ('mvcStart', 'spin', {'value': 0.5, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0], 'suffix': 's', 'siPrefix': True}),
        ('mvcEnd', 'spin', {'value': 2.5, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0], 'suffix': 's', 'siPrefix': True}),
        ('mvcAlpha', 'spin', {'value': 0.01, 'step': 0.005, 'dec': True, 'bounds': [0.0, 1.0]}),
        ('fftWindow', 'combo', {'values': ['dpss', 'hann', 'hamming', 'blackman', 'bartlett'], 'value': 'dpss'}),
        ('fftWindowParam', 'spin', {'value': 1.8, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0]}),
        ('fftSize', 'intSpin', {'value': 512, 'min': 1, 'max': 2048}),
        ('samplesPerFFT', 'intSpin', {'value': 256, 'min': 1, 'max': 2048}),
        ('fftsPerSecond', 'intSpin', {'value': 50, 'min': 1, 'max': 500}),
        ('clipToZero', 'check', {'checked': True}),
    ]

    def __init__(self, name):
        terminals={
            'In': {'io': 'in'},
            'Sxx': {'io': 'out'}, 
            'Out': {'io': 'out'},
            'mvcFFT': {'io': 'out'} 
        }
        CtrlNode.__init__(self, name, terminals=terminals)
    
    def process(self, In, display=True):
        """apply butterworth bandpass filter to data"""

        if In is not None:
            s = self.stateGroup.state()
            lowerFreqThreshold = s['f lower']
            upperFreqThreshold = s['f upper']
            fftsPerSecond = s['fftsPerSecond']
            samplesPerFFT = s['samplesPerFFT']
            if s['fftWindow'] == 'dpss':
                fftWindow = (s['fftWindow'], int(s['fftWindowParam']))
            else:
                fftWindow = s['fftWindow']
            fftSize = s['fftSize']
            fs = None
            clip = s['clipToZero']
            normPeriod = [s['normStart'], s['normEnd']]
            mvcPeriod = [s['mvcStart'], s['mvcEnd']]
            mvcAlpha = s['mvcAlpha']
            filteredData, SxxMeta, mvcFFT = functions.directFFTFilter(In, lowerFreqThreshold, upperFreqThreshold, fftsPerSecond, 
                                                              normPeriod=normPeriod,
                                                              mvcPeriod=mvcPeriod,
                                                              mvcAlpha=mvcAlpha,
                                                              samplesPerFFT=samplesPerFFT, 
                                                              fftWindow=fftWindow, 
                                                              fftSize=fftSize, 
                                                              fs=fs, 
                                                              clip=clip)
            return {'Out':filteredData, 'Sxx':SxxMeta, 'mvcFFT':mvcFFT}
    
class DirectFFTFilterCMSISNode(CtrlNode):
    """Node for applying FFT filter to data using CMSIS-DSP"""
    nodeName = "DirectFFTFilterCMSIS"
    uiTemplate = [
        ('f lower', 'spin', {'value': 50., 'step': 0.5, 'dec': False, 'bounds': [0.0, 1000.0], 'suffix': 'Hz', 'siPrefix': True}),
        ('f upper', 'spin', {'value': 300., 'step': 0.5, 'dec': False, 'bounds': [0.0, 1000.0], 'suffix': 'Hz', 'siPrefix': True}),
        ('normStart', 'spin', {'value': 0.5, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0], 'suffix': 's', 'siPrefix': True}),
        ('normEnd', 'spin', {'value': 2.5, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0], 'suffix': 's', 'siPrefix': True}),
        ('mvcStart', 'spin', {'value': 0.5, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0], 'suffix': 's', 'siPrefix': True}),
        ('mvcEnd', 'spin', {'value': 2.5, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0], 'suffix': 's', 'siPrefix': True}),
        ('mvcAlpha', 'spin', {'value': 0.01, 'step': 0.005, 'dec': True, 'bounds': [0.0, 1.0]}),
        ('samplesPerCycle', 'intSpin', {'value': 15, 'min': 1, 'max': 100}),
        ('samplesPerFFT', 'intSpin', {'value': 256, 'min': 1, 'max': 2048}),
        ('fftWindow', 'combo', {'values': ['dpss', 'hann', 'hamming', 'blackman', 'bartlett'], 'value': 'dpss'}),
        ('fftWindowParam', 'spin', {'value': 1.8, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0]}),
        ('fftSize', 'intSpin', {'value': 256, 'min': 1, 'max': 2048}),
        ('clipToZero', 'check', {'checked': True}),
    ]
    
    def __init__(self, name):
        terminals={
            'In': {'io': 'in'},
            'Sxx': {'io': 'out'}, 
            'Out': {'io': 'out'}    
        }
        CtrlNode.__init__(self, name, terminals=terminals)

    def process(self, In, display=True):
        """apply butterworth bandpass filter to data"""

        if In is not None:
            s = self.stateGroup.state()
            fRange = [s['f lower'], s['f upper']]
            normalizingTime = [s['normStart'], s['normEnd']]
            mvcPeriod = [s['mvcStart'], s['mvcEnd']]
            mvcAlpha = s['mvcAlpha']
            samplesPerCycle = s['samplesPerCycle']
            samplesPerFFT = s['samplesPerFFT']
            fftWindow = (s['fftWindow'], s['fftWindowParam'])
            fftSize = s['fftSize']
            fs = None
            clip = s['clipToZero']
            filteredData, SxxMeta = functions.directFFTFilterCMSIS(In, 
                                                            fRange=fRange, 
                                                            normalizingTime = normalizingTime, 
                                                            mvcPeriod = mvcPeriod,
                                                            mvcAlpha = mvcAlpha,
                                                            samplesPerCycle = samplesPerCycle, 
                                                            samplesPerFFT=samplesPerFFT, 
                                                            fftWindow=fftWindow, 
                                                            fftSize=fftSize, 
                                                            fs=fs, 
                                                            clip=clip)
            return {'Out':filteredData, 'Sxx':SxxMeta}

class MovingAvgConvFilterNode(CtrlNode):
    """Node for applying moving average convolution filter to data"""
    nodeName = "MovingAvgConvFilter"
    uiTemplate = [
        ('time', 'spin', {'value': 0.1, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0], 'suffix': 's', 'siPrefix': True}),
    ]
    
    def process(self, In, display=True):
        """apply moving average convolution filter to data"""

        if In is not None:
            s = self.stateGroup.state()
            time = s['time']
            fs = None
            if time > 0:
                return {'Out':functions.movingAvgConvFilter(In, time, fs)}
            else:
                return {'Out':In}
    
class RootMeanSquareNode(CtrlNode):
    """Node for calculating root mean square of data"""
    nodeName = "RootMeanSquare"
    uiTemplate = [
        ('time', 'spin', {'value': 0.1, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0], 'suffix': 's', 'siPrefix': True}),
    ]
    
    def process(self, In, display=True):
        """calculate root mean square of data"""

        if In is not None:
            s = self.stateGroup.state()
            time = s['time']
            fs = None
            return {'Out':functions.rootMeanSquare(In, time, fs)}
        
class RootMeanSquareCMSISNode(CtrlNode):
    """Node for calculating root mean square of data using CMSIS-DSP"""
    nodeName = "RootMeanSquare"
    uiTemplate = [
        ('time', 'spin', {'value': 0.1, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0], 'suffix': 's', 'siPrefix': True}),
    ]
    
    def process(self, In, display=True):
        """calculate root mean square of data"""

        if In is not None:
            s = self.stateGroup.state()
            time = s['time']
            fs = None
            return {'Out':functions.rootMeanSquareCMSIS(In, time, fs)}
    
class MaxTrackerNode(CtrlNode):
    """Node for tracking maximum value of data"""
    nodeName = "MaxTracker"
    uiTemplate = [
        ('memoryLength', 'spin', {'value': 15, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0], 'suffix': 's', 'siPrefix': True}),
        ('timeResolution', 'spin', {'value': 1, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0], 'suffix': 's', 'siPrefix': True}),
        ('samplesPerCycle', 'intSpin', {'value': 15, 'min': 1, 'max': 100}),
        ('useStartValue', 'check', {'checked': True}),
        ('startValue', 'spin', {'value': 2.0, 'step': 0.1, 'dec': True, 'bounds': [0.0, None]}),
    ]
    
    def process(self, In, display=True):
        """track maximum value of data"""

        if In is not None:
            s = self.stateGroup.state()
            statistic = 'max'
            timeResolution = s['timeResolution']
            memoryLength = s['memoryLength']
            samplesPerCycle = s['samplesPerCycle']
            startValue = s['startValue'] if s['useStartValue'] else None

            return {'Out':functions.statisticTracker(In, statistic, timeResolution, memoryLength, samplesPerCycle, startValue=startValue, fs = None)}
        
class MinTrackerNode(CtrlNode):
    """Node for tracking minimum value of data"""
    nodeName = "MinTracker"
    uiTemplate = [
        ('memoryLength', 'spin', {'value': 15, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0], 'suffix': 's', 'siPrefix': True}),
        ('timeResolution', 'spin', {'value': 1, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0], 'suffix': 's', 'siPrefix': True}),
        ('samplesPerCycle', 'intSpin', {'value': 15, 'min': 1, 'max': 100}),
        ('useStartValue', 'check', {'checked': False}),
        ('startValue', 'spin', {'value': 0.0, 'step': 0.1, 'dec': True, 'bounds': [0.0, None]}),
    ]
    
    def process(self, In, display=True):
        """track minimum value of data"""

        if In is not None:
            s = self.stateGroup.state()
            statistic = 'min'
            timeResolution = s['timeResolution']
            memoryLength = s['memoryLength']
            samplesPerCycle = s['samplesPerCycle']
            startValue = s['startValue'] if s['useStartValue'] else None

            return {'Out':functions.statisticTracker(In, statistic, timeResolution, memoryLength, samplesPerCycle, startValue=startValue, fs = None)}

class HysteresisNode(CtrlNode):
    """Node for applying hysteresis to data"""
    nodeName = "Hysteresis"
    uiTemplate = [
        ('upperThreshold', 'spin', {'value': 0.8, 'step': 0.1, 'dec': True, 'bounds': [0.0, None]}),
        ('lowerThreshold', 'spin', {'value': 0.2, 'step': 0.1, 'dec': True, 'bounds': [0.0, None]}),
    ]
    
    def __init__(self, name):
        terminals={
            'In': {'io': 'in'},
            'binOut': {'io': 'out'}, 
            'Out': {'io': 'out'}    
        }
        CtrlNode.__init__(self, name, terminals=terminals)


    def process(self, In, display=True):
        """apply hysteresis to data"""

        if In is not None:
            s = self.stateGroup.state()
            upperThreshold = s['upperThreshold']
            lowerThreshold = s['lowerThreshold']
            hystVal = functions.hysteresis(In, upperThreshold, lowerThreshold)
            return {'Out':hystVal * In, 'binOut':hystVal}
        
class SquareNode(CtrlNode):
    """Node for squaring data"""
    nodeName = "Square"
    
    def process(self, In, display=True):
        """square data"""

        if In is not None:
            return {'Out':functions.square(In)}
        
class SquareRootNode(CtrlNode):
    """Node for calculating square root of data"""
    nodeName = "SquareRoot"
    
    def process(self, In, display=True):
        """calculate square root of data"""

        if In is not None:
            return {'Out':functions.squareRoot(In)}
        
class ChannelJoinNode(CtrlNode):
    """Concatenates record arrays and/or adds new columns"""
    nodeName = 'ChannelJoin'
    
    def __init__(self, name):
        CtrlNode.__init__(self, name, terminals = {
            'output': {'io': 'out'},
        })
        
        #self.items = []
        
        self.ui = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout()
        self.ui.setLayout(self.layout)
        
        self.tree = pg.TreeWidget()
        self.addInBtn = QtWidgets.QPushButton('+ Input')
        self.remInBtn = QtWidgets.QPushButton('- Input')
        
        self.layout.addWidget(self.tree, 0, 0, 1, 2)
        self.layout.addWidget(self.addInBtn, 1, 0)
        self.layout.addWidget(self.remInBtn, 1, 1)

        self.addInBtn.clicked.connect(self.addInput)
        self.remInBtn.clicked.connect(self.remInput)
        self.tree.sigItemMoved.connect(self.update)
        
    def ctrlWidget(self):
        return self.ui
        
    def addInput(self):
        #print "ColumnJoinNode.addInput called."
        term = Node.addInput(self, 'input', renamable=True, removable=True, multiable=True)
        #print "Node.addInput returned. term:", term
        item = QtWidgets.QTreeWidgetItem([term.name()])
        item.term = term
        term.joinItem = item
        #self.items.append((term, item))
        self.tree.addTopLevelItem(item)

    def remInput(self):
        sel = self.tree.currentItem()
        term = sel.term
        term.joinItem = None
        sel.term = None
        self.tree.removeTopLevelItem(sel)
        self.removeTerminal(term)
        self.update()

    def process(self, display=True, **args):
        order = self.order()
        vals = []
        for name in order:
            if name not in args:
                continue
            val = args[name]
            if val is not None:
                vals.append([val, name])

        if len(vals) == 0:
            return None
        return {'output': functions.concatenateChannels(vals)}

    def order(self):
        return [str(self.tree.topLevelItem(i).text(0)) for i in range(self.tree.topLevelItemCount())]

    def saveState(self):
        state = Node.saveState(self)
        state['order'] = self.order()
        return state
        
    def restoreState(self, state):
        Node.restoreState(self, state)
        inputs = self.inputs()

        ## Node.restoreState should have created all of the terminals we need
        ## However: to maintain support for some older flowchart files, we need
        ## to manually add any terminals that were not taken care of.
        for name in [n for n in state['order'] if n not in inputs]:
            Node.addInput(self, name, renamable=True, removable=True, multiable=True)
        inputs = self.inputs()

        order = [name for name in state['order'] if name in inputs]
        for name in inputs:
            if name not in order:
                order.append(name)
        
        self.tree.clear()
        for name in order:
            term = self[name]
            item = QtWidgets.QTreeWidgetItem([name])
            item.term = term
            term.joinItem = item
            #self.items.append((term, item))
            self.tree.addTopLevelItem(item)

    def terminalRenamed(self, term, oldName):
        Node.terminalRenamed(self, term, oldName)
        item = term.joinItem
        item.setText(0, term.name())
        self.update()

class minMaxScaleNode(CtrlNode):
    """Node for scaling data to min-max range"""
    nodeName = "MinMaxScale"
    uiTemplate = [
        ('outMin', 'spin', {'value': 0.0, 'step': 0.1, 'dec': True, 'bounds': [0.0, None]}),
        ('outMax', 'spin', {'value': 1.0, 'step': 0.1, 'dec': True, 'bounds': [0.0, None]}),
    ]

    def __init__(self, name):
        terminals = {
            'Out': {'io': 'out'},
            'input': {'io': 'in'},
            'max': {'io': 'in'},
            'min': {'io': 'in'},
        }
        CtrlNode.__init__(self, name, terminals=terminals)
    
    def process(self, input, max, min, display=True):
        """scale data to min-max range"""

        if input is not None:
            s = self.stateGroup.state()
            minVal = s['outMin']
            maxVal = s['outMax']
            return {'Out':functions.minMaxScale(input, min, max, minVal, maxVal)}