import pyqtgraph as pg
import numpy as np
from pyqtgraph.flowchart.library.common import CtrlNode

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

        if display and In is not None:
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

        if display and In is not None:
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
        ('lowerFrequencyThreshold', 'spin', {'value': 50., 'step': 0.5, 'dec': False, 'bounds': [0.0, 1000.0], 'suffix': 'Hz', 'siPrefix': True}),
        ('upperFrequencyThreshold', 'spin', {'value': 300., 'step': 0.5, 'dec': False, 'bounds': [0.0, 1000.0], 'suffix': 'Hz', 'siPrefix': True}),
        ('fftWindow', 'combo', {'values': ['dpss', 'hann', 'hamming', 'blackman', 'bartlett'], 'value': 'dpss'}),
        ('fftWindowParam', 'spin', {'value': 1.8, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0]}),
        ('fftSize', 'intSpin', {'value': 512, 'min': 1, 'max': 2048}),
        ('samplesPerFFT', 'intSpin', {'value': 256, 'min': 1, 'max': 2048}),
        ('fftsPerSecond', 'intSpin', {'value': 50, 'min': 1, 'max': 500}),
    ]
    
    def process(self, In, display=True):
        """apply butterworth bandpass filter to data"""

        if display and In is not None:
            s = self.stateGroup.state()
            lowerFreqThreshold = s['lowerFrequencyThreshold']
            upperFreqThreshold = s['upperFrequencyThreshold']
            fftsPerSecond = s['fftsPerSecond']
            samplesPerFFT = s['samplesPerFFT']
            fftWindow = (s['fftWindow'], s['fftWindowParam'])
            fftSize = s['fftSize']
            fs = None
            return {'Out':functions.directFFTFilter(In, lowerFreqThreshold, upperFreqThreshold, fftsPerSecond, samplesPerFFT , fftWindow , fftSize, fs)}
    
class MovingAvgConvFilterNode(CtrlNode):
    """Node for applying moving average convolution filter to data"""
    nodeName = "MovingAvgConvFilter"
    uiTemplate = [
        ('time', 'spin', {'value': 0.1, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0], 'suffix': 's', 'siPrefix': True}),
    ]
    
    def process(self, In, display=True):
        """apply moving average convolution filter to data"""

        if display and In is not None:
            s = self.stateGroup.state()
            time = s['time']
            fs = None
            return {'Out':functions.movingAvgConvFilter(In, time, fs)}
    
class RootMeanSquareNode(CtrlNode):
    """Node for calculating root mean square of data"""
    nodeName = "RootMeanSquare"
    uiTemplate = [
        ('time', 'spin', {'value': 0.1, 'step': 0.1, 'dec': True, 'bounds': [0.0, 1000.0], 'suffix': 's', 'siPrefix': True}),
    ]
    
    def process(self, In, display=True):
        """calculate root mean square of data"""

        if display and In is not None:
            s = self.stateGroup.state()
            time = s['time']
            fs = None
            return {'Out':functions.rootMeanSquare(In, time, fs)}
    
class HysteresisNode(CtrlNode):
    """Node for applying hysteresis to data"""
    nodeName = "Hysteresis"
    uiTemplate = [
        ('upperThreshold', 'spin', {'value': 0.8, 'step': 0.1, 'dec': True, 'bounds': [0.0, None]}),
        ('lowerThreshold', 'spin', {'value': 0.2, 'step': 0.1, 'dec': True, 'bounds': [0.0, None]}),
    ]
    
    def process(self, In, display=True):
        """apply hysteresis to data"""

        if display and In is not None:
            s = self.stateGroup.state()
            upperThreshold = s['upperThreshold']
            lowerThreshold = s['lowerThreshold']
            return {'Out':functions.hysteresis(In, upperThreshold, lowerThreshold)}