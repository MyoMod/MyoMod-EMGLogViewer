import numpy as np
import scipy as sp
import scipy.signal as signal
from pyqtgraph.metaarray import MetaArray

def applyFilter(data, b, a, padding=0, bidir=False):
    """Apply a linear filter with coefficients a, b. Optionally pad the data before filtering
    and/or run the filter in both directions."""
    try:
        import scipy.signal
    except ImportError:
        raise Exception("applyFilter() requires the package scipy.signal.")
    
    d1 = data.asarray()

    if padding > 0:
        d1 = np.hstack([d1[:padding], d1, d1[-padding:]])
    
    if bidir:
        d1 = scipy.signal.lfilter(b, a, scipy.signal.lfilter(b, a, d1)[::-1])[::-1]
    else:
        d1 = scipy.signal.lfilter(b, a, d1)
    
    if padding > 0:
        d1 = d1[padding:-padding]
        

    if (hasattr(data, 'implements') and data.implements('MetaArray')):
        return MetaArray(d1, info=data.infoCopy())
    else:
        return d1

def notchFilter(data, cutoff, fs=None, quality=30, bidir=False):
    """return data passed through notch filter"""
    
    if fs is None:
        try:
            tvals = data.xvals('Time')
            fs =  (len(tvals)-1) / (tvals[-1]-tvals[0])
        except:
            fs = 1.0
    
    b,a = signal.iirnotch(cutoff, quality, fs=fs)
    
    return applyFilter(data, b, a, bidir=bidir)

def butterBandpassFilter(data, lowcut, highcut, fs=None, order=5, bidir=False):
    """return data passed through butterworth bandpass filter"""

    if fs is None:
        try:
            tvals = data.xvals('Time')
            fs =  (len(tvals)-1) / (tvals[-1]-tvals[0])
        except:
            fs = 1.0

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return applyFilter(data, b, a, bidir=bidir)


def movingAvgConvFilter(data, time, fs = None):
    if fs is None:
        try:
            tvals = data.xvals('Time')
            fs =  (len(tvals)-1) / (tvals[-1]-tvals[0])
        except:
            fs = 1.0

    d1 = data.asarray()
    windowSize = int(time * fs)
    # calculate moving average for each channel
    for i in range(d1.shape[0]):
        d1[i, :] = np.convolve(d1[i, :], np.ones(windowSize) / windowSize, mode='same')
    return MetaArray(d1, info=data.infoCopy())

def rootMeanSquare(data, time, fs = None):
    if fs is None:
        try:
            tvals = data.xvals('Time')
            fs =  (len(tvals)-1) / (tvals[-1]-tvals[0])
        except:
            fs = 1.0

    ySquared = data.asarray()
    ySquared = np.square(ySquared)

    ySquared = MetaArray(ySquared, info=data.infoCopy())
    # calculate moving average
    yRMS = movingAvgConvFilter(ySquared, time, fs)
    yRMS = np.sqrt(yRMS)

    return MetaArray(yRMS, info=data.infoCopy())

def hysteresis(data, upperThreshold, lowerThreshold):
    # apply hysteresis
    d1 = data.asarray()

    # calculate hystersis for each channel

    for channel in range(d1.shape[0]):
        d1[i,0] = d1[i,0] > upperThreshold
        for i in range(1, len(d1)):
            if d1[i,i] > upperThreshold:
                d1[i,i] = 1
            elif d1[i,i] < lowerThreshold:
                d1[i,i] = 0
            else:
                d1[i,i] = d1[i,i - 1]
    return MetaArray(d1, info=data.infoCopy())

def directFFTFilter(data, lowerFreqThreshold, upperFreqThreshold, fftsPerSecond, samplesPerFFT = 256, fftWindow = ('dpss', 1.8), fftSize = 512, fs = None):

    if fs is None:
        try:
            tvals = data.xvals('Time')
            fs =  (len(tvals)-1) / (tvals[-1]-tvals[0])
        except:
            fs = 1.0

    newSamplesPerFFT = int(fs / fftsPerSecond)
    noverlap = samplesPerFFT - newSamplesPerFFT

    directFFT = np.empty((data.shape[0], 0))

    # Apply FFT-Filter to each channel
    for chn in range(data.shape[0]):

        #hamming, bohmann, barthann window for better frequency resolution
        f, t, Sxx = signal.spectrogram(data[chn,:], fs, nfft=fftSize, nperseg=samplesPerFFT, \
                                    noverlap=noverlap, scaling='density', window=fftWindow)

        freqSlice = np.where((f > lowerFreqThreshold) & (f < upperFreqThreshold))
        f = f[freqSlice]
        Sxx = Sxx[freqSlice, :][0]

        # Calculate avg FFT for inactivity
        t_avg = np.where((t > 0.5) & (t < 2.5))
        Sxx_avg = Sxx[:, t_avg][:,0,:]
        inactivityFft = np.mean(Sxx_avg, axis=1, keepdims=True)
        
        # Use inactivity FFT to filter the actual FFT and normalize it to 0 for inactivity
        spectrogrammDirectFtt= (Sxx / inactivityFft) - 1

        if directFFT.shape[1] == 0:
            directFFT = np.empty((data.shape[0], spectrogrammDirectFtt.shape[1]))

        directFFT[chn] = np.mean(spectrogrammDirectFtt, axis=0, keepdims=True)[0]

    info = data.infoCopy()
    info[1]['values'] = t
    return MetaArray(directFFT, info=info)