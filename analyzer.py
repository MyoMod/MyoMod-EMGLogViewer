# import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal as signal
from matplotlib.widgets import Button, MultiCursor, RangeSlider
from matplotlib.ticker import EngFormatter
from matplotlib import colormaps

# *** load data ***
emgTimes = np.array([], dtype=np.float32)
emgValues = np.array([], dtype=np.float32)
eventTimes = np.array([], dtype=np.float32)
eventValues = np.array([], dtype=np.float32)
eventData = None

fileName = "laptopSampled.npz"
#fileName = "laptopCapture2128.npz"
fileName = "laptopCapture1.npz"
#fileName = "pc_capture_sampleHopping.npz"
fileName = "dataSamples/pc_capture4.npz"
fileName = "dataSamples/pcIdleCapture.npz"
fileName = "dataSamples/pc_wade.npz"
fileName = "dataSamples/cli_test.mat.npz"
fileName = "dataSamples/cli_test_s=2000_g=1_c=5.npz"
fileName = "dataSamples/pcWade1_s=2000_g=1_c=5.npz"
fileName = "dataSamples/pcWade1_s=2000_g=1_c=5.npz"
fileName = "dataSamples/laptopAtDesktopWithoutPower_s=2000_g=1_c=5.npz"
fileName = "dataSamples/gainsetting.npz"
fileName = "dataSamples/simTest1_s=2000_g=1_c=5.npz"
#fileName = "dataSamples/pcWade1_s=2000_g=1_c=4.npz"

data = np.load(fileName, allow_pickle=True)


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

# close file
data.close()

ts = emgTimes[-1] - emgTimes[-2]
fs = 1 / ts

stabilizeSeconds = 1
stabilizePoints = int(stabilizeSeconds * fs)

downsampleFactor = 1
emgTimes = emgTimes[::downsampleFactor]
emgValues = emgValues[0,::downsampleFactor]
fs = fs / downsampleFactor

# *** filter data ***
def butter_bandstop(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = signal.butter(order, [low, high], btype='bandstop')
    return b, a

def notchData(emgValues, fs, cutoff = 50):
    # Notch filter
    b, a = signal.iirnotch(cutoff, 30, fs=fs)
    #b, a = butter_bandstop(cutoff - 2, cutoff + 1, fs, order=5)

    # apply notch filter to signal
    y_notched = signal.filtfilt(b, a, emgValues)
    y_notched = signal.filtfilt(b, a, y_notched)

    return y_notched

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def convolve(data, time, fs):
    windowSize = int(time * fs)
    yConv = np.convolve(data, np.ones(windowSize) / windowSize, mode='same')
    return yConv

def calculateRMS(emgValues, fs, windowSize = 0.3):
    # calculate rms
    ySquared = emgValues ** 2
    # calculate moving average
    yRMS = convolve(ySquared, windowSize, fs)
    yRMS = np.sqrt(yRMS)

    return yRMS

def applyHysteresis(values, upperThreshold, lowerThreshold):
    # apply hysteresis
    values = values.copy()
    values[0] = values[0] > upperThreshold
    for i in range(1, len(values)):
        if values[i] > upperThreshold:
            values[i] = 1
        elif values[i] < lowerThreshold:
            values[i] = 0
        else:
            values[i] = values[i - 1]
    return values

def applyFFTFilter(values, fs, lowerThreshold, upperThreshold, fftsPerSecond, fftWindow, fftSize = 512):
    samplesPerFFT = int(fs / fftsPerSecond)
    nperseg = 256
    noverlap = 256 - samplesPerFFT
    #hamming, bohmann, barthann window for better frequency resolution
    f, t, Sxx = signal.spectrogram(emgValues, fs, nfft=fftSize, nperseg=nperseg, \
                                   noverlap=noverlap, scaling='density', window=fftWindow)

    freqSlice = np.where((f > lowerThreshold) & (f < upperThreshold))
    f = f[freqSlice]
    Sxx = Sxx[freqSlice, :][0]

    # Calculate avg FFT for inactivity
    t_avg = np.where((t > 0.5) & (t < 2.5))
    Sxx_avg = Sxx[:, t_avg][:,0,:]
    inactivityFft = np.mean(Sxx_avg, axis=1, keepdims=True)
    
    # Use inactivity FFT to filter the actual FFT and normalize it to 0 for inactivity
    spectrogrammDirectFtt= (Sxx / inactivityFft) - 1

    directFFT = np.mean(spectrogrammDirectFtt, axis=0, keepdims=True)[0]
    return t, directFFT

# *** plot helper functions ***

def getAxisRange(value, lowerPercentile = 1, upperPercentile = 99, padding = 0.8):
    lower = np.percentile(value, lowerPercentile)
    upper = np.percentile(value, upperPercentile)
    padIndexOffset = padding * (upper - lower)
    return lower - padIndexOffset, upper + padIndexOffset

def createPlot(fileName):
    fig, axes = plt.subplots(5, 1, figsize = (8, 10), sharex=True, num="EMG Signal Analysis")
    fig.suptitle(fileName)
    return fig, axes

def drawTimeDomain(axis, t, y, title, secondY = None, unit="V", signalNames = None, limits = None, useSiUnits = True, **kwargs):
    plots = []

    for i,signal in enumerate(y):
        name = signalNames[i] if signalNames is not None else ""

        # plot and set axes limits
        plots.append(axis.plot(t, signal, label = name, color = colormaps['tab10'](i)))


    if secondY is not None:
        secondYAxis = axis.twinx()

        for i,signal in enumerate(secondY):
            name = signalNames[i + len(y)] if signalNames is not None else ""
            
            plots.append(secondYAxis.plot(t, signal, label = name, color = colormaps['tab10'](i + len(y))))
            secondYAxis.tick_params(axis='y', labelcolor='tab:orange')

    axis.set_xlim([t[0], t[-1]])
    axis.set_ylabel('Amplitude [{}]'.format(unit))
    axis.grid(True)
    axis.set_title(title)

    #set Limits
    if limits is not None:
        if hasattr(limits[0], "__len__"):
            axis.set_ylim(limits[0])
            secondYAxis.set_ylim(limits[1])
        else:
            axis.set_ylim(limits)
            secondYAxis.set_ylim(getAxisRange(secondY[0], upperPercentile=99.5, padding=0.05))
        
    else:
        axis.set_ylim(getAxisRange(y[0], upperPercentile=99.5, padding=0.05))

    if useSiUnits:
        # Label axis as SI units
        SIformatter = EngFormatter(unit=unit)
        axis.yaxis.set_major_formatter(SIformatter)
        if secondY is not None:
            secondYAxis.yaxis.set_major_formatter(SIformatter)

    if signalNames is not None:
        axis.legend(ncols = len(signalNames)/2,
                         loc='upper left', borderaxespad=0.)
        if secondY is not None:
            secondYAxis.legend(ncols = len(signalNames)/2,
                         loc='upper right', borderaxespad=0.)

    return axis, plots

# *** draw plots ***

def drawFFTFilter(axis, y, fs):
    fftsPerSecond = 50
    actualWindow = ('dpss', 1.8)

    extT, extFiltered = applyFFTFilter(y, fs, 75, 250, fftsPerSecond, actualWindow, fftSize=512)
    extFiltered = convolve(extFiltered, 0.3, fftsPerSecond)
    extFiltered = np.clip(extFiltered, 0, None)
    extFiltered = np.sqrt(extFiltered)
    extFilteredHys = extFiltered * applyHysteresis(extFiltered, 0.9, 0.1)

    signalNames = ["FFT Filtered", "FFT Filtered with hysteresis"]

    return drawTimeDomain(axis, extT, [extFiltered, extFilteredHys], "FFT Filtered", drawRMS = False, unit="", \
                          useSiUnits = False, signalNames = signalNames)

def drawTimeDomainFilter(axis, t, y, fs):
    # Create bandpass-filtered version of signal
    yLowpass = butter_bandpass_filter(y, 75, 250,fs, 7)

    # Create notch-filtered version of signal
    yNotched = yLowpass
    for fNotch in [50, 100, 150, 200, 250, 300, 350]:
        yNotched = notchData(yNotched, fs, fNotch)

    # calculate rms
    nothedRMS = calculateRMS(yNotched, fs, windowSize = 0.3)

    # use hysteresis to only show activity above a certain threshold
    hysterisis = applyHysteresis(nothedRMS, 4e-6, 2.0e-6)
    nothedRMSHyst = nothedRMS * hysterisis

   
    #set Limits
    allY = [yLowpass[stabilizePoints:], yNotched[stabilizePoints:]]
    limits1 = getAxisRange(allY, upperPercentile=99.5)
    limits1 = limits1[0], limits1[1] * 1.5
    limits2 = getAxisRange(nothedRMS[stabilizePoints:], upperPercentile=98, padding=0.5)
    limits2 = limits2[0], limits2[1] * 1.5
    signalNames = ["bandpass", "notch", "rms", "rms with hysteresis"]
    
    drawTimeDomain(axis, t, [yLowpass, yNotched], "Time-Domain Filtered", secondY=[nothedRMS, nothedRMSHyst], \
                   unit="V", limits = [limits1, limits2], signalNames = signalNames, useSiUnits = True)

    return yNotched

def drawSpectrogramm(axis, y, fs):
    actualWindow = ('dpss', 1.8)
    fftsPerSecond = 50
    minFreq = 30
    maxFreq = 500

    samplesPerFFT = int(fs / fftsPerSecond)
    nperseg = 256
    noverlap = 256 - samplesPerFFT
    f, t, Sxx = signal.spectrogram(y, fs, nfft=512, nperseg=nperseg, noverlap=noverlap, scaling='density', window=actualWindow)

    # Slice out frequencies of interest
    freqSlice = np.where((f > minFreq) & (f < maxFreq))
    f = f[freqSlice]
    Sxx = Sxx[freqSlice, :][0]

    # draw spectrogram
    colorMesh = axis.pcolormesh(t, f, Sxx, shading='gouraud', norm="linear")

    axis.set_ylabel('Frequency [Hz]')
    axis.set_xlabel('Time [sec]')

    fig.colorbar(colorMesh, ax=axis, label='Power Spectral \nDensity [V**2/Hz]')

    #set Limits
    percentile = getAxisRange(Sxx, lowerPercentile=1, upperPercentile=90)
    colorMesh.set_clim(0,percentile[1])

    return colorMesh

def drawEvents(axEvent, eventTimes, eventValues):
    numGroups = 1

    # when eventData is available use it to plot multiple eventgroups
    if eventData is not None:
        numGroups = len(eventData)
        for eventName, eventVals in eventData.items():
            eventVals = eventVals.reshape(-1, 2)
            eventTimes = eventVals[:, 0]
            eventValues = eventVals[:, 1]
            axEvent.step(eventTimes, eventValues, where='post', label=eventName)
    else:
        axEvent.step(eventTimes, eventValues, where='post', label='Muscaular Activity')

    axEvent.set_xlim([0, emgTimes[-1]])
    axEvent.set_ylim([-0.1, 1.7])
    axEvent.set_ylabel('Activity')
    axEvent.grid(True)
    axEvent.legend(ncols = numGroups/2,
                         loc='upper left', borderaxespad=0.)
    axEvent.set_title('Events')

fig, axis  = createPlot(fileName)

drawEvents(axis[0], eventTimes, eventValues)

drawTimeDomain(axis[1], emgTimes, [emgValues], "EMG - raw", unit="V", useSiUnits = True)
drawTimeDomainFilter(axis[2], emgTimes, emgValues, fs)
_, fftPlot = drawFFTFilter(axis[-2], emgValues, fs)

colorMesh = drawSpectrogramm(axis[-1], emgValues, fs)

multi = MultiCursor(None, axis, color='darkred', lw=1)
plt.show()
