# import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal as signal
from matplotlib.widgets import Button, MultiCursor, RangeSlider
from matplotlib.ticker import EngFormatter

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

data = np.load(fileName)


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

#remove first x seconds
preSecondsToRemove = 0
prePointsToRemove = int(preSecondsToRemove * fs)

#remove last x seconds
postSecondsToRemove = 0
postPointsToRemove = int(postSecondsToRemove * fs) if postSecondsToRemove > 0 else 1

emgTimes = emgTimes[prePointsToRemove:-postPointsToRemove]
emgValues = emgValues[prePointsToRemove:-postPointsToRemove]

stabilizeSeconds = 1
stabilizePoints = int(stabilizeSeconds * fs)


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

def calculateRMS(emgValues, fs, windowSize = 0.3):
    # calculate rms
    ySquared = emgValues ** 2
    # calculate moving average
    windowSize = int(windowSize * fs)
    yRMS = np.convolve(ySquared, np.ones(windowSize) / windowSize, mode='same')
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

# *** plot data ***

def getAxisRange(value, lowerPercentile = 1, upperPercentile = 99, padding = 0.8):
    lower = np.percentile(value, lowerPercentile)
    upper = np.percentile(value, upperPercentile)
    padIndexOffset = padding * (upper - lower)
    return lower - padIndexOffset, upper + padIndexOffset

def createPlot(fileName):
    fig, axes = plt.subplots(4, 1, figsize = (8*0.9, 10*0.9), sharex=True, num="EMG Signal Analysis")
    fig.subplots_adjust(bottom=0.2, right=0.8)
    fig.suptitle(fileName)
    return fig, axes

def drawTimeAx(axes, emgTimes, emgValues, fs):

    # Create bandpass-filtered version of signal
    yLowpass = butter_bandpass_filter(emgValues, 75, 250,fs, 7)


    # plot and set axes limits
    axes[0].plot(emgTimes, emgValues, marker=".", label = "raw", color = 'tab:gray', )

    # Create notch-filtered version of signal
    yNotched = yLowpass
    for fNotch in [50, 100, 150, 200, 250, 300, 350]:
        yNotched = notchData(yNotched, fs, fNotch)
    axes[1].plot(emgTimes, yNotched, color = 'tab:olive', label = "notch")

    # calculate rms
    nothedRMS = calculateRMS(yNotched, fs, windowSize = 0.3)
    rawRMS = calculateRMS(yLowpass, fs, windowSize = 0.3)

    # use hysteresis to only show activity above a certain threshold
    hysterisis = applyHysteresis(nothedRMS, 4e-6, 2.0e-6)
    nothedRMSHyst = nothedRMS * hysterisis

    axRMSFiltered = axes[1].twinx()
    axRMSFiltered.set_ylabel('RMS [V]')
    axRMSFiltered.plot(emgTimes, nothedRMS, color = 'tab:blue', label = "rms")
    axRMSFiltered.plot(emgTimes, nothedRMSHyst, color = 'tab:red', label = "rms_hysteresis")
    axRMSFiltered.tick_params(axis='y', labelcolor='tab:orange')

    axRMSRaw = axes[0].twinx()
    axRMSRaw.set_ylabel('RMS [V]')
    axRMSRaw.plot(emgTimes, rawRMS, color = 'tab:orange', label = "rms")
    axRMSRaw.tick_params(axis='y', labelcolor='tab:orange')
    

    for ax in axes[:2]:
        ax.set_xlim([emgTimes[0], emgTimes[-1]])
        ax.set_ylabel('Amplitude [V]')
        ax.grid(True)
    axes[0].set_title('EMG - raw')
    axes[1].set_title('EMG - filtered')

    #set Limits
    axes[1].set_ylim(getAxisRange(yNotched[stabilizePoints:], upperPercentile=99.5))
    axes[0].set_ylim(getAxisRange(emgValues[stabilizePoints:], upperPercentile=98, padding=0.5))
    axRMSFiltered.set_ylim(getAxisRange(nothedRMS[stabilizePoints:], upperPercentile=98))
    axRMSRaw.set_ylim(getAxisRange(rawRMS[stabilizePoints:], upperPercentile=98, padding=0.5))

    
    # Label axis as SI units
    SIformatter = EngFormatter(unit='V')
    axes[0].yaxis.set_major_formatter(SIformatter)
    axRMSRaw.yaxis.set_major_formatter(SIformatter)
    axes[1].yaxis.set_major_formatter(SIformatter)
    axRMSFiltered.yaxis.set_major_formatter(SIformatter)

    return yNotched


def drawSpectrogramm(axFreq, emgValues, fs):
    f, t, Sxx = signal.spectrogram(emgValues, fs, nfft=2048, nperseg=2048, noverlap=512, mode='psd')

    freqSlice = np.where((f > 0) & (f < 500))
    f = f[freqSlice]
    Sxx = Sxx[freqSlice, :][0]

    timeDiff = emgTimes[-1] - t[-1]
    t = t + timeDiff
    colorMesh = axFreq.pcolormesh(t, f, Sxx, shading='gouraud', norm="linear")

    axFreq.set_ylabel('Frequency [Hz]')
    axFreq.set_xlabel('Time [sec]')

    fig.colorbar(colorMesh, ax=axFreq, label='Power Spectral Density [V**2/Hz]')

    #set Limits
    ypbot = np.percentile(Sxx, 1)
    yptop = np.percentile(Sxx, 98)
    ypad = 0.5*(yptop - ypbot)

    #colorMesh.set_clim(0, 1e-13)
    colorMesh.set_clim(0, yptop + ypad)

    return f, t, Sxx, colorMesh

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

fig, axes  = createPlot(fileName)



downsampleFactor = 1
emgTimes = emgTimes[::downsampleFactor]
emgValues = emgValues[::downsampleFactor]
fs = fs / downsampleFactor

drawEvents(axes[0], eventTimes, eventValues)

filtererSignal = drawTimeAx(axes[1:3], emgTimes, emgValues, fs)

colorMesh = drawSpectrogramm(axes[-1], filtererSignal, fs)[-1:]

multi = MultiCursor(None, axes, color='darkred', lw=1)

plt.show()
plt.title("EMG tz Signal")
