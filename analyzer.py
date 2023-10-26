# import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal as signal
from matplotlib.widgets import Button, MultiCursor, RangeSlider

# *** load data ***
emgTimes = np.array([], dtype=np.float32)
emgValues = np.array([], dtype=np.float32)
eventTimes = np.array([], dtype=np.float32)
eventValues = np.array([], dtype=np.float32)

fileName = "laptopSampled.npz"
fileName = "laptopCapture2128.npz"
fileName = "laptopCapture1.npz"
fileName = "pc_capture.npz"

data = np.load(fileName)


if "timeArray" in data:
    emgTimes = data["timeArray"]
    emgValues = data["valueArray"]
else:
    emgTimes = data["emgTimes"]
    emgValues = data["emgValues"]
    eventTimes = data["eventTimes"]
    eventValues = data["eventValues"]

# close file
data.close()

ts = emgTimes[-1] - emgTimes[-2]
fs = 1 / ts

#remove first x seconds
preSecondsToRemove = 5
prePointsToRemove = int(preSecondsToRemove * fs)

#remove last x seconds
postSecondsToRemove = 0
postPointsToRemove = int(postSecondsToRemove * fs) if postSecondsToRemove > 0 else 1

emgTimes = emgTimes[prePointsToRemove:-postPointsToRemove]
emgValues = emgValues[prePointsToRemove:-postPointsToRemove]

stabilizeSeconds = 2
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

# *** plot data ***

def createPlot(fileName):
    fig, (axEvent, axTime, axFreq) = plt.subplots(3, 1, figsize = (8*0.9, 10*0.9), sharex=True, num="EMG Signal Analysis")
    fig.subplots_adjust(bottom=0.2, right=0.8)
    fig.suptitle(fileName)
    return fig, axEvent, axTime, axFreq

def drawTimeAx(axTime, emgTimes, emgValues, fs):
    # plot and set axes limits
    axTime.plot(emgTimes, emgValues, label = "raw", color = 'tab:gray')

    # Create bandpass-filtered version of signal
    yLowpass = butter_bandpass_filter(emgValues, 50, 250,fs, 7)

    # Create notch-filtered version of signal
    yNotched = yLowpass
    for fNotch in [50, 100, 150, 200, 250, 300, 350]:
        yNotched = notchData(yNotched, fs, fNotch)
    axTime.plot(emgTimes, yNotched, color = 'tab:olive', label = "notch")

    # calculate rms
    ySquared = yNotched ** 2
    # calculate moving average
    windowSize = 0.3 # seconds
    windowSize = int(windowSize * fs)
    yRMS = np.convolve(ySquared, np.ones(windowSize) / windowSize, mode='same')
    yRMS = np.sqrt(yRMS)

    axRMS = axTime.twinx()
    axRMS.set_ylabel('RMS [V]')
    axRMS.plot(emgTimes, yRMS, color = 'tab:blue', label = "rms")
    axRMS.tick_params(axis='y', labelcolor='tab:orange')
    

    axTime.set_xlim([emgTimes[0], emgTimes[-1]])
    axTime.set_ylabel('Amplitude [V]')
    axTime.grid(True)
    axTime.set_title('EMG')
    
    #set Limits
    ypbot = np.percentile(yNotched, 1)
    yptop = np.percentile(yNotched, 99.5)
    ypad = 0.8*(yptop - ypbot)
    axTime.set_ylim([ypbot - ypad, yptop + ypad])
    ypbot = np.percentile(yRMS, 1)
    yptop = np.percentile(yRMS, 98)
    ypad = 0.5*(yptop - ypbot)
    axRMS.set_ylim([0, yptop + ypad])


    #draw legend
    axTime.legend()
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
    axEvent.step(eventTimes, eventValues, where='post', label='Muscaular Activity')

    axEvent.set_xlim([eventTimes[0], eventTimes[-1]])
    axEvent.set_ylim([-0.1, 1.1])
    axEvent.set_ylabel('Activity')
    axEvent.grid(True)
    axEvent.set_title('Muscaular Activity [0, 1] provided by GUI')

fig, axEvent, axTime, axFreq  = createPlot(fileName)

downsampleFactor = 1
emgTimes = emgTimes[::downsampleFactor]
emgValues = emgValues[::downsampleFactor]
fs = fs / downsampleFactor

if len(eventTimes) > 0:
    drawEvents(axEvent, eventTimes, eventValues)

filtererSignal = drawTimeAx(axTime, emgTimes, emgValues, fs)

colorMesh = drawSpectrogramm(axFreq, filtererSignal, fs)[-1:]

multi = MultiCursor(None, (axEvent, axTime, axFreq), color='darkred', lw=1)

plt.show()
plt.title("EMG tz Signal")
