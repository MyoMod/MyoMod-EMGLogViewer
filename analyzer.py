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

fileName = "laptopSampled.npz"
#fileName = "laptopCapture2128.npz"
fileName = "laptopCapture1.npz"
#fileName = "pc_capture_sampleHopping.npz"
fileName = "dataSamples/pc_capture4.npz"
fileName = "dataSamples/pcIdleCapture.npz"
fileName = "dataSamples/bizepsCapture2.npz"

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

def calculateRMS(emgValues, fs, windowSize = 0.3):
    # calculate rms
    ySquared = emgValues ** 2
    # calculate moving average
    windowSize = int(windowSize * fs)
    yRMS = np.convolve(ySquared, np.ones(windowSize) / windowSize, mode='same')
    yRMS = np.sqrt(yRMS)

    return yRMS

# *** plot data ***

def createPlot(fileName):
    fig, axes = plt.subplots(4, 1, figsize = (8*0.9, 10*0.9), sharex=True, num="EMG Signal Analysis")
    fig.subplots_adjust(bottom=0.2, right=0.8)
    fig.suptitle(fileName)
    return fig, axes

def drawTimeAx(axes, emgTimes, emgValues, fs):
    # plot and set axes limits
    axes[0].plot(emgTimes, emgValues, label = "raw", color = 'tab:gray')

    # Create bandpass-filtered version of signal
    yLowpass = butter_bandpass_filter(emgValues, 50, 250,fs, 7)

    # Create notch-filtered version of signal
    yNotched = yLowpass
    for fNotch in [50, 100, 150, 200, 250, 300, 350]:
        yNotched = notchData(yNotched, fs, fNotch)
    axes[1].plot(emgTimes, yNotched, color = 'tab:olive', label = "notch")

    # calculate rms
    nothedRMS = calculateRMS(yNotched, fs, windowSize = 0.3)
    rawRMS = calculateRMS(emgValues, fs, windowSize = 0.3)

    axRMSFiltered = axes[1].twinx()
    axRMSFiltered.set_ylabel('RMS [V]')
    axRMSFiltered.plot(emgTimes, nothedRMS, color = 'tab:blue', label = "rms")
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
    ypbot = np.percentile(yNotched, 1)
    yptop = np.percentile(yNotched, 99.5)
    ypad = 0.8*(yptop - ypbot)
    axes[1].set_ylim([ypbot - ypad, yptop + ypad])
    ypbot = np.percentile(nothedRMS, 1)
    yptop = np.percentile(nothedRMS, 98)
    ypad = 0.5*(yptop - ypbot)
    axRMSFiltered.set_ylim([0, yptop + ypad])
    ypbot = np.percentile(emgValues, 1)
    yptop = np.percentile(emgValues, 98)
    ypad = 0.5*(yptop - ypbot)
    axes[0].set_ylim([ypbot - ypad, yptop + ypad])
    ypbot = np.percentile(rawRMS, 1)
    yptop = np.percentile(rawRMS, 98)
    ypad = 0.5*(yptop - ypbot)
    axRMSRaw.set_ylim([0, yptop + ypad])

    
    # Label axis as SI units
    SIformatter = EngFormatter(unit='V')
    axes[0].yaxis.set_major_formatter(SIformatter)
    axRMSRaw.yaxis.set_major_formatter(SIformatter)
    axes[1].yaxis.set_major_formatter(SIformatter)
    axRMSFiltered.yaxis.set_major_formatter(SIformatter)

    #draw legend
    axes[1].legend()
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

fig, axes  = createPlot(fileName)

downsampleFactor = 10
emgTimes = emgTimes[::downsampleFactor]
emgValues = emgValues[::downsampleFactor]
fs = fs / downsampleFactor

if len(eventTimes) > 0:
    drawEvents(axes[0], eventTimes, eventValues)

filtererSignal = drawTimeAx(axes[1:3], emgTimes, emgValues, fs)

colorMesh = drawSpectrogramm(axes[-1], filtererSignal, fs)[-1:]

multi = MultiCursor(None, axes, color='darkred', lw=1)

plt.show()
plt.title("EMG tz Signal")
