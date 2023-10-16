# import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal as signal
from matplotlib.widgets import Button

# *** load data ***
data = np.load("data.npz")
timeArray = data["timeArray"]
cutAmount = int(len(timeArray)*0.4)
timeArray = timeArray[:-cutAmount]
valueArray = data["valueArray"][:-cutAmount]
ts = timeArray[-1] - timeArray[-2]
fs = 1 / ts

# *** filter data ***
def notchData(valueArray, fs, cutoff = 50):
    # Notch filter
    b, a = signal.iirnotch(cutoff, 30, fs=fs)

    # apply notch filter to signal
    y_notched = signal.filtfilt(b, a, valueArray)

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

def createPlot():
    fig, (axTime, axFreq) = plt.subplots(2, 1, figsize = (8*0.9, 10*0.9), sharex=True)
    fig.subplots_adjust(bottom=0.2)
    return fig, axTime, axFreq

def drawTimeAx(axTime, timeArray, valueArray, fs):
    # plot and set axes limits
    #axTime.plot(timeArray, valueArray, label = "raw")

    # Create bandpass-filtered version of signal
    yLowpass = butter_bandpass_filter(valueArray, 20,500,fs)

    # Create notch-filtered version of signal
    yNotched = yLowpass
    for fNotch in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
        yNotched = notchData(yNotched, fs, fNotch)
    axTime.plot(timeArray, yNotched, color = 'r', label = "notch")


    axTime.set_xlim([timeArray[0], timeArray[-1]])
    axTime.set_ylim([min(yNotched) - 5, max(yNotched) + 5])
    #axTime.plot(timeArray, yLowpass, color = 'b', label = "bandpass")

    #draw legend
    axTime.legend()
    return yLowpass


def drawSpectrogramm(axFreq, valueArray, fs):
    f, t, Sxx = signal.spectrogram(valueArray, fs, nfft=1024, nperseg=1024, noverlap=512, mode='psd')

    freqSlice = np.where((f > 0) & (f < 1000))
    f = f[freqSlice]
    Sxx = Sxx[freqSlice, :][0]

    axFreq.pcolormesh(t, f, Sxx, shading='gouraud')

    axFreq.set_ylabel('Frequency [Hz]')

    axFreq.set_xlabel('Time [sec]')

    return f, t, Sxx


fig, axTime, axFreq = createPlot()

filtererSignal = drawTimeAx(axTime, timeArray, valueArray, fs)

drawSpectrogramm(axFreq, filtererSignal, fs)

plt.show()
