# import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from highSpeedSerial import ReadLine
import serial
import scipy.signal as signal
from matplotlib.widgets import Button

# *** load data ***
data = np.load("data.npz")
timeArray = data["timeArray"]
valueArray = data["valueArray"]
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
    axTime.plot(timeArray, valueArray, label = "raw")
    axTime.set_xlim([timeArray[0], timeArray[-1]])
    axTime.set_ylim([-2, 4])

    # Create notch-filtered version of signal
    yNotched = notchData(valueArray, fs)
    axTime.plot(timeArray, yNotched, color = 'r', label = "notch")

    # Create bandpass-filtered version of signal
    yLowpass = butter_bandpass_filter(yNotched, 10,500,fs)
    axTime.plot(timeArray, yLowpass, color = 'b', label = "bandpass")

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
