# import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal as signal
from matplotlib.widgets import Button
from EMG_Logger import EMG_Logger
from scipy import fft

#globals
gain = 1
samplePeriod = 50 * 1e-6 # 50 us in seconds
valueArray = np.array([], dtype=np.float32)
timeArray = np.array([], dtype=np.float32)
globalMin = 2**24
globalMax = -2**24
showfft = False


# init communication
emgLogger = EMG_Logger()


def saveValues(event):
    np.savez("data.npz", timeArray = timeArray, valueArray = valueArray)

def resetMaxMin(event):
    global globalMin
    globalMin = 2**24
    global globalMax
    globalMax = -2**24


# figure preparation
fig, ax = plt.subplots(1, 1, figsize = (8*0.9, 6*0.9))
fig.subplots_adjust(bottom=0.2)
axSave = fig.add_axes([0.8, 0.05, 0.1, 0.075])
bsave = Button(axSave, 'Save')
bsave.on_clicked(saveValues)
axPause = fig.add_axes([0.7, 0.05, 0.1, 0.075])
bpause = Button(axPause, 'Pause')
bpause.on_clicked(lambda event: anim.pause())
axResume = fig.add_axes([0.6, 0.05, 0.1, 0.075])
bresume = Button(axResume, 'Resume')
bresume.on_clicked(lambda event: anim.resume())
axReset = fig.add_axes([0.5, 0.05, 0.1, 0.075])
breset = Button(axReset, 'Reset')
breset.on_clicked(resetMaxMin)
pl, = ax.plot([], [])

def animation(i):
    # delete previous frame
    #

    if showfft:
        N = len(valueArray)
        if N == 0:
            return []
        
        ax.cla()
        N = 1024
        # calculate fft
        T = samplePeriod
        n = fft.next_fast_len(N, real=True)
        yf = np.fft.fft(valueArray, n)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
        ax.set_xlim([0, 1000])
        ax.set_ylim([0, 32000])
        return pl,
    else:
        timeToKeep = 1.5
        samplesToKeep = int(timeToKeep / samplePeriod)

        if len(valueArray) > samplesToKeep:
            #downsample for plotting
            downsampleFactor = 1
            downsampledValueArray = valueArray[-samplesToKeep::downsampleFactor]
            downsampledTimeArray = timeArray[-samplesToKeep::downsampleFactor]

            # plot and set axes limits
            global globalMin
            globalMin = min(min(downsampledValueArray), globalMin)
            global globalMax
            globalMax = max(max(downsampledValueArray), globalMax)
            pl.set_data(downsampledTimeArray, downsampledValueArray)
            ax.set_xlim([downsampledTimeArray[0], downsampledTimeArray[-1]])
            ax.set_ylim([globalMin, globalMax])

            return pl,
    return []

# run animation
anim = FuncAnimation(fig, animation, frames = 1000, interval = 250, blit = False)
plt.show(block = False)

iterations = 0
blockSize = 1 * 1024

while True:
    plt.pause(0.1)
    block = emgLogger.readBlock(blockSize)
    if(len(block) == blockSize):
        block = np.frombuffer(block, dtype=np.int32)
        lastTime = 0 if len(timeArray) == 0 else timeArray[-1]

        # read metadata
        gain = block[-1].astype(np.uint16)
        samplePeriod = block[-1].astype(np.uint16) * 1e-6
        block = block[:-1].astype(np.float32) / 2**23 * 3.0
        times = np.linspace(lastTime, lastTime + len(block)*samplePeriod, len(block))
        timeArray = np.append(timeArray, times)
        valueArray = np.append(valueArray, block)