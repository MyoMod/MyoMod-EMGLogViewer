# import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal as signal
from matplotlib.widgets import Button, Slider, RadioButtons
from EMG_Logger import EMG_Logger
from scipy import fft
import math

#data types
Header_t = np.dtype([('magic', 'u4'), ('sampleRate', 'u2'), ('payload', 'u2'), ('gain', 'u1'), ('channels', 'u1'), ('reserved', 'u2')])

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

def updateSampleRate(val):
    print("set sample rate to " + str(val))

def updateGain(val):
    print("set gain to " + str(val))

def updateChannel(val):
    channel = int(val[-1]) - 1

    # create header
    header = np.zeros(1, dtype=Header_t)
    header["channels"] = 1 << channel

    print("set channel to " + str(header["channels"]))
    emgLogger.writeHeader(header.tobytes())

# figure preparation
# main figure
fig, ax = plt.subplots(1, 1, figsize = (8*0.9, 6*0.9))
fig.subplots_adjust(bottom=0.25, top=0.9, left=0.15, right=0.8)

# radio buttons
axRadio = fig.add_axes([0.81, 0.75, 0.18, 0.15])
axRadio.set_frame_on(False)
radio = RadioButtons(axRadio, ('Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5', 'Channel 6' ))
radio.on_clicked(updateChannel)

# sliders
axSampleRate = fig.add_axes([0.25, 0.1, 0.65, 0.05])
axGain = fig.add_axes([0.25, 0.15, 0.65, 0.05])

allowedSampleRates = np.array([1.9, 3.9, 7.8, 15.6, 31.2, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000])
sSampleRate = Slider(
    axSampleRate, "SampleRate", 0, 64000,
    valinit=2000, valstep=allowedSampleRates,
    initcolor='red'  # Remove the line marking the valinit position.
)

allowedGains = np.array([1, 2, 4, 8, 16, 32, 64, 128])
sGain = Slider(
    axGain, "Gain", 1, 128,
    valinit=1, valstep=allowedGains,
    initcolor='green'  # Remove the line marking the valinit position.
)

sSampleRate.on_changed(updateSampleRate)
sGain.on_changed(updateGain)

# Buttons
axSave = fig.add_axes([0.8, 0.05, 0.1, 0.05])
axPause = fig.add_axes([0.7, 0.05, 0.1, 0.05])
axResume = fig.add_axes([0.6, 0.05, 0.1, 0.05])
axReset = fig.add_axes([0.5, 0.05, 0.1, 0.05])

bsave = Button(axSave, 'Save')
bsave.on_clicked(saveValues)

bpause = Button(axPause, 'Pause')
bpause.on_clicked(lambda event: anim.pause())

bresume = Button(axResume, 'Resume')
bresume.on_clicked(lambda event: anim.resume())

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
            downsampleFactor = 10
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
blockSize = 32 * 1024

startLastSample = 0

while True:
    plt.pause(0.1)
    block = emgLogger.readBlock(blockSize)
    if(len(block) == 8192):
        lastTime = 0 if len(timeArray) == 0 else timeArray[-1]

        # read metadata
        header = np.frombuffer(block[:3], dtype=Header_t)
        magic = int(header["magic"][0])
        assert magic == 0xFFFFFFFF
        sampleRate = int(header["sampleRate"][0])
        payloadSize = int(header["payload"][0])
        gain = int(header["gain"][0])
        channels = int(header["channels"][0])

        activeChannelText =  math.log2(channels) + 1

        samplePeriod = 1.0 / sampleRate

        # read samples
        samples = block[3:].astype(dtype='i4')
        values = (((samples<<8)>>8).astype(np.float32)/ 2**23) * 3.

        # Read status
        status = samples >> 24
        status = status & 0xF8 #remove channel number

        times = np.linspace(lastTime, lastTime + len(values)*samplePeriod, len(values))
        timeArray = np.append(timeArray, times)
        valueArray = np.append(valueArray, values)