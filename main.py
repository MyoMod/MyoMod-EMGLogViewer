# import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from highSpeedSerial import ReadLine
import serial
import scipy.signal as signal
from matplotlib.widgets import Button
from EMG_Logger import EMG_Logger

#globals
gain = 1
samplePeriod = 50 * 1e-6 # 50 us in seconds
valueArray = np.array([], dtype=np.float32)
timeArray = np.array([], dtype=np.float32)

# init communication
emgLogger = EMG_Logger()


def saveValues(event):
    np.savez("data.npz", timeArray = timeArray, valueArray = valueArray)


def animation(i):
    # delete previous frame
    ax.cla()

    timeToKeep = 0.25
    samplesToKeep = int(timeToKeep / samplePeriod)

    if len(valueArray) > samplesToKeep:
        #downsample for plotting
        downsampleFactor = 10
        downsampledValueArray = valueArray[-samplesToKeep::downsampleFactor]
        downsampledTimeArray = timeArray[-samplesToKeep::downsampleFactor]

        # plot and set axes limits
        ax.plot(downsampledTimeArray, downsampledValueArray)
        ax.set_xlim([downsampledTimeArray[0], downsampledTimeArray[-1]])
        ax.set_ylim([downsampledValueArray.min(), downsampledValueArray.max()])


# figure preparation
fig, ax = plt.subplots(1, 1, figsize = (8*0.9, 6*0.9))
fig.subplots_adjust(bottom=0.2)
axSave = fig.add_axes([0.8, 0.05, 0.1, 0.075])
bnext = Button(axSave, 'Save')
bnext.on_clicked(saveValues)

# run animation
anim = FuncAnimation(fig, animation, frames = 1000, interval = 50)
plt.show(block = False)

iterations = 0

while True:
    plt.pause(0.1)
    block = emgLogger.readBlock(32*1024)
    if(len(block) == (32*1024)):
        block = np.frombuffer(block, dtype=np.uint32)
        lastTime = 0 if len(timeArray) == 0 else timeArray[-1]

        # read metadata
        gain = block[-1].astype(np.uint16)
        samplePeriod = block[-1].astype(np.uint16) * 1e-6
        block = block[:-1].astype(np.float32) / 4096 * 3.3
        times = np.linspace(lastTime, lastTime + len(block)*samplePeriod, len(block))
        timeArray = np.append(timeArray, times)
        valueArray = np.append(valueArray, block)