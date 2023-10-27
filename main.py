# import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, RadioButtons
from scipy import fft
import math
import tkfilebrowser
import os
import time

import ComHandler
import SignalProcessor
from USBInterface import USBInterface

#Params
showfft = False
timeToKeep = 5

class GUI:
    def __init__(self):

        self.resetXRange()

        self.dataHandler = SignalProcessor.DataHandler()
        self.comHandler = ComHandler.ComHandler()
        self.comHandler.setCallback(self.dataHandler.addData)
        self.signalProcessor = SignalProcessor.SignalProcessor(self.dataHandler)

        self.startTime = 0

        self.lastSampleRate = 0
        self.lastGain = 0
        self.lastChannels = 0

    def start(self):
        self.anim = FuncAnimation(self.fig, self.animation, frames = 1000, interval = 100, blit = False)
        plt.show(block = False)
        self.comHandler.initCommunication()
        self.startTime = time.time()

    def run(self):
        plt.pause(0.1)
        self.comHandler.run()
        self.signalProcessor.run()

    def animation(self, i):
        if not self.comHandler.comminucationIsInitialized():
            return []

        #check for new config
        if self.lastSampleRate != self.comHandler.sampleRate:
            self.samplerateSlider.set_val(self.comHandler.sampleRate)
            self.lastSampleRate = self.comHandler.sampleRate
        
        if self.lastGain != self.comHandler.gain:
            self.gainSlider.set_val(self.comHandler.gain)
            self.lastGain = self.comHandler.gain
        
        if self.lastChannels != self.comHandler.channels:
            self.channelRadio.set_active(int(math.log2(self.comHandler.channels)))
            self.lastChannels = self.comHandler.channels

        if showfft:
            N = len(self.emgValues)
            if N == 0:
                return []
            
            self.ax.cla()
            N = 1024
            # calculate fft
            T = 1/self.comHandler.sampleRate
            n = fft.next_fast_len(N, real=True)
            yf = np.fft.fft(self.emgValues, n)
            xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
            self.ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
            self.ax.set_xlim([0, 1000])
            self.ax.set_ylim([0, 32000])
            return self.plot,
        else:            
            rawTimes, rawValues = self.dataHandler.getData(timeToKeep, "processed")

            if rawTimes is None:
                return []

            rawTimes = rawTimes[:rawValues.shape[0]]

            #downsample for plotting
            downsampleFactor = 10
            downsampledValueArray = rawValues[::downsampleFactor]
            downsampledTimeArray = rawTimes[::downsampleFactor]
            assert len(downsampledValueArray) == len(downsampledTimeArray)

            # plot and set axes limits
            #self.xMin = min(min(downsampledValueArray), self.xMin)
            #self.xMax = max(max(downsampledValueArray), self.xMax)
            self.plot.set_data(downsampledTimeArray, downsampledValueArray)
            self.plot.axes.relim()
            self.plot.axes.autoscale_view()
            #self.ax.set_xlim([oldestTime, max(relTime, timeToKeep)])
            #self.ax.set_ylim([self.xMin, self.xMax])

            return self.plot,

    def saveValues(self, event):
        currentDir = os.getcwd()
        filename = tkfilebrowser.asksaveasfilename(initialdir = currentDir, title = "Select file", filetypes = (("npz files","*.npz"),("all files","*.*")))
        print(filename)
        if filename != "":
            np.savez(filename, emgTimes = self.emgTimes, emgValues = self.emgValues, eventTimes = self.eventTimes, eventValues = self.eventValues)

    def resetXRange(self, event = 0):
        self.xMin = +math.inf
        self.xMax = -math.inf

    def muscleActivated(self, val):
        print("muscle activated")
        self.eventTimes = np.append(self.eventTimes, self.emgTimes[-1])

        if val == "Active":
            self.eventValues = np.append(self.eventValues, 1.0)
        else:
            self.eventValues = np.append(self.eventValues, 0.0)


    def updateSampleRate(self, sampleRate):
        print("set sample rate to " + str(sampleRate))
        self.comHandler.sampleRate = sampleRate

    def updateGain(self, val):
        print("set gain to " + str(val))
        self.comHandler.gain = val

    def updateChannel(self, val):
        channel = int(val[-1]) - 1
        print("set channel to " + str(channel))
        self.comHandler.channels = 1 << channel

    def setupUI(self):
        # figure preparation

        # main figure
        self.fig, self.ax = plt.subplots(1, 1, figsize = (8*0.9, 6*0.9))
        self.fig.subplots_adjust(bottom=0.25, top=0.9, left=0.15, right=0.8)

        # radio buttons
        self.axChannelRadio = self.fig.add_axes([0.81, 0.75, 0.18, 0.15])
        self.axChannelRadio.set_frame_on(False)
        self.channelRadio = RadioButtons(self.axChannelRadio, ('Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5', 'Channel 6' ))
        self.channelRadio.on_clicked(self.updateChannel)

        # radio buttons muscle activation
        self.axMuscleRadio = self.fig.add_axes([0.81, 0.55, 0.18, 0.15])
        self.axMuscleRadio.set_frame_on(False)
        self.muscleRadio = RadioButtons(self.axMuscleRadio, ('Active',  'Inactive'))
        self.muscleRadio.on_clicked(self.muscleActivated)

        # sliders
        self.axSamplerateSlider = self.fig.add_axes([0.25, 0.1, 0.65, 0.05])
        self.axGainSlider = self.fig.add_axes([0.25, 0.15, 0.65, 0.05])

        allowedSamplerates = np.array([1.9, 3.9, 7.8, 15.6, 31.2, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000])
        self.samplerateSlider = Slider(
            self.axSamplerateSlider, "SampleRate", 0, 64000,
            valinit=2000, valstep=allowedSamplerates,
            initcolor='red'  # Remove the line marking the valinit position.
        )

        allowedGains = np.array([1, 2, 4, 8, 16, 32, 64, 128])
        self.gainSlider = Slider(
            self.axGainSlider, "Gain", 1, 128,
            valinit=1, valstep=allowedGains,
            initcolor='green'  # Remove the line marking the valinit position.
        )

        self.samplerateSlider.on_changed(self.updateSampleRate)
        self.gainSlider.on_changed(self.updateGain)

        # Buttons
        self.axSaveBtn = self.fig.add_axes([0.8, 0.05, 0.1, 0.05])
        self.axPauseBtn = self.fig.add_axes([0.7, 0.05, 0.1, 0.05])
        self.axResumeBtn = self.fig.add_axes([0.6, 0.05, 0.1, 0.05])
        self.axResetBtn = self.fig.add_axes([0.5, 0.05, 0.1, 0.05])

        self.saveBtn = Button(self.axSaveBtn, 'Save')
        self.saveBtn.on_clicked(self.saveValues)

        self.pauseBtn = Button(self.axPauseBtn, 'Pause')
        self.pauseBtn.on_clicked(lambda event: self.anim.pause())

        self.resumeBtn = Button(self.axResumeBtn, 'Resume')
        self.resumeBtn.on_clicked(lambda event: self.anim.resume())

        self.resetBtn = Button(self.axResetBtn, 'Reset')
        self.resetBtn.on_clicked(self.resetXRange)

        self.plot, = self.ax.plot([], [])    

def payloadToBlocksize(payload):
    return payload * 4 + 12

if __name__ == "__main__":
    gui = GUI()
    gui.setupUI()
    gui.start()
    while True:
        gui.run()