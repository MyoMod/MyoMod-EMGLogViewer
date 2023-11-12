# import
import numpy as np
import os
import time
import argparse
import tty
import sys
import termios
import threading

import ComHandler
import SignalProcessor

#Params
showfft = False
timeToKeep = 5

class EventListener(threading.Thread):
    def __init__(self, startTime):
        threading.Thread.__init__(self)
        self.events = {}
        self.startTime = startTime
        self.start()
        self.terminated = False
        self.terminationState = "Ok" # Ok, Reset, Cancel

    def run(self):
        print("Press Enter to exit and save data")
        print("      ESC for exit without saving")
        print("      Tab to restart recording\n") 

        print("Press any alpha-numeric key to add event named after the pressed key\n")

        orig_settings = termios.tcgetattr(sys.stdin)

        # Disable line buffering so we can read keys immediately
        tty.setcbreak(sys.stdin)
        while not self.terminated:
            x=sys.stdin.read(1)[0]
            
            if x == '\n': # Enter -> exit and save
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)   
                self.terminated = True

            elif ord(x) == 0x09: # Tab -> restart recording
                self.terminationState = "Reset"
                self.terminated = True

            elif x == '\x1b': # ESC -> exit without saving
                self.terminationState = "Cancel"
                self.terminated = True

            # only add alpha-numeric keys as events
            elif 0x21 <= ord(x) <= 0x7e:
                self.addEvent(x)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)

    def getEventArrays(self):
        eventDict = {}
        endTime = time.time() - self.startTime

        for eventName, eventTimes in self.events.items():
            eventArray = np.array([[]],ndmin=2)

            # Events allways starts with 0 at time 0
            eventArray = np.append(eventArray, [0,0])

            eventState = False
            for eventTime in eventTimes:
                eventState = not eventState
                eventArray = np.append(eventArray, [eventTime, eventState])

            # Events allways ends with 0 at endTime
            eventArray = np.append(eventArray, [endTime, 0])

            eventDict[eventName] = eventArray
        return eventDict

    def addEvent(self, event):
        # get time
        relTime = time.time() - self.startTime
        # add event to group
        if not event in self.events:
            self.events[event] = []
        self.events[event].append(relTime)
        enabledEvent = len(self.events[event]) % 2 == 1
        print("event " + event + " " + ("enabled" if enabledEvent else "disabled"))

class CLI_Handler:
    def __init__(self, filename, channel, gain):

        self.dataHandler = SignalProcessor.DataHandler()
        self.comHandler = ComHandler.ComHandler()
        self.comHandler.setCallback(self.dataHandler.addData)
        self.signalProcessor = SignalProcessor.SignalProcessor(self.dataHandler)

        self.samplerate = 2000
        self.gain = gain
        self.channel = channel
        self.filename = filename

    def start(self):
        self.comHandler.initCommunication()

        print("\n\n*** FreeThetics Data Logger ***\n")
        print("File will be saved to " + self.filename)
        print()

        # configure adc
        while not self.comHandler.comminucationIsInitialized():
            time.sleep(0.1)
            self.run()

        while not self.comHandler.validateConfig(self.samplerate, self.gain, 1 << (self.channel-1)):
            time.sleep(0.1)
            self.comHandler.sampleRate = self.samplerate
            self.comHandler.gain = self.gain
            self.comHandler.channels = 1 << (self.channel-1)
            self.run()

        # print current config
        print("Sampling rate: {}Hz".format(self.comHandler.sampleRate), end=' | ')
        print("Gain: {}".format(self.comHandler.gain), end=' | ')
        print("Channels: {0:06b}\n\n".format(self.comHandler.channels))

        eventListener = EventListener(time.time())
        while not eventListener.terminated:
<<<<<<< HEAD
            if self.useGui:
                plt.pause(0.1)
=======
            time.sleep(0.1)
>>>>>>> parent of cdfe416 (Add window with measured Data to cli)
            self.run()  
        
        # get events
        self.events = eventListener.getEventArrays()

        self.terminate(eventListener.terminationState)

    def run(self):
        self.comHandler.run()
        self.signalProcessor.run()

    def terminate(self, terminationState):
        if terminationState == "Ok":
            print("terminate and save data")
            self.saveValues()
            exit(0)

        elif terminationState == "Reset":
            print("reset")

        elif terminationState == "Cancel":
            print("terminate without saving")
            exit(0)

    def saveValues(self):
        currentDir = os.getcwd()
        emgTimes, emgValues = self.dataHandler.getData(0, "raw")
        filename = os.path.join(currentDir, self.filename)
        np.savez(filename, emgTimes = emgTimes, emgValues = emgValues, **self.events)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FreeThetics Data Logger', epilog='Terminate with Ctrl+C saves data to file')
    parser.add_argument('filename', help='filename to save data to')
    parser.add_argument('-c', '--channel', choices=range(2**6), help='channel to log', type=int)
    parser.add_argument('-g', '--gain', choices=[2**g for g in range(8)], help='gain to use', type=int)
    parser.add_argument('-s', '--samplerate', help='sample rate to use', type=int, choices=[1.9, 3.9, 7.8, 15.6, 31.2, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000], default=2000)
    parser.add_argument('-a', '--auto', help='automaticly add config to filename', action='store_true', default=False)
    args = parser.parse_args()

    filename = args.filename.strip()

    if args.auto:
        filename = filename + "_s=" + str(args.samplerate) + "_g=" + str(args.gain) + "_c=" + str(args.channel)

    while True:
        cliHandler = CLI_Handler(filename, args.channel, args.gain)
        cliHandler.start()
