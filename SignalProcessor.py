import numpy as np


class DataHandler:
    def __init__(self):
        self.emgTimes = np.array([], dtype=np.float32)
        self.emgValues = np.empty([6,0], dtype=np.float32)
        self.processedValues = np.empty([6,0], dtype=np.float32)
        self.eventTimes = np.array([], dtype=np.float32)
        self.eventValues = np.array([], dtype=np.float32)
        self.labelTimes = np.array([], dtype=np.float32)
        self.labelValues = np.array([], dtype=np.float32)

    def addData(self, times, values, type = "raw"):
        if type == "raw":
            self.emgTimes = np.append(self.emgTimes, times)
            self.emgValues = np.append(self.emgValues, values, axis=1)
        elif type == "processed":
            self.processedValues = np.append(self.processedValues, values, axis=1)
        elif type == "event":
            self.eventTimes = np.append(self.eventTimes, times)
            self.eventValues = np.append(self.eventValues, values)
        elif type == "labelData":
            self.labelTimes = np.append(self.labelTimes, times)
            self.labelValues = values

    def getNumSamples(self, type = "raw"):
        times, values = self._getDataPair(type)
        return values.shape[1]

    def getData(self, seconds, type = "raw"):
        times, values = self._getDataPair(type)
        if len(times) == 0:
            return None, None

        fs = 1 / (times[-1] - times[-2])

        if seconds == 0:
            nSamples = len(times)
        else:
            # find index of last sample
            nSamples = int(seconds * fs)

            # return None if no new samples are available
            if nSamples > len(times):
                nSamples = len(times)

        # get new samples
        times = times[-nSamples:]
        values = values[:,-nSamples:]

        return times, values

    def getDataInRange(self, startTime, endTime, type = "raw"):
        times, values = self._getDataPair(type)
        if len(times) == 0:
            return None, None

        # find index of last sample
        startIndex = np.searchsorted(times, startTime, side='left')
        endIndex = np.searchsorted(times, endTime, side='right')

        # return None if no new samples are available
        if startIndex == len(times):
            return None, None

        # get new samples
        times = times[startIndex:endIndex]
        values = values[:,startIndex:endIndex]

        return times, values
    
    def _getDataPair(self, type = "raw"):
        case = {
            "raw": (self.emgTimes, self.emgValues),
            "processed": (self.emgTimes, self.processedValues),
            "event": (self.eventTimes, self.eventValues),
            "labelData": (self.labelTimes, self.labelValues),
        }
        return case.get(type, (None, None))

class SignalProcessor:
    def __init__(self, dataHandler):
        self.dataHandler = dataHandler
        
    def process(self, values, fs):

        self.dataHandler.addData(None ,values, "processed")

    def run(self):
        # Check if there is unprocessed data
        delta = self.dataHandler.getNumSamples("raw") - self.dataHandler.getNumSamples("processed")
        if delta > 0:
            # get data
            values = self.dataHandler.emgValues[-delta:]
            times = self.dataHandler.emgTimes[:2]
            # process data
            self.process(values, 1/(times[1] - times[0]))
