import numpy as np
import time

import USBInterface

#data types
Header_t = np.dtype([('magic', 'u4'), ('sampleRate', 'u2'), ('payload', 'u2'), ('gain', 'u1'), ('channels', 'u1'), ('reserved', 'u2')])


class ComHandler:
    def __init__(self, updatesPerSecond = 4) -> None:
        self._payloadLength = None
        self._updatesPerSecond = updatesPerSecond
        self._usbInterface = None
        self._callback = None

        self._time = 0

    def setCallback(self, callback):
        self._callback = callback

    def run(self):
        times , values, _ = self.getSamples()
        self._callback(times, values)

    def initCommunication(self):
        self._usbInterface = USBInterface.USBInterface()
        self._usbInterface.initCommunication()

    def comminucationIsInitialized(self):
        return hasattr(self, "_gain")

    def getSamples(self, timeout = 10):
        # read Header
        headerBlock = self._usbInterface.readBlock(64)
        assert len(headerBlock) == (64/4)
        header = np.frombuffer(headerBlock[:3], dtype=Header_t)
        if not self.isAligned(header):
            headerBlock = self.realign()
            header = np.frombuffer(headerBlock[:3], dtype=Header_t)
        
        # process header to get payload size and metadata
        payloadSize = self.processHeader(header)

        # now read payload with the correct size

        # calculate block size from payload size and decrease by 64 bytes for header
        blockLength = self.payloadToBlocksize(payloadSize) - 64 # blockLength in bytes
        block = self._usbInterface.readBlock(blockLength, timeout * 1000)

        if(len(block) == (blockLength / 4)):
            # read samples
            samples = headerBlock[3:].astype(dtype='i4')
            samples = np.append(samples, block.astype(dtype='i4'))
            values = (((samples<<8)>>8).astype(np.float32)/ 2**23) * 3.

            # Read status
            status = samples >> 24
            status = status & 0xF8 #remove channel number

            samplePeriod = 1.0 / self.sampleRate
            firstSampleTime = self._time + samplePeriod
            lastSampleTime = firstSampleTime + len(values)*samplePeriod
            self._time = lastSampleTime
            times = np.linspace(firstSampleTime, lastSampleTime, len(values))
            return times, values, status
        else:
            return None, None, None

    
    def processHeader(self, header):
        # read metadata
        assert self.isAligned(header)
        sampleRate = int(header["sampleRate"][0])
        payloadSize = int(header["payload"][0])
        gain = int(header["gain"][0])
        channels = int(header["channels"][0])

        self._sampleRate = sampleRate
        self._gain = gain
        self._channels = channels

        self._payloadLength = payloadSize
        return payloadSize

    def payloadToBlocksize(self, payload):
        return payload * 4 + 12

    def isAligned(self, header):
        assert header.dtype == Header_t
        magic = int(header["magic"][0])
        return magic == 0xFFFFFFFF
    
    # read data until we are aligned again, data is discarded
    def realign(self, timeout = 1):
        startTime = time.time()
        dataBlock = self._usbInterface.readBlock(64)
        header = np.frombuffer(dataBlock[:3], dtype=Header_t)
        while not self.isAligned(header):
            dataBlock = self._usbInterface.readBlock(64)
            header = np.frombuffer(dataBlock[:3], dtype=Header_t)

            if timeout > 0:
                if time.time() - startTime > timeout:
                    raise Exception("Timeout while realigning")
        
        return dataBlock

    @property
    def sampleRate(self):
        if not hasattr(self, "_sampleRate"):
            raise Exception("No Data received yet")
        return self._sampleRate
    
    @sampleRate.setter
    def sampleRate(self, sampleRate):
        # Check if initialized
        assert self._usbInterface is not None

        changed = self._sampleRate != sampleRate
        
        if changed:
            # Update samplerate in device
            header = np.zeros(1, dtype=Header_t)
            header["sampleRate"] = int(sampleRate)

            # Set payload size so that we have _updatesPerSecond
            optimalNPerBlock = sampleRate / self._updatesPerSecond
            # make sure that N is in the row 13, 29, 45, 61, ...
            actualNPerBlock = 13 + int(optimalNPerBlock / 16) * 16
            # Max payload size is 8189
            actualNPerBlock = min(actualNPerBlock, 8189)
            header["payload"] = actualNPerBlock

            # buffer size must be a multiple of 64
            blockSize = header["payload"] * 4 + 12
            assert (blockSize) % 64 == 0
            # buffer size must be smaller than BUFFER_SIZE (32k)
            assert blockSize <= 32 * 1024

            self._usbInterface.writeHeader(header.tobytes())

            self._sampleRate = sampleRate

    @property
    def gain(self):
        if not hasattr(self, "_gain"):
            raise Exception("No Data received yet")
        return self._gain
    
    @gain.setter
    def gain(self, gain):
        changed = self._gain != gain
        
        if changed:
            # Update gain in device
            header = np.zeros(1, dtype=Header_t)
            header["gain"] = int(gain)
            self._usbInterface.writeHeader(header.tobytes())
            
            self._gain = gain

    @property
    def channels(self):
        if not hasattr(self, "_channels"):
            raise Exception("No Data received yet")
        return self._channels
    
    @channels.setter
    def channels(self, channels):
        changed = self._channels != channels
        
        if changed:
            # Update channels in device
            header = np.zeros(1, dtype=Header_t)
            header["channels"] = channels
            self._usbInterface.writeHeader(header.tobytes())
            
            self._channels = channels

    def activateChannel(self, channel):
        self.channels = self.channels | (1 << channel)

    def deactivateChannel(self, channel):
        self.channels = self.channels & ~(1 << channel)