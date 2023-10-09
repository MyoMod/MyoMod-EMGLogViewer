import serial
import numpy as np


class ReadLine:
    def __init__(self, serial):
        self.buf = bytearray()
        self.serial = serial
    
    def readBlock(self):
        while True:
            endOfBlockIndex = self.buf.find(b"\xff\xff\xff\xff")

            if endOfBlockIndex >= 0:
                returnBuffer = self.buf[:endOfBlockIndex]
                self.buf = self.buf[endOfBlockIndex+4:]
                return returnBuffer
        
            bytesToRead = max(1, min(4096, self.serial.in_waiting))
            data = self.serial.read(bytesToRead)
            endOfBlockIndex = data.find(b"\xff\xff\xff\xff")
            if endOfBlockIndex >= 0:
                returnBuffer = self.buf + data[:endOfBlockIndex]
                self.buf[0:] = data[endOfBlockIndex+4:]
                return returnBuffer
            else:
                self.buf.extend(data)