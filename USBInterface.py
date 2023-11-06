import usb.core
from sys import platform
import numpy as np
import time

if platform == "win32":
    import libusb_package

class USBInterface:
    def __init__(self):
        self.initCommunication()

    def __del__(self):
        usb.util.dispose_resources(self.dev)
        if self.reattach:
            self.dev.attach_kernel_driver(0)

    def readBlock(self, blockSize, timeout = 10000):
        assert blockSize % 64 == 0
        ret = self.dev.read(self.endpoint_in, blockSize, timeout)

        block = np.frombuffer(ret, dtype=np.uint32)

        return block
    
    def writeHeader(self, data, timeout = 10000):
        assert len(data) == 12
        # fill with zeros
        data = data + b'\x00' * (64 - len(data))

        self.dev.write(self.endpoint_out, data, timeout)

    def find_test_device(self):
        if platform == "win32":
            backend = libusb_package.get_libusb1_backend()

        dev = usb.core.find(find_all=False, custom_match=_find_test_device)
        return dev


    def initCommunication(self):
        self.dev = self.find_test_device()
        if not self.dev:
            print('No device found')
            return
        
        self.reattach = False
        if platform == "linux" or platform == "linux2":
            if self.dev.is_kernel_driver_active(0):
                reattach = True
                self.dev.detach_kernel_driver(0)

        self.endpoint_in = self.dev[0][(0,0)][0]
        self.endpoint_out = self.dev[0][(0,0)][1]

    
def _find_test_device(dev):
    if dev.bDeviceClass != 0xff:
        return False

    for config in dev:
        for intf in config:
            if intf.bNumEndpoints not in [2,4]:
                return False

    return True