import usb.core
import numpy as np
import time

class EMG_Logger:
    def __init__(self):
        self.dev, self.endpoint_in, self.endpoint_out, self.reattach = initCommunication()

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
    


def _find_test_device(dev):
    if dev.bDeviceClass != 0xff:
        return False

    for config in dev:
        for intf in config:
            if intf.bNumEndpoints not in [2,4]:
                return False

    return True


def find_test_device():
    return usb.core.find(find_all=False, custom_match=_find_test_device)


def initCommunication():
    dev = find_test_device()
    if not dev:
        print('No device found')
        return
    
    reattach = False
    if dev.is_kernel_driver_active(0):
        reattach = True
        dev.detach_kernel_driver(0)

    endpoint_in = dev[0][(0,0)][0]
    endpoint_out = dev[0][(0,0)][1]

    return dev, endpoint_in, endpoint_out, reattach
  

if __name__ == '__main__':
    emgLogger = EMG_Logger()


    completeSize = 1000000
    paketSize = 32 * 1024
    cycles = (completeSize // paketSize) + 1
    
    timeStart = time.time()

    for i in range(cycles):
        ret = emgLogger.readBlock(paketSize, 100000)
        print(len(ret))
        print(ret)

    timeEnd = time.time()
    deltaTime = timeEnd - timeStart
    print("time elapsed: {:.2f} s".format(deltaTime), end='')

    total = cycles * paketSize
    throughput = total / deltaTime
    if throughput < 1024:
        print(f' - {throughput:.0f} bytes/s')
    elif throughput < 1024 * 1024:
        print(f' - {(throughput / 1024):.1f} kB/s')
    else:
        print(f' - {(throughput / 1024 / 1024):.1f} MB/s')