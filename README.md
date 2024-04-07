# MyoMod-EMGLogViewer
The EMGLogViewer consists of several analyzing tools for EMG-Data recorded by the electrode using the MyoMod-EMGLogger. 
It is primarily build ontop of the pyqtgraph flowchart, so that there is a simple interface for parametrization of all components.
Usally the funtion is first implemented in SciPy and later on converted to the CMSIS-python framework. 
Using the methodology, it is possible to do fast prototyping of filters, etc., and then move on to a more realistic implementation once the function is promising.
The CMSIS-python wrapper then allows to implement the function in the same way as it would be on a microcontroller, but inside a python environment, whiich allows a direkt comparsison to the SciPy implementation.
Finally, when the CMSIS and the SciPy version are working in the same way, porting to the actual microcontroller is usually a simple step.


## Install
### Windows

1. Install Python 3 (for example with miniconda)
2. Install vscode
3. Install Zadig (https://zadig.akeo.ie/)
4. Plug in the board via USB, launch Zadig and install the WinUSB driver
5. Create a new virtual environment (For example in vscode-extension "Python Environment Manager" and anaconda) (optional, but recommended)
6. install the python-extensions scipy, matplotlib, tkfilebrowser, pyusb, pywin32 and libusb-package
7. Run main.py for recording and analyzer.py for analyzing

### ubuntu

1. Install Python 3 (for example with miniconda)
2. Install vscode
3. Create a new virtual environment (For example in vscode-extension "Python Environment Manager" and anaconda) (optional, but recommended)
4. install the python-extensions scipy, matplotlib, tkfilebrowser, pyusb
5. Install libusb-1.0-0-dev
6. Add udev-rule "SUBSYSTEMS=="usb", ATTRS{idVendor}=="0525", ATTRS{idProduct}=="a4a0", MODE="0666""
7. Run main.py for recording and analyzer.py for analyzing

