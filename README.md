# FreeThetics-EMGLogViewer

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

