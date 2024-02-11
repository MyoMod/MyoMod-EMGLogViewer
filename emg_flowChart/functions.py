import cmsisdsp as dsp
import numpy as np
import scipy as sp
import scipy.signal as signal
from pyqtgraph.metaarray import MetaArray

def applyFilter(data, b, a, padding=0, bidir=False):
    """Apply a linear filter with coefficients a, b. Optionally pad the data before filtering
    and/or run the filter in both directions."""
    try:
        import scipy.signal
    except ImportError:
        raise Exception("applyFilter() requires the package scipy.signal.")
    
    d1 = data.asarray()

    if padding > 0:
        d1 = np.hstack([d1[:padding], d1, d1[-padding:]])
    
    if bidir:
        d1 = scipy.signal.lfilter(b, a, scipy.signal.lfilter(b, a, d1)[::-1])[::-1]
    else:
        d1 = scipy.signal.lfilter(b, a, d1)
    
    if padding > 0:
        d1 = d1[padding:-padding]
        

    if (hasattr(data, 'implements') and data.implements('MetaArray')):
        return MetaArray(d1, info=data.infoCopy())
    else:
        return d1

def notchFilter(data, cutoff, fs=None, quality=30, bidir=False):
    """return data passed through notch filter"""
    
    if fs is None:
        try:
            tvals = data.xvals('Time')
            fs =  (len(tvals)-1) / (tvals[-1]-tvals[0])
        except:
            fs = 1.0
    
    b,a = signal.iirnotch(cutoff, quality, fs=fs)
    
    return applyFilter(data, b, a, bidir=bidir)

def butterBandpassFilter(data, lowcut, highcut, fs=None, order=5, bidir=False):
    """return data passed through butterworth bandpass filter"""

    if fs is None:
        try:
            tvals = data.xvals('Time')
            fs =  (len(tvals)-1) / (tvals[-1]-tvals[0])
        except:
            fs = 1.0

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return applyFilter(data, b, a, bidir=bidir)


def movingAvgConvFilter(data, time, useCMSIS, fs = None):
    if fs is None:
        try:
            tvals = data.xvals('Time')
            fs =  (len(tvals)-1) / (tvals[-1]-tvals[0])
        except:
            fs = 1.0

    d1 = data.asarray().copy()
    windowSize = int(time * fs)
    # calculate moving average for each channel
    for i in range(d1.shape[0]):
        d1[i, :] = np.convolve(d1[i, :], np.ones(windowSize) / windowSize, mode='same')
    return MetaArray(d1, info=data.infoCopy())

def square(data):
    d1 = data.asarray().copy()
    d1 = np.square(d1)
    return MetaArray(d1, info=data.infoCopy())

def squareRoot(data):
    d1 = data.asarray().copy()
    d1 = np.sqrt(d1)
    return MetaArray(d1, info=data.infoCopy())

def rootMeanSquare(data, time, fs = None):
    if fs is None:
        try:
            tvals = data.xvals('Time')
            fs =  (len(tvals)-1) / (tvals[-1]-tvals[0])
        except:
            fs = 1.0

    ySquared = data.asarray()
    ySquared = np.square(ySquared)

    ySquared = MetaArray(ySquared, info=data.infoCopy())
    # calculate moving average
    yRMS = movingAvgConvFilter(ySquared, time, fs)
    yRMS = np.sqrt(yRMS)

    return MetaArray(yRMS, info=data.infoCopy())

def rootMeanSquareCMSIS(data, time, fs = None):
    if fs is None:
        try:
            tvals = data.xvals('Time')
            fs =  (len(tvals)-1) / (tvals[-1]-tvals[0])
        except:
            fs = 1.0

    yRMS = np.zeros(data.shape)
    windowSize = int(time * fs)

    for chn in range(data.shape[0]):
        for i in range(windowSize, data.shape[1]):
            window = data[chn, i-windowSize:i]
            yRMS[chn, i] = dsp.arm_rms_f32(window)

    return MetaArray(yRMS, info=data.infoCopy())

def hysteresis(data, upperThreshold, lowerThreshold):
    # apply hysteresis
    d1 = data.asarray().copy()

    # calculate hystersis for each channel

    for channel in range(d1.shape[0]):
        d1[channel,0] = d1[channel,0] > upperThreshold
        for i in range(1, d1.shape[1]):
            if d1[channel,i] > upperThreshold:
                d1[channel,i] = 1
            elif d1[channel,i] < lowerThreshold:
                d1[channel,i] = 0
            else:
                d1[channel,i] = d1[channel,i - 1]
    return MetaArray(d1, info=data.infoCopy())

def directFFTFilterCMSIS(data, lowerFreqThreshold, upperFreqThreshold, normalizingTime = [0.5,2.5], normalizingAlpha = 0.98, samplesPerCycle = 15, samplesPerFFT = 256, fftWindow = ('dpss', 1.8), fftSize = 256, fs = None, clip = True):
    if fs is None:
        try:
            tvals = data.xvals('Time')
            fs =  (len(tvals)-1) / (tvals[-1]-tvals[0])
        except:
            fs = 1.0

    assert samplesPerFFT == fftSize, "samplesPerFFT and fftSize must be equal for CMSIS-DSP"

    directFFT = np.empty((data.shape[0], data.shape[1] // samplesPerCycle))
    inactivityFft = np.ones((data.shape[0], fftSize//2))
    firstCycle = np.ones(data.shape[0])

    if fftWindow[0] == 'hann':
        window = dsp.arm_hanning_f32(samplesPerFFT)
    else:
        window = dsp.arm_hamming_f32(samplesPerFFT)
    windowedData = np.zeros(samplesPerFFT, dtype=np.float32)

    dataBuffer = np.zeros((data.shape[0], fftSize), dtype=np.float32)
    fft = np.zeros(fftSize, dtype=np.float32)

    Sxx = np.zeros((data.shape[0], fftSize//2, data.shape[1] // samplesPerCycle), dtype=np.float32)
    f = np.linspace(0, fs/2, fftSize//2)
    elementsInBuffer = 0

    # Apply FFT-Filter to each channel
    for chn in range(data.shape[0]):
        for i in range(data.shape[1] // samplesPerCycle):
            # Shift elements of the buffer
            for j in range(samplesPerCycle, samplesPerFFT):
                dataBuffer[chn, j - samplesPerCycle] = dataBuffer[chn, j]

            # Fill buffer with data
            #dataBuffer[chn, [samplesPerFFT - samplesPerCycle, samplesPerFFT]] = data[chn, [i * samplesPerCycle, (i+1) * samplesPerCycle]]
            dataBufferOffset = samplesPerFFT - samplesPerCycle
            dataOffset = i * samplesPerCycle
            for j in range(samplesPerCycle):
                dataBuffer[chn, j + dataBufferOffset] = data[chn, j + dataOffset]

            # Apply windowing
            windowedData = dsp.arm_mult_f32(dataBuffer[chn], window)

            # Calculate FFT
            fftInst = dsp.arm_rfft_fast_instance_f32()
            status = dsp.arm_rfft_fast_init_f32(fftInst, fftSize)
            fft = dsp.arm_rfft_fast_f32(fftInst, windowedData, 0)
            fft = dsp.arm_cmplx_mag_f32(fft)

            # Calculate avg FFT for inactivity
            t_now = data.xvals('Time')[i * samplesPerCycle]
            if normalizingTime[0] < t_now < normalizingTime[1]:
                if firstCycle[chn] == 1:
                    firstCycle[chn] = 0
                    inactivityFft[chn] = fft
                else:
                    inactivityFft[chn] = dsp.arm_scale_f32(inactivityFft[chn], normalizingAlpha) + dsp.arm_scale_f32(fft, 1-normalizingAlpha)
                dsp.arm_clip_f32(inactivityFft[chn], 0.1, 100000)

            # Use inactivity FFT to filter the actual FFT and normalize it to 0 for inactivity
            filteredFft = (fft / inactivityFft[chn]) - 1
            filteredFft = fft

            # bandpass the fft
            fSlice = (f > lowerFreqThreshold) & (f < upperFreqThreshold)
            filteredFft = filteredFft * fSlice

            Sxx[chn, :, i] = filteredFft
            directFFT[chn][i] = dsp.arm_accumulate_f32(filteredFft)
            directFFT[chn][i] = directFFT[chn][i] / (np.sum(fSlice) * 2)
            
    infoIn = data.infoCopy()
    t = np.linspace(data.xvals('Time')[0], data.xvals('Time')[-1], data.shape[1] // samplesPerCycle)
    infoIn[1]['values'] = t

    if clip:
        directFFT = np.clip(directFFT, 0, None)


    SxxMeta = MetaArray(Sxx, info=[infoIn[0], {'name': 'Frequency', 'values': f}, {'name': 'Time', 'values': t}])

    return MetaArray(directFFT, info=infoIn), SxxMeta

def directFFTFilter(data, lowerFreqThreshold, upperFreqThreshold, fftsPerSecond, samplesPerFFT = 256, fftWindow = ('dpss', 1.8), fftSize = 512, fs = None, clip = True):

    if fs is None:
        try:
            tvals = data.xvals('Time')
            fs =  (len(tvals)-1) / (tvals[-1]-tvals[0])
        except:
            fs = 1.0

    newSamplesPerFFT = int(fs / fftsPerSecond)
    noverlap = samplesPerFFT - newSamplesPerFFT

    directFFT = np.empty((data.shape[0], 0))

    SxxReturn = np.zeros((data.shape[0], 0, 0))
    fReturn = np.zeros((0))
    tReturn = np.zeros((0))

    # Apply FFT-Filter to each channel
    for chn in range(data.shape[0]):

        #hamming, bohmann, barthann window for better frequency resolution
        f, t, Sxx = signal.spectrogram(data[chn,:], fs, nfft=fftSize, nperseg=samplesPerFFT, \
                                    noverlap=noverlap, scaling='density', window=fftWindow)
        


        freqSlice = np.where((f > lowerFreqThreshold) & (f < upperFreqThreshold))
        f = f[freqSlice]
        Sxx = Sxx[freqSlice, :][0]

        # Calculate avg FFT for inactivity
        t_avg = np.where((t > 0.5) & (t < 2.5))
        Sxx_avg = Sxx[:, t_avg][:,0,:]
        inactivityFft = np.mean(Sxx_avg, axis=1, keepdims=True)
        
        # Use inactivity FFT to filter the actual FFT and normalize it to 0 for inactivity
        spectrogrammDirectFtt= (Sxx / inactivityFft) - 1
        if SxxReturn.shape[1] == 0:
            SxxReturn = np.zeros((data.shape[0], spectrogrammDirectFtt.shape[0], spectrogrammDirectFtt.shape[1]))
            fReturn = f
            tReturn = t
        SxxReturn[chn] = spectrogrammDirectFtt

        if directFFT.shape[1] == 0:
            directFFT = np.empty((data.shape[0], spectrogrammDirectFtt.shape[1]))

        directFFT[chn] = np.mean(spectrogrammDirectFtt, axis=0, keepdims=True)[0]

        

    infoIn = data.infoCopy()
    infoIn[1]['values'] = t

    if clip:
        directFFT = np.clip(directFFT, 0, None)

    SxxMeta = MetaArray(SxxReturn, info=[infoIn[0], {'name': 'Frequency', 'values': fReturn}, {'name': 'Time', 'values': tReturn}])
    return MetaArray(directFFT, info=infoIn), SxxMeta