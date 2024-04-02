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

def recalculateNormFFT(accu, accuLength, normTemp, reScale, mvcFFT = None ):
    normTemp = accu / accuLength
    #output = dsp.arm_f64_to_float(normTemp)
    output = normTemp.astype(np.float32)
    output = dsp.arm_clip_f32(output, 1e-15, 100000)
    output = 1 / output

    # When mvc is calculated, rescale the FFT
    if reScale:
        mvcResponse = np.mean(mvcFFT * output)
        output = output / mvcResponse
    return output

def directFFTFilterCMSIS(data, fRange, normalizingTime, mvcPeriod, mvcAlpha, samplesPerCycle , samplesPerFFT, fftWindow, fftSize, fs = None, clip = True):
    if fs is None:
        try:
            tvals = data.xvals('Time')
            fs =  (len(tvals)-1) / (tvals[-1]-tvals[0])
        except:
            fs = 1.0

    assert samplesPerFFT == fftSize, "samplesPerFFT and fftSize must be equal for CMSIS-DSP"

    # Misc variables
    fftInst = dsp.arm_rfft_fast_instance_f32()
    status = dsp.arm_rfft_fast_init_f32(fftInst, fftSize)
    nChannels = data.shape[0]

    # subFft Slice calculation
    fBinSize = fs/fftSize
    subFftRange = [int(i // fBinSize) for i in fRange]
    subFftSize = subFftRange[1] - subFftRange[0]

    # FFT buffers
    directFFT = np.zeros((nChannels, data.shape[1] // samplesPerCycle))
    fft = np.zeros(fftSize, dtype=np.float32)
    subFft = np.zeros(subFftSize, dtype=np.float32)
    tempFft = np.zeros(subFftSize, dtype=np.float32)

    # Normalisation data
    normAccumulator = np.zeros((nChannels, subFftSize), dtype=np.float64)
    normTemp = np.zeros(subFftSize, np.float64)
    normFft = np.ones((nChannels, subFftSize), np.float32)
    normAccLength = np.zeros(nChannels, dtype=np.uint32)

    mvcFFT = np.ones((nChannels, subFftSize), np.float32)
    mvcMax = np.zeros(nChannels, np.float32)
    mvcMovAvgFFT = np.zeros((nChannels, subFftSize), np.float32)

    # Memories
    dataMemory = np.zeros((nChannels, fftSize), dtype=np.float32)
    elementsInBuffer = np.zeros(nChannels, dtype=np.uint32)

    # Window
    if fftWindow[0] == 'hann':
        window = dsp.arm_hanning_f32(samplesPerFFT)
    else:
        window = dsp.arm_hamming_f32(samplesPerFFT)
    windowedData = np.zeros(samplesPerFFT, dtype=np.float32)

    # Output variables
    Sxx = np.zeros((nChannels, subFftSize, data.shape[1] // samplesPerCycle), dtype=np.float32)
    f = np.linspace(fRange[0], fRange[1], subFftSize)

    # Apply FFT-Filter to each channel
    for chn in range(nChannels):
        for i in range(data.shape[1] // samplesPerCycle):
            # Shift elements of the buffer
            for j in range(samplesPerCycle, samplesPerFFT):
                dataMemory[chn, j - samplesPerCycle] = dataMemory[chn, j]

            # Fill buffer with data
            dataBufferOffset = samplesPerFFT - samplesPerCycle
            dataOffset = i * samplesPerCycle
            for j in range(samplesPerCycle):
                dataMemory[chn, j + dataBufferOffset] = data[chn, j + dataOffset]
            
            elementsInBuffer[chn] += samplesPerCycle 
            
            if elementsInBuffer[chn] < samplesPerFFT:
                continue

            # Apply windowing
            windowedData = dsp.arm_mult_f32(dataMemory[chn], window)

            # Calculate FFT
            fft = dsp.arm_rfft_fast_f32(fftInst, windowedData, 0)
            subCFtt = fft[subFftRange[0]*2 : subFftRange[1]*2]
            subFft = dsp.arm_cmplx_mag_f32(subCFtt)

            # Calculate avg FFT for inactivity
            t_now = data.xvals('Time')[i * samplesPerCycle]
            t_next = t_now + samplesPerCycle / fs
            if normalizingTime[0] < t_now < normalizingTime[1]:
                #normTemp = dsp.arm_float_to_f64(subFft)
                normTemp = subFft.astype(np.float64)
                normAccumulator[chn] = dsp.arm_add_f64(normTemp, normAccumulator[chn])
                normAccLength[chn] += 1

                normFft[chn] = recalculateNormFFT(normAccumulator[chn], normAccLength[chn], normTemp, t_now > mvcPeriod[1], mvcFFT[chn])

            # Calculate mvcFFT for rescaling
            if mvcPeriod[0] < t_now < mvcPeriod[1]:
                if mvcMovAvgFFT[chn][0] == 0:
                    mvcMovAvgFFT[chn] = subFft
                # Update moving average (For loop in mcu code)
                mvcMovAvgFFT[chn] = mvcMovAvgFFT[chn] * (1-mvcAlpha) + subFft * mvcAlpha

                mvcSum = dsp.arm_dot_prod_f32(normFft[chn], mvcMovAvgFFT[chn])

                if mvcSum > mvcMax[chn]:
                    mvcMax[chn] = mvcSum
                    mvcFFT[chn] = mvcMovAvgFFT[chn]

                if t_next > mvcPeriod[1]:
                    normFft[chn] = recalculateNormFFT(normAccumulator[chn], normAccLength[chn], normTemp, True, mvcFFT[chn])

            # Use inactivity FFT to filter the actual FFT and normalize it to 0 for inactivity
            if t_now > mvcPeriod[1]:
                subFft = (subFft * normFft[chn])
            else:
                subFft = np.zeros(subFftSize, dtype=np.float32)

            Sxx[chn][:, i] = subFft
            directFFT[chn][i] = dsp.arm_mean_f32(subFft)
            if clip:
                directFFT[chn][i] = max(0, directFFT[chn][i])

            
            
    infoIn = data.infoCopy()
    t = np.linspace(data.xvals('Time')[0], data.xvals('Time')[-1], data.shape[1] // samplesPerCycle)
    infoIn[1]['values'] = t

    SxxMeta = MetaArray(Sxx, info=[infoIn[0], {'name': 'Frequency', 'values': f}, {'name': 'Time', 'values': t}])

    return MetaArray(directFFT, info=infoIn), SxxMeta

def directFFTFilter(data, lowerFreqThreshold, upperFreqThreshold, fftsPerSecond, normPeriod, mvcPeriod, mvcAlpha, samplesPerFFT, fftWindow, fftSize, fs = None, clip = True):

    if fs is None:
        try:
            tvals = data.xvals('Time')
            fs =  (len(tvals)-1) / (tvals[-1]-tvals[0])
        except:
            fs = 1.0

    newSamplesPerFFT = fs // fftsPerSecond
    noverlap = samplesPerFFT - newSamplesPerFFT

    directFFT = np.zeros((data.shape[0], 0))

    SxxReturn = np.zeros((data.shape[0], 0, 0))
    fReturn = np.zeros((0))
    tReturn = np.zeros((0))
    mvcFFT = np.zeros((data.shape[0], 0))


    # Apply FFT-Filter to each channel
    for chn in range(data.shape[0]):

        #hamming, bohmann, barthann window for better frequency resolution
        f, t, Sxx = signal.spectrogram(data[chn,:], fs, nfft=fftSize, nperseg=samplesPerFFT, \
                                    noverlap=noverlap, scaling='density', window=fftWindow, mode="magnitude",
                                    detrend=False)

        freqSlice = np.where((f > lowerFreqThreshold) & (f < upperFreqThreshold))
        f = f[freqSlice]
        Sxx = Sxx[freqSlice, :][0]

        # Calculate avg FFT for inactivity
        t_avg = np.where((t > normPeriod[0]) & (t < normPeriod[1]))
        Sxx_avg = Sxx[:, t_avg][:,0,:]
        normFft = np.mean(Sxx_avg, axis=1, keepdims=True)

        # Calculate avg FFT for MVC
        t_mvc = np.where((t > mvcPeriod[0]) & (t < mvcPeriod[1]))
        Sxx_mvc = Sxx[:, t_mvc][:,0,:]

        # Calculate moving average for normalization
        alpha = mvcAlpha
        b, a = [alpha], [1, -(1-alpha)]
        z = signal.lfilter(b, a, Sxx_mvc)

        mvcValues = np.mean(z, axis=0, keepdims=True)[0]
        mvcMaxIndex = np.argmax(mvcValues)

        if mvcFFT.shape[1] == 0:
            mvcFFT = np.zeros((data.shape[0], len(f)))
        mvcFFT[chn] = z[:, mvcMaxIndex]

        # Get the normalization factor
        normFft = 1 / normFft
        normFactor = np.mean(mvcFFT[chn] * normFft)
        normFft = normFft / normFactor

        # Use inactivity FFT to filter the actual FFT
        spectrogrammDirectFtt= Sxx * normFft

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

    mvcFftMeta = MetaArray(mvcFFT, info=[infoIn[0], {'name': 'Time', 'values': fReturn}])
    SxxMeta = MetaArray(SxxReturn, info=[infoIn[0], {'name': 'Frequency', 'values': fReturn}, {'name': 'Time', 'values': tReturn}])
    return MetaArray(directFFT, info=infoIn), SxxMeta, mvcFftMeta

def statisticTracker(data, statistic, timeResolution, memoryLength, samplesPerCycle, startValue = None, fs = None):
    if fs is None:
        try:
            tvals = data.xvals('Time')
            fs =  (len(tvals)-1) / (tvals[-1]-tvals[0])
        except:
            fs = 1.0

    statisticFunctions = {'mean': np.mean, 'std': np.std, 'min': np.min, 'max': np.max}
    statistic = statisticFunctions[statistic]
    resetValue = np.finfo(np.float32).min if statistic == np.max else np.finfo(np.float32).max 

    dataIn = data.asarray()
    binDurationC = int((timeResolution * fs) // samplesPerCycle) # bin duration in cycles
    nBins = int(memoryLength // timeResolution) # number of bins

    bins = np.full((dataIn.shape[0], nBins), np.nan if startValue is None else startValue) # bins for each channel
    sortedBins = np.zeros(nBins)

    output = np.zeros((dataIn.shape[0], dataIn.shape[1])) # output array

    for chn in range(dataIn.shape[0]):
        for cycleCount in range(dataIn.shape[1] // samplesPerCycle):
            # update elements in current bin
            cycleData = dataIn[chn, cycleCount * samplesPerCycle : (cycleCount + 1) * samplesPerCycle]
            bins[chn, 0] = statistic(np.append(cycleData,bins[chn, 0]))

            # update elements in previous bins
            if cycleCount % binDurationC == 0:
                for binIndex in range(nBins-1, 0, -1):
                    bins[chn, binIndex] = bins[chn, binIndex - 1]
                bins[chn, 0] = resetValue

            # Fill output array
            sortedBins = np.sort(bins[chn])
            if len(sortedBins) > 5:
                cycleValue = statistic(sortedBins[2:-3])
            else:
                cycleValue = sortedBins[int(len(sortedBins)/2)]
            output[chn, cycleCount * samplesPerCycle : (cycleCount + 1) * samplesPerCycle] = cycleValue

    infoIn = data.infoCopy()

    return MetaArray(output, info=infoIn)

def concatenateChannels(data):
    t = data[0][0].xvals('Time')
    for inputTuple in data:
        if not np.array_equal(inputTuple[0].xvals('Time'), t):
            raise ValueError("Inputs must have the same time axis.")
        
    # join channel info
    channelInfo = data[0][0].infoCopy()
    if len(data) > 1:
        for inputTuple in data[1:]:
            cols = inputTuple[0].infoCopy()[0]['cols']
            for col in cols:
                col['name'] = "{}: {}".format(inputTuple[1], col['name'])
            channelInfo[0]['cols'] += cols
            channelInfo[0]['colors'] += inputTuple[0].infoCopy()[0]['colors']

    # join data
    d1 = np.vstack([inputTuple[0].asarray() for inputTuple in data])

    return MetaArray(d1, info=channelInfo)

def minMaxScale(data, inMin, inMax, outMin, outMax):
    d1 = data.asarray().copy()

    inMin = np.nan_to_num(inMin, nan=np.finfo(d1.dtype).max, copy=False)
    inMax = np.nan_to_num(inMax, nan=np.finfo(d1.dtype).min, copy=False)
    d1 = ((d1 - inMin) / (inMax - inMin)) * (outMax - outMin) + outMin
            
    np.clip(d1, outMin, outMax, out=d1)
    return MetaArray(d1, info=data.infoCopy())