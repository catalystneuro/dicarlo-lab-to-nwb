# Notes concerning the conversion conversion




## Notes about session structure

These are the files that were shared with us:

```python
├── exp_1_shapes
├── exp_Co3D   # Has raw files for Intan, and videos as stimuli
├── exp_domain-transfer-2023  # Has raw files for Intan, images as stimuli
├── exp_IAPS
├── exp_IAPS100
├── exp_Kar_2023_degraded
├── norm_FOSS    # Recordings to normalize session
└── StimulusSets
```

We have sessions for

The experiment 'exp_domain-transfer-2023' has raw files for Intan, it has images as stimuli.
the experiemnt 'exp_Co3D' has raw files and videos as stimuli.

the experiment 'norm_FOSS' does not follow the exp format. Why?
Thes are images that are used to normalize stuff


## Probe information
For this instance each array 96 channels, 400 micrometes apart



## Spike thresholding / Peak detection pipeline

The code for spikethresholding is here:

I think is the function `get_spike_times` located here:

https://github.com/AliyaAbl/DiCarlo_NWB/blob/4b8638635864432fc20ddc5c3738f898a7fb510f/spike-tools-chong/spike_tools/utils/spikeutils.py#L17-L52

The pipeline is composed of the following steps:
* Notch filter at 60 Hz and bandwith  (own implementation)
* Bandpass filter (implemented with scipy.signal)
* Noise estimation (per chunk) by using the MAD estimator for the standard deviation of the signal.


Does not take into account double counting (`sweep_ms` in spikeinterface) and this is done at a per-channel basis (`by_channel` in spikeinterface).

The two functions as of 2024-04-06 are:


### Bandpass filter
```python
def bandpass_filter(signal, f_sampling, f_low, f_high):
    wl = f_low / (f_sampling / 2.)
    wh = f_high / (f_sampling / 2.)
    wn = [wl, wh]

    # Designs a 2nd-order Elliptic band-pass filter which passes
    # frequencies between normalized f_low and f_high, and with 0.1 dB of ripple
    # in the passband, and 40 dB of attenuation in the stopband.
    b, a = ellip(2, 0.1, 40, wn, 'bandpass', analog=False)
    # To match Matlab output, we change default padlen from
    # 3*(max(len(a), len(b))) to 3*(max(len(a), len(b)) - 1)
    return filtfilt(b, a, signal, padlen=3 * (max(len(a), len(b)) - 1))
```

### Notch filter
```python

def notch_filter(signal, f_sampling, f_notch, bandwidth):
    """Implements a notch filter (e.g., for 50 or 60 Hz) on vector `data`.

    f_sampling = sample rate of data (input Hz or Samples/sec)
    f_notch = filter notch frequency (input Hz)
    bandwidth = notch 3-dB bandwidth (input Hz).  A bandwidth of 10 Hz is
    recommended for 50 or 60 Hz notch filters; narrower bandwidths lead to
    poor time-domain properties with an extended ringing response to
    transient disturbances.

    Example:  If neural data was sampled at 30 kSamples/sec
    and you wish to implement a 60 Hz notch filter:

    out = notch_filter(input, 30000, 60, 10);
    """
    tstep = 1.0 / f_sampling
    Fc = f_notch * tstep

    L = len(signal)

    # Calculate IIR filter parameters
    d = math.exp(-2.0 * math.pi * (bandwidth / 2.0) * tstep)
    b = (1.0 + d * d) * math.cos(2.0 * math.pi * Fc)
    a0 = 1.0
    a1 = -b
    a2 = d * d
    a = (1.0 + d * d) / 2.0
    b0 = 1.0
    b1 = -2.0 * math.cos(2.0 * math.pi * Fc)
    b2 = 1.0

    out = np.zeros(len(signal))
    out[0] = signal[0]
    out[1] = signal[1]
    # (If filtering a continuous data stream, change out[0:1] to the
    #  previous final two values of out.)

    # Run filter
    for i in range(2, L):
        out[i] = (a * b2 * signal[i - 2] + a * b1 * signal[i - 1] + a * b0 * signal[i] - a2 * out[i - 2] - a1 * out[
            i - 1]) / a0

    return out
```

The core of the peak detection is done in the following function:

```python
for i in range(nrSegments):
    # print( i*nrPerSegment, (i+1)*nrPerSegment )
    timeIdxs = np.arange(i * nrPerSegment, (i + 1) * nrPerSegment) / f_sampling * 1000.  # In ms
    # print(timeIdxs)

    # Apply bandpass (IIR) filter
    v1 = bandpass_filter(v[i * nrPerSegment:(i + 1) * nrPerSegment], f_sampling, f_low, f_high)
    v2 = v1 - np.nanmean(v1)

    # We threshold at noise_threshold * std (generally we use noise_threshold=3)
    # The denominator 0.6745 comes from some old neuroscience
    # paper which shows that when you have spikey data, correcting
    # with this number is better than just using plain standard
    # deviation value
    noiseLevel = -noise_threshold * np.median(np.abs(v2)) / 0.6745
    outside = np.array(v2) < noiseLevel  # Spits a logical array
    outside = outside.astype(int)  # Convert logical array to int array for diff to work

    cross = np.concatenate(([outside[0]], np.diff(outside, n=1) > 0))

    idxs = np.nonzero(cross)[0]
    spike_times_ms.extend(timeIdxs[idxs])
```

In SpikeInterface lingo:
nrSegments = number of chunks and v is a chunk, f_sampling is the recording sampling frequency.
