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

## Single session structure

run:

stimulus set might have more than one subject, each subject has more than one session.

run is what I usually call a session

experiment is defined by the stimuli set

Description:

monkey comes into the rig, you do the normalization, then yrou run the experiment (let’s say a stimulus set 1 with monkey 1). Then this will be saved on the local PC. And then you upload to the data server.

## Notes about synchornization

The following code can be found in `mkw_ultra.py` in the repository of the DiCarlo lab and:


```python
filename = os.path.join(mworksRawDir, filenameList[args.session_num])
photodiode_file = os.path.join(intanRawDir, 'board-ANALOG-IN-1.dat')
sample_on_file = os.path.join(intanRawDir, 'board-DIGITAL-IN-02.dat')
```

## Mworks

The code for this is divided in two functions. `dump_events` and `get_events` in `mkw_ultra.py` in the repository of the DiCarlo lab here. They can be find here:

https://github.com/AliyaAbl/DiCarlo_NWB/blob/be36d5f710fd5fa2620a495865d280457bc7a847/spike-tools-chong/spike_tools/mwk_ultra.py#L207-L216

And here:
https://github.com/AliyaAbl/DiCarlo_NWB/blob/be36d5f710fd5fa2620a495865d280457bc7a847/spike-tools-chong/spike_tools/mwk_ultra.py#L67-L74


The variable naming convention is sub-optimal.

```
names = ['trial_start_line',
            'correct_fixation',
            'stimulus_presented',
            'stim_on_time',
            'stim_off_time',
            'stim_on_delay',
            'stimulus_size',
            'fixation_window_size',
            'fixation_point_size_min',
            'eye_h',
            'eye_v']
```

Trial structure:
One behavioral experiment. Train the monkey to mantain a fix gaze (around ~1 to 2 seconds).
While the monkey mantain fixation, we first make sure that for the first ~300 ms we are sure that is fixating, then we flash images. The time in between them is the inter stimulus interval.

One trial equals one fixation. And within one trial we present multiple images (e.g. 8 images).

150ms off



Notes about the columns:
* `stim_on_time_ms`: onset duration (how long as this image presented)
* `stim_off_time_ms`: inter stimulus interval
* `stimulus_order_on_trial`: this goes between 1 and 8 the max value in `stimulus_order_on_trial` corresponds to number of images presented in one trial
* `trial_start_line` : pulse at the beginning of the trial
* `correct_fixation`: (boolean) sometimes the monkey breaks the fixation prematurely then it will be 0. So you use this variable to discard. Constantly monitor the eye position, example a monkey correctly fixates 4 out of 8, so the first 4 are valid.
* `stimulus_presented`: image_id they match the image convention in the file name of the corresponding stimuli set. Typically they are named by count.
* `stim_on_delay`: From the point of view of the animal, fixation point first comes on, minimal duration to ensure that we are looking at it, the delay time between fixation and the first image presentation.
* `samp_on_us` : sample in microseconds, counte for mworks, digital counter internally for Mworks.
* `photodiode_on_us`: time that the photo diode was on

Note:
total_fixation_time = stim_on_delay +(stim_on_time_ms + stim_off_time_ms) * max(stimulus_order_on_trial)

trial_start_line: the trial starts (what is a trial)

### Rig structure (notes from meeting)


Three computers:
* One for the stimulus presentation (mac) which works as an orchestrator and initiates the experiment. Mworks run on the mac computer and presents the stimuli.
* Another for eye linke eye tracking.
* Another for the Intan recording system. Pasive.

There is an Arduino in between the Intan and the mac that converts the USB signal from the mac to a digital flag in the Intan recording system. This is stored in the digital in channels of Intan:

* `board-DIGITAL-IN-01.dat`: beginning of the experiment (stays high during the recording sessions).
* `board-DIGITAL-IN-02.dat`: stimulus onset from mac computer


#### Photodiode

A small corner of the screen is flashed when a stimuli is presented. The photodiode is connected to the Intan recording system in `board-ANALOG-IN-1.dat`. This is another trigger to stimuli onset and is used to deal with frame drop and delay.

Extra note:
Intan has a delay buffer, triggers one second before, then I Guess the [DIGITAL-IN-02.data](http://DIGITAL-IN-02.data) goes up.



### Questions:
* Why have an intermediate preprocessing step here. Why first produce a csv file, why not taking it directly to NWB?
* Is this the only place were the `stimulis_id` appears?

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
