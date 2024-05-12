from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
import numpy as np
from scipy.signal import ellip, filtfilt

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
    padlen = 3 * (max(len(a), len(b)) - 1)
    
    
    #TODO, test if it vectorizes makes a difference
    
    return filtfilt(b, a, signal, padlen=padlen)  


def bandpass_filter_vectorized(signal, f_sampling, f_low, f_high):
    
    
    wl = f_low / (f_sampling / 2.)
    wh = f_high / (f_sampling / 2.)
    wn = [wl, wh]

    # Designs a 2nd-order Elliptic band-pass filter which passes
    # frequencies between normalized f_low and f_high, and with 0.1 dB of ripple
    # in the passband, and 40 dB of attenuation in the stopband.
    b, a = ellip(2, 0.1, 40, wn, 'bandpass', analog=False)
    # To match Matlab output, we change default padlen from
    # 3*(max(len(a), len(b))) to 3*(max(len(a), len(b)) - 1)
    padlen = 3 * (max(len(a), len(b)) - 1)
    
    return filtfilt(b, a, signal, axis=0 , padlen=padlen)

class DiCarloBandPass(BasePreprocessor):
    
    def __init__(self, recording, f_low, f_high):
        BasePreprocessor.__init__(self, recording)
        self.f_low = f_low
        self.f_high = f_high
        self.f_sampling = recording.get_sampling_frequency()
        
        for parent_segment in recording._recording_segments:
            
            segment = DiCarloBandPassSegment(parent_segment, self.f_sampling, self.f_low, self.f_high)
            self.add_recording_segment(segment)
        
        
class DiCarloBandPassSegment(BasePreprocessorSegment):
    
    def __init__(self, parent_segment, f_sampling, f_low, f_high):
        BasePreprocessorSegment.__init__(self, parent_segment)
        self.parent_segment = parent_segment
        self.f_sampling = f_sampling
        self.f_low = f_low
        self.f_high = f_high
        
    def get_traces(self, start_frame, end_frame, channel_indices):
        
        traces = self.parent_segment.get_traces(start_frame, end_frame, channel_indices)
        
        return bandpass_filter(traces, self.f_sampling, self.f_low, self.f_high)
    


def notch_filter(signal, f_sampling, f_notch, bandwidth):
    tstep = 1.0 / f_sampling
    Fc = f_notch * tstep
    d = np.exp(-2.0 * np.pi * (bandwidth / 2.0) * tstep)
    b = (1.0 + d * d) * np.cos(2.0 * np.pi * Fc)
    
    a0 = 1.0
    a1 = -b
    a2 = d * d
    a = (1.0 + d * d) / 2.0
    b0 = 1.0
    b1 = -2.0 * np.cos(2.0 * np.pi * Fc)
    b2 = 1.0
    
    out = np.zeros_like(signal)
    out[:, 0] = signal[:, 0]
    if signal.shape[1] > 1:
        out[:, 1] = signal[:, 1]
    
    for n in range(2, signal.shape[0]):
        out[n, :] = (a * b2 * signal[n - 2, :] + a * b1 * signal[n - 1, :] + a * b0 * signal[n, :] -
                     a2 * out[n - 2, :] - a1 * out[n - 1, :]) / a0

    return out

class DiCarloNotch(BasePreprocessor):
    def __init__(self, recording, f_notch, bandwidth):
        super().__init__(recording)
        self.f_notch = f_notch
        self.bandwidth = bandwidth
        self.f_sampling = recording.get_sampling_frequency()
        
        for parent_segment in recording._recording_segments:
            segment = DiCarloNotchSegment(parent_segment, self.f_sampling, self.f_notch, self.bandwidth)
            self.add_recording_segment(segment)

class DiCarloNotchSegment(BasePreprocessorSegment):
    def __init__(self, segment, f_sampling, f_notch, bandwidth):
        super().__init__(segment)
        self.parent_segment = segment
        self.f_sampling = f_sampling
        self.f_notch = f_notch
        self.bandwidth = bandwidth
        
    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_segment.get_traces(start_frame, end_frame, channel_indices)
        return notch_filter(traces, self.f_sampling, self.f_notch, self.bandwidth)
