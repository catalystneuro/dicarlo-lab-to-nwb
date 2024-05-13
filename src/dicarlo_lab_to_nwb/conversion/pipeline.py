import numpy as np
from scipy.signal import ellip, filtfilt
from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessor,
    BasePreprocessorSegment,
)


def bandpass_filter(signal, f_sampling, f_low, f_high):
    wl = f_low / (f_sampling / 2.0)
    wh = f_high / (f_sampling / 2.0)
    wn = [wl, wh]

    # Designs a 2nd-order Elliptic band-pass filter which passes
    # frequencies between normalized f_low and f_high, and with 0.1 dB of ripple
    # in the passband, and 40 dB of attenuation in the stopband.
    b, a = ellip(2, 0.1, 40, wn, "bandpass", analog=False)
    # To match Matlab output, we change default padlen from
    # 3*(max(len(a), len(b))) to 3*(max(len(a), len(b)) - 1)
    padlen = 3 * (max(len(a), len(b)) - 1)

    # TODO, test if it vectorizes makes a difference

    return filtfilt(b, a, signal, padlen=padlen)


def bandpass_filter_vectorized(signal, f_sampling, f_low, f_high):

    wl = f_low / (f_sampling / 2.0)
    wh = f_high / (f_sampling / 2.0)
    wn = [wl, wh]

    # Designs a 2nd-order Elliptic band-pass filter which passes
    # frequencies between normalized f_low and f_high, and with 0.1 dB of ripple
    # in the passband, and 40 dB of attenuation in the stopband.
    b, a = ellip(2, 0.1, 40, wn, "bandpass", analog=False)
    # To match Matlab output, we change default padlen from
    # 3*(max(len(a), len(b))) to 3*(max(len(a), len(b)) - 1)
    padlen = 3 * (max(len(a), len(b)) - 1)

    return filtfilt(b, a, signal, axis=0, padlen=padlen)


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
        out[n, :] = (
            a * b2 * signal[n - 2, :]
            + a * b1 * signal[n - 1, :]
            + a * b0 * signal[n, :]
            - a2 * out[n - 2, :]
            - a1 * out[n - 1, :]
        ) / a0

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


import numpy as np
from spikeinterface.core.job_tools import ChunkRecordingExecutor


def init_method(recording, noise_threshold=3):
    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["noise_threshold"] = noise_threshold

    return worker_ctx


def calculate_peak_in_chunks(segment_index, start_frame, end_frame, worker_ctx):

    recording = worker_ctx["recording"]
    noise_threshold = worker_ctx["noise_threshold"]

    traces = recording.get_traces(segment_index, start_frame=start_frame, end_frame=end_frame)
    centered_traces = traces - np.median(traces, axis=0)
    std_estimate = np.median(np.abs(centered_traces), axis=0) / 0.6744
    noiseLevel = -noise_threshold * std_estimate

    # Core of method
    outside = np.array(centered_traces) < noiseLevel
    outside = outside.astype(int)  # Convert logical array to int array for diff to work
    cross = np.concatenate(([outside[0]], np.diff(outside, n=1) > 0))

    idxs = np.nonzero(cross)[0]

    sampling_frequency = recording.get_sampling_frequency()
    times_in_chunk = np.arange(start_frame, end_frame) / sampling_frequency

    return times_in_chunk[idxs]


def calculate_peak_in_chunks(segment_index, start_frame, end_frame, worker_ctx):
    recording = worker_ctx["recording"]
    noise_threshold = worker_ctx["noise_threshold"]

    traces = recording.get_traces(segment_index, start_frame=start_frame, end_frame=end_frame)
    sampling_frequency = recording.get_sampling_frequency()
    times_in_chunk = np.arange(start_frame, end_frame) / sampling_frequency

    # Centering the traces
    centered_traces = traces - np.median(traces, axis=0)

    # Estimating standard deviation by with the MAD
    std_estimate = np.median(np.abs(centered_traces), axis=0) / 0.6744

    # Calculating the noise level threshold for each channel
    noise_level = -noise_threshold * std_estimate

    # Detecting crossings below the noise level for each channel
    outside = centered_traces < noise_level[np.newaxis, :]
    cross_diff = np.diff(outside.astype(int), axis=0, prepend=outside[0:1, :])
    cross = cross_diff > 0

    # Find indices where crossings occur
    peaks_idx = np.nonzero(cross)
    peak_times_channels = times_in_chunk[peaks_idx[0]]

    # Reshape the results into a list of arrays, one for each channel
    channel_indices = peaks_idx[1]
    all_peak_times = [peak_times_channels[channel_indices == i] for i in range(traces.shape[1])]

    return all_peak_times


if __name__ == "__main__":

    from spikeinterface.core.generate import generate_ground_truth_recording

    from dicarlo_lab_to_nwb.conversion.pipeline import DiCarloBandPass, DiCarloNotch

    recording, sorting = generate_ground_truth_recording(num_channels=4, num_units=1, durations=[1], seed=0)
    recording, sorting

    f_notch = 50
    bandwidth = 10
    f_low = 300.0
    f_high = 6000.0

    notched_recording = DiCarloNotch(recording, f_notch=f_notch, bandwidth=bandwidth)
    preprocessed_recording = DiCarloBandPass(notched_recording, f_low=f_low, f_high=f_high)

    job_name = "DiCarloPeakDetectionPipeline"
    job_kwargs = dict(n_jobs=1, verbose=True, progress_bar=True, chunk_duration=1.0)
    init_args = (preprocessed_recording, 3)
    processor = ChunkRecordingExecutor(
        recording,
        calculate_peak_in_chunks,
        init_method,
        init_args,
        handle_returns=True,
        job_name=job_name,
        **job_kwargs,
    )
