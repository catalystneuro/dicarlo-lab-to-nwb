import math

import numpy as np
from scipy.signal import ellip, filtfilt, lfilter, lfilter_zi, lfiltic
from spikeinterface.core import BaseRecording, ChunkRecordingExecutor
from spikeinterface.preprocessing import ScaleRecording
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
    filtered_signal = np.zeros_like(signal, dtype=np.float64)
    num_channels = signal.shape[1]
    for channel_index in range(num_channels):
        filtered_signal[:, channel_index] = filtfilt(b, a, signal[:, channel_index], axis=0, padlen=padlen)
    return filtered_signal


def bandpass_filter_vectorized(signal, f_sampling, f_low, f_high):
    wl = f_low / (f_sampling / 2.0)
    wh = f_high / (f_sampling / 2.0)
    wn = [wl, wh]

    # Design a 2nd-order Elliptic band-pass filter
    b, a = ellip(2, 0.1, 40, wn, "bandpass", analog=False)
    # Change default padlen to match Matlab output
    padlen = 3 * (max(len(a), len(b)) - 1)
    signal_size_GiB = signal.nbytes / 1024**3
    limit_chunk_size_GiB = 1

    if signal_size_GiB > limit_chunk_size_GiB:
        num_channels = signal.shape[1]
        channel_chunk_size = int(limit_chunk_size_GiB * 1024**3 / (signal.nbytes / num_channels))

        num_chunks = int(np.ceil(num_channels / channel_chunk_size))

        for i in range(num_chunks):
            start_idx = i * channel_chunk_size
            end_idx = min((i + 1) * channel_chunk_size, num_channels)
            signal[:, start_idx:end_idx] = filtfilt(b, a, signal[:, start_idx:end_idx], axis=0, padlen=padlen)
    else:
        signal = filtfilt(b, a, signal, axis=0, padlen=padlen)

    return signal


class DiCarloBandPass(BasePreprocessor):

    def __init__(self, recording: BaseRecording, f_low: float, f_high: float, vectorized: bool = False):
        BasePreprocessor.__init__(self, recording)
        self.f_low = f_low
        self.f_high = f_high
        self.f_sampling = recording.get_sampling_frequency()

        for parent_segment in recording._recording_segments:

            segment = DiCarloBandPassSegment(
                parent_segment, self.f_sampling, self.f_low, self.f_high, vectorized=vectorized
            )
            self.add_recording_segment(segment)

        self._kwargs = {
            "recording": recording,
            "f_low": f_low,
            "f_high": f_high,
            "vectorized": vectorized,
        }


class DiCarloBandPassSegment(BasePreprocessorSegment):

    def __init__(self, parent_segment, f_sampling, f_low, f_high, vectorized=False):
        BasePreprocessorSegment.__init__(self, parent_segment)
        self.parent_segment = parent_segment
        self.f_sampling = f_sampling
        self.f_low = f_low
        self.f_high = f_high
        self.vectorized = vectorized

    def get_traces(self, start_frame, end_frame, channel_indices):

        traces = self.parent_segment.get_traces(start_frame, end_frame, channel_indices)
        if self.vectorized:
            return bandpass_filter_vectorized(traces, self.f_sampling, self.f_low, self.f_high)
        else:
            return bandpass_filter(traces, self.f_sampling, self.f_low, self.f_high)


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

    """

    tstep = 1.0 / f_sampling
    Fc = f_notch * tstep

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

    filtered_signal = np.zeros_like(signal)
    filtered_signal[0:2, :] = signal[0:2, :]

    num_samples = signal.shape[0]
    num_channels = signal.shape[1]

    for channel_index in range(num_channels):
        for sample_index in range(2, num_samples):
            filtered_signal[sample_index, channel_index] = (
                a * b2 * signal[sample_index - 2, channel_index]
                + a * b1 * signal[sample_index - 1, channel_index]
                + a * b0 * signal[sample_index, channel_index]
                - a2 * filtered_signal[sample_index - 2, channel_index]
                - a1 * filtered_signal[sample_index - 1, channel_index]
            ) / a0

    return filtered_signal


def notch_filter_vectorized(signal, f_sampling, f_notch, bandwidth):

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

    filtered_signal = np.zeros_like(signal, dtype=np.float64)
    filtered_signal[0:2, :] = signal[0:2, :]

    num_samples = signal.shape[0]
    for sample_index in range(2, num_samples):
        filtered_signal[sample_index, :] = (
            a * b2 * signal[sample_index - 2, :]
            + a * b1 * signal[sample_index - 1, :]
            + a * b0 * signal[sample_index, :]
            - a2 * filtered_signal[sample_index - 2, :]
            - a1 * filtered_signal[sample_index - 1, :]
        ) / a0

    return filtered_signal


class DiCarloNotch(BasePreprocessor):
    def __init__(self, recording: BaseRecording, f_notch: float, bandwidth: float, vectorized: bool = False):
        super().__init__(recording)
        self.f_notch = f_notch
        self.bandwidth = bandwidth
        self.f_sampling = recording.get_sampling_frequency()

        for parent_segment in recording._recording_segments:
            segment = DiCarloNotchSegment(
                parent_segment,
                self.f_sampling,
                self.f_notch,
                self.bandwidth,
                vectorized=vectorized,
            )
            self.add_recording_segment(segment)

        self._kwargs = {
            "recording": recording,
            "f_notch": f_notch,
            "bandwidth": bandwidth,
            "vectorized": vectorized,
        }


class DiCarloNotchSegment(BasePreprocessorSegment):
    def __init__(self, segment, f_sampling, f_notch, bandwidth, vectorized=False):
        super().__init__(segment)
        self.parent_segment = segment
        self.f_sampling = f_sampling
        self.f_notch = f_notch
        self.bandwidth = bandwidth
        self.vectorized = vectorized

    def get_traces(self, start_frame, end_frame, channel_indices):

        traces = self.parent_segment.get_traces(start_frame, end_frame, channel_indices).astype(np.float64)

        if self.vectorized:
            return notch_filter_vectorized(traces, self.f_sampling, self.f_notch, self.bandwidth)
        else:
            return notch_filter(traces, self.f_sampling, self.f_notch, self.bandwidth)


def init_method(recording, noise_threshold=3):
    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["noise_threshold"] = noise_threshold

    return worker_ctx


def calculate_peak_in_chunks(segment_index, start_frame, end_frame, worker_ctx):

    recording = worker_ctx["recording"]
    noise_threshold = worker_ctx["noise_threshold"]

    traces = recording.get_traces(segment_index, start_frame=start_frame, end_frame=end_frame).astype("float32")
    number_of_channels = recording.get_num_channels()
    sampling_frequency = recording.get_sampling_frequency()
    times_in_chunk = np.arange(start_frame, end_frame) / sampling_frequency

    spikes_per_channel = []
    for channel_index in range(number_of_channels):
        channel_traces = traces[:, channel_index]
        centered_channel_traces = channel_traces - np.nanmean(channel_traces)

        std_estimate = np.median(np.abs(centered_channel_traces)) / 0.6745
        noise_level = -noise_threshold * std_estimate

        # Core of method
        outside = centered_channel_traces < noise_level
        outside = outside.astype(int)  # Convert logical array to int array for diff to work
        cross = np.concatenate(([outside[0]], np.diff(outside, n=1) > 0))

        indices_by_channel = np.nonzero(cross)[0]

        spikes_per_channel.append(times_in_chunk[indices_by_channel])

    return spikes_per_channel


def calculate_peak_in_chunks_vectorized(segment_index, start_frame, end_frame, worker_ctx):
    recording = worker_ctx["recording"]
    noise_threshold = worker_ctx["noise_threshold"]

    traces = recording.get_traces(segment_index, start_frame=start_frame, end_frame=end_frame)
    sampling_frequency = recording.get_sampling_frequency()
    times_in_chunk = np.arange(start_frame, end_frame) / sampling_frequency

    # Centering the traces (in-place)
    traces -= np.nanmean(traces, axis=0, keepdims=True)

    # Estimating standard deviation with the MAD
    absolute_traces = np.abs(traces, out=traces)
    std_estimate = np.median(absolute_traces, axis=0) / 0.6745

    # Calculating the noise level threshold for each channel
    noise_level = -noise_threshold * std_estimate

    # Detecting crossings below the noise level for each channel
    outside = traces < noise_level[np.newaxis, :]

    # Initialize cross array
    cross = np.zeros_like(outside, dtype=bool)

    # Manually calculate differences and handle the initial state
    cross[1:] = outside[1:] & ~outside[:-1]
    cross[0] = outside[0]

    # Find indices where crossings occur
    peaks_idx = np.nonzero(cross)
    peak_times_channels = times_in_chunk[peaks_idx[0]]

    # Reshape the results into a list of arrays, one for each channel
    channel_indices = peaks_idx[1]
    num_channels = traces.shape[1]
    all_peak_times = [peak_times_channels[channel_indices == channel_index] for channel_index in range(num_channels)]

    return all_peak_times


def thresholding_preprocessing(
    recording: BaseRecording,
    f_notch: float = 60.0,
    bandwidth: float = 10,
    f_low: float = 300.0,
    f_high: float = 6_000.0,
    vectorized: bool = True,
) -> BasePreprocessor:
    if recording.has_scaleable_traces():
        gain = recording.get_channel_gains()
        offset = recording.get_channel_offsets()
    else:
        gain = np.ones(recording.get_num_channels(), dtype="float32")
        offset = np.zeros(recording.get_num_channels(), dtype="float32")
    scaled_to_uV_recording = ScaleRecording(recording=recording, gain=gain, offset=offset)
    notched_recording = DiCarloNotch(
        scaled_to_uV_recording, f_notch=f_notch, bandwidth=bandwidth, vectorized=vectorized
    )
    preprocessed_recording = DiCarloBandPass(
        recording=notched_recording, f_low=f_low, f_high=f_high, vectorized=vectorized
    )

    return preprocessed_recording


def thresholding_peak_detection(
    recording: BaseRecording,
    noise_threshold: float = 3,
    vectorized: bool = True,
    job_kwargs: dict = None,
) -> dict:
    job_name = "DiCarloPeakDetectionPipeline"

    if job_kwargs is None:
        chunk_size = math.ceil(recording.get_num_samples() / 10)
        job_kwargs = dict(n_jobs=1, verbose=True, progress_bar=True, chunk_size=chunk_size)
    init_args = (recording, noise_threshold)
    processor = ChunkRecordingExecutor(
        recording,
        calculate_peak_in_chunks_vectorized if vectorized else calculate_peak_in_chunks,
        init_method,
        init_args,
        handle_returns=True,
        job_name=job_name,
        **job_kwargs,
    )

    values = processor.run()

    spike_times_per_channel = {}

    channel_ids = recording.get_channel_ids()

    for channel_index, channel_id in enumerate(channel_ids):
        channel_spike_times = [times[channel_index] for times in values]
        channel_ids
        spike_times_per_channel[channel_id] = np.concatenate(channel_spike_times)

    return spike_times_per_channel


def thresholding_pipeline(
    recording: BaseRecording,
    f_notch: float = 60.0,
    bandwidth: float = 10,
    f_low: float = 300.0,
    f_high: float = 6_000.0,
    noise_threshold: float = 3,
    vectorized: bool = True,
    job_kwargs: dict = None,
):

    preprocessed_recording = thresholding_preprocessing(
        recording=recording,
        f_notch=f_notch,
        bandwidth=bandwidth,
        f_low=f_low,
        f_high=f_high,
        vectorized=vectorized,
    )
    spike_times_per_channel = thresholding_peak_detection(
        recording=preprocessed_recording,
        noise_threshold=noise_threshold,
        vectorized=vectorized,
        job_kwargs=job_kwargs,
    )

    return spike_times_per_channel


if __name__ == "__main__":
    from spikeinterface.core.generate import generate_ground_truth_recording
    from spikeinterface.core.job_tools import ChunkRecordingExecutor

    from dicarlo_lab_to_nwb.conversion.pipeline import di_carlo_pipeline

    recording, sorting = generate_ground_truth_recording(num_channels=4, num_units=1, durations=[1], seed=0)

    f_notch = 60
    bandwidth = 10
    f_low = 300.0
    f_high = 6000.0
    noise_threshold = 3

    spikes_per_channel = di_carlo_pipeline(
        recording=recording,
        f_notch=f_notch,
        bandwidth=bandwidth,
        f_low=f_low,
        f_high=f_high,
        noise_threshold=noise_threshold,
    )
