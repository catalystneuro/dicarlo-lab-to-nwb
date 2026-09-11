import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from pynwb.misc import Units
from scipy.signal import ellip, filtfilt
from spikeinterface.core import BaseRecording, ChunkRecordingExecutor
from spikeinterface.extractors import IntanRecordingExtractor, NwbRecordingExtractor
from spikeinterface.preprocessing import ScaleRecording
from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessor,
    BasePreprocessorSegment,
)

from .probe import attach_probe_to_recording


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

    # Designs a 2nd-order Elliptic band-pass filter which passes
    # frequencies between normalized f_low and f_high, and with 0.1 dB of ripple
    # in the passband, and 40 dB of attenuation in the stopband.
    b, a = ellip(2, 0.1, 40, wn, "bandpass", analog=False)
    # To match Matlab output, we change default padlen from
    # 3*(max(len(a), len(b))) to 3*(max(len(a), len(b)) - 1)
    padlen = 3 * (max(len(a), len(b)) - 1)

    return filtfilt(b, a, signal, axis=0, padlen=padlen)


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
    # This extracts the data after notch and bandpass filtering has been applied
    traces = recording.get_traces(segment_index, start_frame=start_frame, end_frame=end_frame)
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

    # Estimating standard deviation with the MAD and the noise level
    absolute_traces = np.abs(traces)
    std_estimate = np.median(absolute_traces, axis=0) / 0.6745
    noise_level = -noise_threshold * std_estimate

    # Detecting crossings below the noise level for each channel
    outside = traces < noise_level[np.newaxis, :]

    # Creating a cross_diff array with prepend to ensure we capture the initial crossing
    cross_diff = np.diff(outside.astype(int), axis=0, prepend=outside[0:1, :])
    cross = cross_diff > 0

    # Checking the first point separately
    cross[0, :] = outside[0, :]

    # Find indices where crossings occur
    peaks_idx = np.nonzero(cross)
    peak_times_channels = times_in_chunk[peaks_idx[0]]

    # Reshape the results into a list of arrays, one for each channel
    channel_indices = peaks_idx[1]
    all_peak_times = [peak_times_channels[channel_indices == i] for i in range(traces.shape[1])]

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
    verbose: bool = False,
) -> dict:
    job_name = "DiCarloPeakDetectionPipeline"

    if job_kwargs is None:
        progress_bar = verbose
        job_kwargs = dict(n_jobs=-1, progress_bar=progress_bar, chunk_duration=10.0)  # Fixed chunks to 10 seconds

    init_args = (recording, noise_threshold)
    processor = ChunkRecordingExecutor(
        recording,
        calculate_peak_in_chunks_vectorized if vectorized else calculate_peak_in_chunks,
        init_method,
        init_args,
        handle_returns=True,
        job_name=job_name,
        verbose=verbose,
        **job_kwargs,
    )

    values = processor.run()

    spike_times_per_channel = {}

    channel_ids = recording.get_channel_ids()

    for channel_index, channel_id in enumerate(channel_ids):
        channel_spike_times = [times[channel_index] for times in values]
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
    verbose: bool = False,
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
        verbose=verbose,
    )

    return spike_times_per_channel


def calculate_thresholding_events(
    file_path: Path | str,
    f_notch: float = 60.0,
    bandwidth: float = 10,
    f_low: float = 300.0,
    f_high: float = 6_000.0,
    noise_threshold: float = 3,
    vectorized: bool = True,
    job_kwargs: dict = None,
    stub_test: bool = False,
    verbose: bool = False,
    probe_info_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Extracts spike events from neural recordings using a thresholding pipeline and returns the sorted spike times.
    Supports both Intan (.rhd) and NWB file formats.

    Parameters
    ----------
    file_path : Path or str
        Path to the recording file (.rhd for Intan or .nwb for NWB format)
    f_notch : float, optional
        Notch filter frequency to remove power line noise, by default 60.0 Hz
    bandwidth : float, optional
        Bandwidth for the notch filter, by default 10 Hz
    f_low : float, optional
        Low cutoff frequency for bandpass filtering, by default 300.0 Hz
    f_high : float, optional
        High cutoff frequency for bandpass filtering, by default 6000.0 Hz
    noise_threshold : float, optional
        Threshold for detecting spikes, by default 3
    vectorized : bool, optional
        Whether to use a vectorized implementation of the thresholding, by default True
    job_kwargs : dict, optional
        Additional keyword arguments for job control, such as chunk size, by default None
    stub_test : bool, optional
        Whether to run a short test on the first 10 seconds of the recording, by default False
    verbose : bool, optional
        Whether to print progress messages, by default False
    probe_info_path : Path | str, optional
        Path to the CSV file containing probe information. Only used for Intan recordings.
        If not provided for Intan recordings, the default path is used. see `attach_probe_to_recording` documentation.

    Returns
    -------
    pd.DataFrame
        One row per channel, with its `channel_name`, `probe`, `unit_location_um` and `spike_times` in seconds

    Notes
    -----
    This function automatically detects the file format from the extension and applies the appropriate loading method.
    It applies filtering and thresholding to detect spike events, processing each probe separately.
    The processing pipeline includes:
    1. Loading the recording
    2. Applying notch and bandpass filters
    3. Detecting spikes using threshold crossing
    4. Organizing results by probe and channel
    """
    file_path = Path(file_path)
    assert file_path.is_file(), f"{file_path} does not exist"

    # Initialize recording based on file extension
    if file_path.suffix.lower() == ".rhd":
        # Always use amplifier channels for Intan recordings
        stream_name = "RHD2000 amplifier channel"
        recording = IntanRecordingExtractor(file_path=file_path, ignore_integrity_checks=True, stream_name=stream_name)
        attach_probe_to_recording(recording=recording, probe_info_path=probe_info_path)
    else:  # .nwb
        recording = NwbRecordingExtractor(file_path=file_path, use_pynwb=True)

    sampling_frequency = recording.get_sampling_frequency()

    # Handle stub test case
    if stub_test:
        duration = recording.get_duration() - 1 / sampling_frequency
        end_time = min(10.0, duration)
        processed_recording = recording.time_slice(start_time=0, end_time=end_time)
        progress_bar = verbose
        job_kwargs = dict(n_jobs=1, progress_bar=progress_bar, chunk_duration=end_time)
    else:
        processed_recording = recording

    # Process each probe
    dict_of_recordings = processed_recording.split_by(property="probe", outputs="dict")
    dict_of_spikes_times_per_channel = {}

    for probe_name, probe_recording in dict_of_recordings.items():
        if verbose:
            print(f"Calculating thresholding events for probe {probe_name}")
            print(probe_recording)

        spikes_times_per_channel = thresholding_pipeline(
            recording=probe_recording,
            f_notch=f_notch,
            bandwidth=bandwidth,
            f_low=f_low,
            f_high=f_high,
            noise_threshold=noise_threshold,
            vectorized=vectorized,
            job_kwargs=job_kwargs,
            verbose=verbose,
        )

        dict_of_spikes_times_per_channel[probe_name] = spikes_times_per_channel

    # One row per channel, which is what the units table needs. Going through a spikeinterface sorting object
    # instead makes it keep several copies of every spike in memory until the units are written
    channel_ids = processed_recording.get_channel_ids()
    channel_locations = processed_recording.get_channel_locations()
    channel_probe_names = processed_recording.get_property("probe")

    spikes_times_per_channel = {}
    for probe_spikes_times_per_channel in dict_of_spikes_times_per_channel.values():
        spikes_times_per_channel.update(probe_spikes_times_per_channel)

    units_dataframe = pd.DataFrame(
        {
            "channel_name": channel_ids,
            "probe": channel_probe_names,
            "unit_location_um": list(channel_locations),
            "spike_times": [spikes_times_per_channel[channel_id] for channel_id in channel_ids],
        }
    )
    # The detection returns times on the sample grid, so this is the smallest difference between two spikes
    units_dataframe.attrs["sampling_frequency"] = sampling_frequency

    if verbose:
        number_of_spikes = sum(len(spike_times) for spike_times in units_dataframe["spike_times"])
        print(f"Detected {number_of_spikes} spikes in {len(units_dataframe)} channels")

    # Clean up
    del recording
    if "processed_recording" in locals():
        del processed_recording

    return units_dataframe


def write_thresholding_events_to_nwb(
    units_dataframe: pd.DataFrame,
    nwbfile_path: str | Path,
    thresholindg_pipeline_kwargs: dict,
    append: bool = True,
    verbose: bool = False,
):

    mode = "a" if append else "r"

    with NWBHDF5IO(nwbfile_path, mode=mode) as io:
        nwbfile = io.read()

        units_description = (
            "Spike times detected using thresholding with DiCarlo lab pipeline and the following parameters: \n "
            f"{thresholindg_pipeline_kwargs}"
        )

        # We map the units to electrodes based on the unit name
        # For the Utah Array probe there is only site per channel
        channel_names = nwbfile.electrodes["channel_name"].data[:].tolist()
        unit_electrode_indices = [[channel_names.index(unit_name)] for unit_name in units_dataframe["channel_name"]]

        # The table is written column by column, which keeps the spike times in one array instead of the
        # several copies per spike that a spikeinterface sorting object holds
        units_table = Units(
            name="units",
            description=units_description,
            resolution=1.0 / units_dataframe.attrs["sampling_frequency"],
        )
        units_table.id.extend(list(range(len(units_dataframe))))
        units_table.add_column(
            name="spike_times",
            description="the spike times for each unit in seconds",
            data=units_dataframe["spike_times"].tolist(),
            index=True,
        )
        units_table.add_column(
            name="electrodes",
            description="the electrodes that each spike unit came from",
            data=unit_electrode_indices,
            index=True,
            table=nwbfile.electrodes,
        )
        units_table.add_column(
            name="unit_name",
            description="Unique reference for each unit.",
            data=units_dataframe["channel_name"].tolist(),
        )
        units_table.add_column(
            name="unit_location_um",
            description="Location of the electrode of the unit in micrometers.",
            data=units_dataframe["unit_location_um"].tolist(),
            index=True,
        )
        units_table.add_column(
            name="probe",
            description="Probe the unit was recorded from.",
            data=units_dataframe["probe"].tolist(),
        )
        nwbfile.units = units_table

        if append:
            io.write(nwbfile)

        else:
            nwbfile.generate_new_id()
            nwbfile_path = nwbfile_path.with_name(nwbfile_path.stem + "_sorted.nwb")

            with NWBHDF5IO(nwbfile_path, mode="w") as export_io:
                export_io.export(src_io=io, nwbfile=nwbfile)
    if verbose:
        print(f"Thresholding events written to {nwbfile_path}")

    return nwbfile_path


if __name__ == "__main__":
    from spikeinterface.core.generate import generate_ground_truth_recording
    from spikeinterface.core.job_tools import ChunkRecordingExecutor

    from dicarlo_lab_to_nwb.conversion.pipeline import thresholding_pipeline

    recording, sorting = generate_ground_truth_recording(num_channels=4, num_units=1, durations=[1], seed=0)

    f_notch = 60
    bandwidth = 10
    f_low = 300.0
    f_high = 6000.0
    noise_threshold = 3

    spikes_per_channel = thresholding_pipeline(
        recording=recording,
        f_notch=f_notch,
        bandwidth=bandwidth,
        f_low=f_low,
        f_high=f_high,
        noise_threshold=noise_threshold,
    )
