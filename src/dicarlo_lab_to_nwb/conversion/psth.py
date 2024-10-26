from pathlib import Path

import natsort
import numpy as np
from ndx_binned_spikes import BinnedAlignedSpikes
from pynwb import NWBHDF5IO, NWBFile
from tqdm.auto import tqdm


def calculate_event_psth_numpy_naive(
    spike_times_list,
    event_times_seconds,
    bin_width_in_milliseconds,
    number_of_bins,
    milliseconds_from_event_to_first_bin=0.0,
    number_of_events=None,
):
    """
    Calculate Peri-Stimulus Time Histogram (PSTH) for given events.

    Parameters
    ----------
    spike_times_list : list of arrays
        List where each element is an array of spike times (in seconds) for a single unit.
    event_times_seconds : array-like
        Array of event times (in seconds) for which the PSTH should be calculated.
    bin_width_in_milliseconds : float
        Width of each bin in the histogram (in milliseconds).
    number_of_bins : int
        Number of bins to include in the histogram.
    milliseconds_from_event_to_first_bin : float, optional
        Time offset (in milliseconds) from the event time to the start of the first bin.
        Default is 0.0. Negative times indicate that the bins start before the event time whereas positive times
        indicate that the bins start after the event time.
    number_of_events : int, optional
        Number of events to include in the calculation. If None, all events in
        `event_times_seconds` are included. Default is None. This is used if you want to aggregate this PSTH
        with other PSTHS that have different number of events.

    Returns
    -------
    event_psth : ndarray
        3D array of shape (number_of_units, number_of_events, number_of_bins) containing
        the PSTH for each unit and event.
    """
    if number_of_events is None:
        number_of_events = len(event_times_seconds)

    # We do everything in seconds
    event_times_seconds = np.asarray(event_times_seconds)
    bin_width_in_seconds = bin_width_in_milliseconds / 1000.0
    seconds_from_event_to_first_bin = milliseconds_from_event_to_first_bin / 1000.0

    base_bins_end = bin_width_in_seconds * number_of_bins
    base_bins = np.arange(0, base_bins_end + bin_width_in_seconds, bin_width_in_seconds)
    base_bins += seconds_from_event_to_first_bin

    number_of_units = len(spike_times_list)
    event_psth = np.full(shape=(number_of_units, number_of_events, number_of_bins), fill_value=np.nan)
    for channel_index, spike_times in enumerate(spike_times_list):
        for event_index, event_time in enumerate(event_times_seconds):
            event_bins = event_time + base_bins
            event_psth[channel_index, event_index] = np.histogram(spike_times, bins=event_bins)[0]

    return event_psth


def calculate_event_psth(
    spike_times_list,
    event_times_seconds,
    bin_width_in_milliseconds,
    number_of_bins,
    milliseconds_from_event_to_first_bin=0.0,
    number_of_events=None,
):
    """
    Calculate Peri-Stimulus Time Histogram (PSTH) for given events.

    Parameters
    ----------
    spike_times_list : list of arrays
        List where each element is an array of spike times (in seconds) for a single unit.
    event_times_seconds : array-like
        Array of event times (in seconds) for which the PSTH should be calculated.
    bin_width_in_milliseconds : float
        Width of each bin in the histogram (in milliseconds).
    number_of_bins : int
        Number of bins to include in the histogram.
    milliseconds_from_event_to_first_bin : float, optional
        Time offset (in milliseconds) from the event time to the start of the first bin.
        Default is 0.0. Negative times indicate that the bins start before the event time whereas positive times
        indicate that the bins start after the event time.
    number_of_events : int, optional
        Number of events to include in the calculation. If None, all events in
        `event_times_seconds` are included. Default is None. This is used if you want to aggregate this PSTH
        with other PSTHS that have different number of events.

    Returns
    -------
    event_psth : ndarray
        3D array of shape (number_of_units, number_of_events, number_of_bins) containing
        the PSTH for each unit and event.
    """

    event_times_seconds = np.asarray(event_times_seconds)
    if number_of_events is None:
        number_of_events = len(event_times_seconds)

    if hasattr(calculate_event_psth, "_cached_function"):
        event_psth = calculate_event_psth._cached_function(
            spike_times_list=spike_times_list,
            event_times_seconds=event_times_seconds,
            bin_width_in_milliseconds=bin_width_in_milliseconds,
            number_of_bins=number_of_bins,
            milliseconds_from_event_to_first_bin=milliseconds_from_event_to_first_bin,
            number_of_events=number_of_events,
        )

        return event_psth

    import numba
    from numba import prange

    @numba.jit(nopython=True, parallel=True)
    def _optimized_calculate_event_psth(
        spike_times_list,
        event_times_seconds,
        bin_width_in_milliseconds,
        number_of_bins,
        milliseconds_from_event_to_first_bin,
        number_of_events,
    ):
        # We do everything in seconds
        bin_width_in_seconds = bin_width_in_milliseconds / 1000.0
        seconds_from_event_to_first_bin = milliseconds_from_event_to_first_bin / 1000.0

        base_bins_end = bin_width_in_seconds * number_of_bins
        base_bins = np.arange(0, base_bins_end + bin_width_in_seconds, bin_width_in_seconds)
        base_bins += seconds_from_event_to_first_bin

        number_of_units = len(spike_times_list)
        event_psth = np.full(shape=(number_of_units, number_of_events, number_of_bins), fill_value=np.nan)
        for channel_index in prange(number_of_units):
            spike_times = spike_times_list[channel_index]
            for event_index, event_time in enumerate(event_times_seconds):
                event_bins = event_time + base_bins
                event_psth[channel_index, event_index] = np.histogram(spike_times, bins=event_bins)[0]

        return event_psth

    # Cache the compiled function
    calculate_event_psth._cached_function = _optimized_calculate_event_psth

    event_psth = calculate_event_psth._cached_function(
        spike_times_list=spike_times_list,
        event_times_seconds=event_times_seconds,
        bin_width_in_milliseconds=bin_width_in_milliseconds,
        number_of_bins=number_of_bins,
        milliseconds_from_event_to_first_bin=milliseconds_from_event_to_first_bin,
        number_of_events=number_of_events,
    )

    return event_psth


def calculate_psth_for_event_from_spikes_pespective(
    spike_trains_per_unit,
    event_times_seconds,
    bin_width_in_milliseconds,
    number_of_bins,
    milliseconds_from_event_to_first_bin=0.0,
    number_of_events=None,
):

    if number_of_events is None:
        number_of_events = len(event_times_seconds)

    bin_width_in_seconds = bin_width_in_milliseconds / 1000.0
    seconds_from_event_to_first_bin = milliseconds_from_event_to_first_bin / 1000.0

    base_bins_end = bin_width_in_seconds * number_of_bins
    base_bins = np.linspace(0, base_bins_end, number_of_bins + 1, endpoint=True)

    number_of_units = len(spike_trains_per_unit)
    event_psth = np.full(shape=(number_of_units, number_of_events, number_of_bins), fill_value=np.nan)

    base_bins_adjusted = base_bins + seconds_from_event_to_first_bin

    # Calculate last spike shifts
    spike_values = list(spike_trains_per_unit.values())
    previous_last_spikes = np.zeros(len(spike_values))
    previous_last_spikes[1:] = [spikes[-1] for spikes in spike_values[:-1]]

    last_spike_shifted = np.cumsum(previous_last_spikes)

    # Concatenate spikes and adjust bins
    spikes_concatenated = np.concatenate(
        [spikes + last_spike for spikes, last_spike in zip(spike_values, last_spike_shifted)]
    )
    all_bins = np.concatenate([base_bins_adjusted + last_spike for last_spike in last_spike_shifted])

    all_bins = np.append(all_bins, np.inf)

    all_bins = all_bins + event_times_seconds[:, np.newaxis]

    for event_time_index, event_bins in enumerate(all_bins):
        spikes_in_bins = np.histogram(spikes_concatenated, bins=event_bins)[0]
        repeat_psth = spikes_in_bins.reshape(number_of_units, number_of_bins + 1)[:, :-1]
        event_psth[:, event_time_index, :] = repeat_psth

    return event_psth


def build_psth_from_nwbfile(
    nwbfile: NWBFile,
    bin_width_in_milliseconds: float,
    number_of_bins: int,
    milliseconds_from_event_to_first_bin: float = 0.0,
    verbose: bool = False,
) -> tuple[dict, dict, list]:
    """
    Calculate peristimulus time histograms (PSTHs) for each stimulus from an NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file containing spike times and stimulus information.
    bin_width_in_milliseconds : float
        Width of each time bin in milliseconds.
    number_of_bins : int
        Total number of bins in the PSTH.
    milliseconds_from_event_to_first_bin : float, optional
        Time offset (in milliseconds) from the stimulus onset to the first bin center. Default is 0.0.
    verbose : bool, optional
        If True, display a progress bar during calculation. Default is False.

    Returns
    -------
    stimuli_id_psth : dict
        Dictionary where keys are stimulus IDs and values are arrays of PSTH counts.
    stimuli_id_times : dict
        Dictionary where keys are stimulus IDs and values are arrays of stimulus presentation times in seconds.
    unit_ids_order: list
        List of unit ids sorted by their name (A-001, A-002, ...). Used to build a map between the units
        in the psth and the units in the NWB file.
    Raises
    ------
    AssertionError
        If the NWB file does not contain a units table.
    """
    # list of spike_times
    units_data_frame = nwbfile.units.to_dataframe()
    unit_ids = units_data_frame["unit_name"].values
    spike_times = units_data_frame["spike_times"].values

    dict_of_spikes_times = {id: st for id, st in zip(unit_ids, spike_times)}

    # For the DiCarlo project it is important that units are sorted by their name (A-001, A-002, ...)
    unit_ids_order = sorted(unit_ids)
    spike_times_list = [dict_of_spikes_times[id] for id in unit_ids_order]

    trials_dataframe = nwbfile.trials.to_dataframe()
    stimuli_times = trials_dataframe["start_time"]  # In seconds
    stimuli_presentation_id = trials_dataframe["stimulus_presented"]
    stimuli_ids = stimuli_presentation_id.unique()

    # We also sort the stimuli by their id
    stimuli_ids_sorted = sorted(stimuli_ids)
    id_to_df_indices = {id: stimuli_presentation_id == id for id in stimuli_ids_sorted}
    stimuli_id_times = {id: stimuli_times[indices] for id, indices in id_to_df_indices.items()}

    stimuli_id_psth = {}
    desc = "Calculating PSTH for stimuli"
    for stimuli_id in tqdm(stimuli_ids_sorted, desc=desc, unit=" stimuli processed", disable=not verbose):
        stimulus_presentation_times = stimuli_id_times[stimuli_id]
        psth_per_stimuli = calculate_event_psth(
            spike_times_list=spike_times_list,
            event_times_seconds=stimulus_presentation_times,
            bin_width_in_milliseconds=bin_width_in_milliseconds,
            number_of_bins=number_of_bins,
            milliseconds_from_event_to_first_bin=milliseconds_from_event_to_first_bin,
        )
        stimuli_id_psth[stimuli_id] = psth_per_stimuli

    return stimuli_id_psth, stimuli_id_times, unit_ids_order


def build_binned_aligned_spikes_from_nwbfile(
    nwbfile: NWBFile,
    bin_width_in_milliseconds: float,
    number_of_bins: int,
    milliseconds_from_event_to_first_bin: float = 0.0,
    verbose: bool = False,
) -> BinnedAlignedSpikes:
    """
    Build `BinnedAlignedSpikes` objects for each stimulus from an NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file containing spike times and stimulus information.
    bin_width_in_milliseconds : float
        Width of each time bin in milliseconds.
    number_of_bins : int
        Total number of bins.
    milliseconds_from_event_to_first_bin : float, optional
        Time offset (in milliseconds) from the stimulus onset to the first bin center. Default is 0.0.
    verbose : bool, optional
        If True, display a progress bar during calculation. Default is False.

    Returns
    -------
    binned_spikes_dict : dict
        Dictionary where keys are stimulus IDs and values are `BinnedAlignedSpikes` objects.
    """
    from hdmf.common import DynamicTableRegion

    assert nwbfile.units is not None, "NWBFile does not have units table, psths cannot be calculated."

    stimuli_id_psth, stimuli_id_times, unit_ids_order = build_psth_from_nwbfile(
        nwbfile=nwbfile,
        bin_width_in_milliseconds=bin_width_in_milliseconds,
        number_of_bins=number_of_bins,
        milliseconds_from_event_to_first_bin=milliseconds_from_event_to_first_bin,
        verbose=verbose,
    )

    units_table = nwbfile.units

    # This guarantees that we link the unit to the corresponding psth in the
    # way that they were sorted in the psth calculation
    region_indices = []
    unit_names = units_table["unit_name"].data[:].tolist()
    for id in unit_ids_order:
        index = unit_names.index(id)
        region_indices.append(index)

    units_region = DynamicTableRegion(
        data=region_indices, table=units_table, description="region of units table", name="units_region"
    )

    event_timestamps = [stimuli_id_times[id] for id in stimuli_id_psth.keys()]
    data = [stimuli_id_psth[id] for id in stimuli_id_psth.keys()]
    condition_labels = [id for id in stimuli_id_psth.keys()]
    condition_indices = [[index] * len(stimuli_id_times[id]) for index, id in enumerate(stimuli_id_psth.keys())]

    event_timestamps = np.concatenate(event_timestamps)
    data = np.concatenate(data, axis=1)  # We concatenate across the events axis
    data = data.astype("uint64")
    condition_indices = np.concatenate(condition_indices)
    condition_indices = condition_indices.astype("uint64")

    data, event_timestamps, condition_indices = BinnedAlignedSpikes.sort_data_by_event_timestamps(
        data=data,
        event_timestamps=event_timestamps,
        condition_indices=condition_indices,
    )

    binned_aligned_spikes = BinnedAlignedSpikes(
        name=f"BinnedAlignedSpikesToStimulus",
        data=data,
        event_timestamps=event_timestamps,
        condition_indices=condition_indices,
        condition_labels=condition_labels,
        bin_width_in_milliseconds=bin_width_in_milliseconds,
        milliseconds_from_event_to_first_bin=milliseconds_from_event_to_first_bin,
        units_region=units_region,
    )

    return binned_aligned_spikes


def write_binned_spikes_to_nwbfile(
    nwbfile_path: Path | str,
    number_of_bins: int,
    bin_width_in_milliseconds: float,
    milliseconds_from_event_to_first_bin: float = 0.0,
    append: bool = True,
    verbose: bool = False,
) -> Path | str:
    """
    Calculate and write binned spike data to an NWB file.

    Parameters
    ----------
    nwbfile_path : Path or str
        Path to the NWB file.
    number_of_bins : int
        Total number of bins.
    bin_width_in_milliseconds : float
        Width of each time bin in milliseconds.
    milliseconds_from_event_to_first_bin : float, optional
        Time offset (in milliseconds) from the stimulus onset to the first bin center. Default is 0.0.
    append : bool, optional
        If True, append to the existing file. If False, create a new file. Default is False.
    verbose : bool, optional
        If True, print a message when finished. Default is False.

    Returns
    -------
    nwbfile_path : Path or str
        Path to the modified or new NWB file.
    """
    mode = "a" if append else "r"

    with NWBHDF5IO(nwbfile_path, mode=mode) as io:
        nwbfile = io.read()

        binned_aligned_spikes = build_binned_aligned_spikes_from_nwbfile(
            nwbfile=nwbfile,
            bin_width_in_milliseconds=bin_width_in_milliseconds,
            number_of_bins=number_of_bins,
            milliseconds_from_event_to_first_bin=milliseconds_from_event_to_first_bin,
            verbose=verbose,
        )

        ecephys_processing_module = nwbfile.create_processing_module(
            name="ecephys",
            description="Intermediate data derived from extracellular electrophysiology recordings such as PSTHs.",
        )

        ecephys_processing_module.add(binned_aligned_spikes)

        if append:
            io.write(nwbfile)

        else:
            nwbfile.generate_new_id()
            nwbfile_path = nwbfile_path.with_name(nwbfile_path.stem + "_with_binned_spikes.nwb")

            with NWBHDF5IO(nwbfile_path, mode="w") as export_io:
                export_io.export(src_io=io, nwbfile=nwbfile)

    if verbose:
        print(f"Appended binned spikes to {nwbfile_path}")
    return nwbfile_path


from ndx_binned_spikes import BinnedAlignedSpikes


def reshape_binned_spikes_data_to_pipeline_format(binned_aligned_spikes: BinnedAlignedSpikes) -> np.ndarray:
    """
    Extracts the PSTH data from the BinnedAlignedSpikes object and reshapes it to the DiCarlo lab convention.

    Reorganizes spike data from (units, stimuli, bins) format to
    (units, stimuli_index, stimuli_repetitions, time_bins) format, with stimuli ordered by
    naturally sorted condition labels. Empty repetitions are filled with NaN values.

    Parameters
    ----------
    binned_aligned_spikes : BinnedAlignedSpikes
        Object containing aligned spike data with conditions and labels. Must have attributes:
        - data : array of spike counts
        - condition_indices : indices mapping trials to conditions
        - condition_labels : labels for each condition
        - number_of_units : total number of units
        - number_of_bins : number of time bins
        - number_of_conditions : number of unique stimuli

    see `ndx-binned-spikes` for more information on the BinnedAlignedSpikes object.

    Returns
    -------
    np.ndarray
        Reshaped array with dimensions (units, stimuli_index, stimuli_repetitions, time_bins).
        The stimuli are ordered according to natural sorting of their condition labels.
        Missing repetitions are filled with NaN values.

    Notes
    -----
    The function uses natural sorting for condition labels, meaning that labels like
    ['stim1', 'stim2', 'stim10'] will be correctly ordered numerically instead of
    lexicographically. Empty repetition slots are filled with NaN to maintain consistent
    array dimensions across all stimuli.
    """
    import ndx_binned_spikes

    data = binned_aligned_spikes.data[:]
    condition_indices = binned_aligned_spikes.condition_indices[:]
    condition_labels = binned_aligned_spikes.condition_labels[:]

    number_of_units = binned_aligned_spikes.number_of_units
    number_of_bins = binned_aligned_spikes.number_of_bins
    number_of_stimuli = binned_aligned_spikes.number_of_conditions

    unique_conditions = np.unique(condition_indices)
    max_repetitions = max(np.sum(condition_indices == cond) for cond in unique_conditions)

    psth_pipepline_format = np.full((number_of_units, number_of_stimuli, max_repetitions, number_of_bins), np.nan)

    # Now, we will fill the data in the ordered of the sorted condition labels
    stimuli_labels_order = natsort.natsorted(condition_labels)

    for stimulus_index, label in enumerate(stimuli_labels_order):

        # Find corresponding condition index
        condition_index = np.where(condition_labels == label)[0][0]
        stimulus_psth = binned_aligned_spikes.get_data_for_condition(condition_index=condition_index)
        n_reps = stimulus_psth.shape[1]

        # Directly index the data using the condition trials
        psth_pipepline_format[:, stimulus_index, :n_reps, :] = stimulus_psth

    return psth_pipepline_format


def extract_psth_pipeline_format_from_nwbfile(
    nwbfile: NWBFile,
    binned_aligned_spikes_name: str | None = None,
) -> np.ndarray:
    """
    Extracts the PSTH data from the BinnedAlignedSpikes object and reshapes it to the DiCarlo lab convention.

    Reorganizes spike data from (units, stimuli, bins) format to
    (units, stimuli_index, stimuli_repetitions, time_bins) format, with stimuli ordered by
    naturally sorted condition labels. Empty repetitions are filled with NaN values.

    Parameters
    ----------
    binned_aligned_spikes : BinnedAlignedSpikes
        Object containing aligned spike data with conditions and labels. Must have attributes:
        - data : array of spike counts
        - condition_indices : indices mapping trials to conditions
        - condition_labels : labels for each condition
        - number_of_units : total number of units
        - number_of_bins : number of time bins
        - number_of_conditions : number of unique stimuli

    Returns
    -------
    np.ndarray
        Reshaped array with dimensions (units, stimuli_index, stimuli_repetitions, time_bins).
        The stimuli are ordered according to natural sorting of their condition labels.
        Missing repetitions are filled with NaN values.

    Notes
    -----
    The function uses natural sorting for condition labels, meaning that labels like
    ['stim1', 'stim2', 'stim10'] will be correctly ordered numerically instead of
    lexicographically. Empty repetition slots are filled with NaN to maintain consistent
    array dimensions across all stimuli.
    """

    from pynwb import NWBHDF5IO

    binned_aligned_spikes_name = binned_aligned_spikes_name or "BinnedAlignedSpikesToStimulus"

    binned_aligned_spikes = nwbfile.processing["ecephys"][binned_aligned_spikes_name]
    psth_pipeline_format = reshape_binned_spikes_data_to_pipeline_format(binned_aligned_spikes)

    return psth_pipeline_format


def write_psth_pipeline_format_to_nwbfile(
    nwbfile_path: str | Path,
    binned_aligned_spikes_name: str = "BinnedAlignedSpikesToStimulus",
    append: bool = True,
    verbose: bool = False,
) -> str | Path:

    import ndx_binned_spikes  # This is necessary to avoid some pynwb extensio bug where methods are not loaded

    mode = "a" if append else "r"

    with NWBHDF5IO(nwbfile_path, mode=mode) as io:
        nwbfile = io.read()

        psth_pipeline_format = extract_psth_pipeline_format_from_nwbfile(nwbfile, binned_aligned_spikes_name)

        description = f"PSTH data in the DiCarlo lab format (units, stimuli_index, stimuli_repetitions, time_bins)"
        nwbfile.add_scratch(psth_pipeline_format, name="psth_pipeline_format", description=description)

        if append:
            io.write(nwbfile)

        else:
            nwbfile.generate_new_id()
            nwbfile_path = nwbfile_path.with_name(nwbfile_path.stem + "_with_psth_in_scratch_pad.nwb")

            with NWBHDF5IO(nwbfile_path, mode="w") as export_io:
                export_io.export(src_io=io, nwbfile=nwbfile)
    if verbose:
        print(f"Appended PSTH in pipeline format to {nwbfile_path}")

    return nwbfile_path
