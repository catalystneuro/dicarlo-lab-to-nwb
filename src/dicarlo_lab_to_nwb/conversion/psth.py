from pathlib import Path

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
) -> tuple[dict, dict]:
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
    psth_dict : dict
        Dictionary where keys are stimulus IDs and values are arrays of PSTH counts.
    stimuli_presentation_times_dict : dict
        Dictionary where keys are stimulus IDs and values are arrays of stimulus presentation times in seconds.

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
    unit_ids_sorted = sorted(unit_ids)
    spike_times_list = [dict_of_spikes_times[id] for id in unit_ids_sorted]

    trials_dataframe = nwbfile.trials.to_dataframe()
    stimuli_presentation_times_seconds = trials_dataframe["start_time"]
    stimuli_presentation_id = trials_dataframe["stimulus_presented"]
    stimuli_ids = stimuli_presentation_id.unique()

    # We also sort the stimuli by their id
    stimuli_ids_sorted = sorted(stimuli_ids)

    stimuli_presentation_times_dict = {
        stimulus_id: stimuli_presentation_times_seconds[stimuli_presentation_id == stimulus_id].values
        for stimulus_id in stimuli_ids_sorted
    }

    psth_dict = {}
    desc = "Calculating PSTH for stimuli"
    for stimuli_id in tqdm(stimuli_ids_sorted, desc=desc, unit=" stimuli processed", disable=not verbose):
        stimulus_presentation_times = stimuli_presentation_times_dict[stimuli_id]
        psth_per_stimuli = calculate_event_psth(
            spike_times_list=spike_times_list,
            event_times_seconds=stimulus_presentation_times,
            bin_width_in_milliseconds=bin_width_in_milliseconds,
            number_of_bins=number_of_bins,
            milliseconds_from_event_to_first_bin=milliseconds_from_event_to_first_bin,
        )
        psth_dict[stimuli_id] = psth_per_stimuli

    return psth_dict, stimuli_presentation_times_dict


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

    psth_dict, stimuli_presentation_times_dict = build_psth_from_nwbfile(
        nwbfile=nwbfile,
        bin_width_in_milliseconds=bin_width_in_milliseconds,
        number_of_bins=number_of_bins,
        milliseconds_from_event_to_first_bin=milliseconds_from_event_to_first_bin,
        verbose=verbose,
    )

    units_table = nwbfile.units
    num_units = units_table["id"].shape[0]
    region_indices = [i for i in range(num_units)]
    units_region = DynamicTableRegion(
        data=region_indices, table=units_table, description="region of units table", name="units_region"
    )

    event_timestamps = [stimuli_presentation_times_dict[stimulus_id] for stimulus_id in psth_dict.keys()]
    data = [psth_dict[stimulus_id] for stimulus_id in psth_dict.keys()]
    condition_labels = [stimulus_id for stimulus_id in psth_dict.keys()]
    condition_indices = [
        [index] * len(stimuli_presentation_times_dict[stimulus_id])
        for index, stimulus_id in enumerate(psth_dict.keys())
    ]

    event_timestamps = np.concatenate(event_timestamps)
    data = np.concatenate(data, axis=1)  # We concatenate across the events axis
    condition_indices = np.concatenate(condition_indices)

    data, event_timestamps, condition_indices = BinnedAlignedSpikes.sort_data_by_event_timestamps(
        data=data,
        event_timestamps=event_timestamps,
        condition_indices=condition_indices,
    )

    binned_aligned_spikes = BinnedAlignedSpikes(
        name=f"BinnedAlignedSpikesToStimulus",
        data=data,
        event_timestamps=event_timestamps,
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
    append: bool = False,
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
