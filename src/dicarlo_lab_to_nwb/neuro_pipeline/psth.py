from pathlib import Path
import numpy as np
from ndx_binned_spikes import BinnedAlignedSpikes
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.core import DynamicTable, VectorData
from tqdm.auto import tqdm


def calculate_event_psth(
    spike_times_list,
    stimulus_event_times,
    bin_width_ms: float = 10.0,
    psth_start_time_ms: float = -200.0,
    psth_end_time_ms: float = 400.0,
    number_of_events=None,
):
    """
    Calculate Peri-Stimulus Time Histogram (PSTH) for given events.

    Parameters
    ----------
    spike_times_list : list of arrays
        List where each element is an array of spike times (in seconds) for a single unit.
    stimulus_event_times : array-like
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

    stimulus_event_times = np.asarray(stimulus_event_times)
    if number_of_events is None:
        number_of_events = len(stimulus_event_times)

    if hasattr(calculate_event_psth, "_cached_function"):
        event_psth = calculate_event_psth._cached_function(
            spike_times_list=spike_times_list,
            event_times_seconds=stimulus_event_times,
            bin_width_ms=bin_width_ms,
            psth_start_time_ms=psth_start_time_ms,
            psth_end_time_ms=psth_end_time_ms,
            number_of_events=number_of_events,
        )

        return event_psth

    import numba
    from numba import prange

    @numba.jit(nopython=True, parallel=True)
    def _optimized_calculate_event_psth(
        spike_times_list,
        event_times_seconds,
        bin_width_ms,
        psth_start_time_ms,
        psth_end_time_ms,
        number_of_events,
    ):
        # We do everything in seconds
        bin_width_in_seconds = bin_width_ms / 1000.0
        seconds_from_event_to_first_bin = psth_start_time_ms / 1000.0

        psth_duration_s = (psth_end_time_ms - psth_start_time_ms) / 1000.0
        number_of_bins = int(psth_duration_s / bin_width_in_seconds)
        # max_repetitions = stimuli_presentation_id.value_counts().max()

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
        event_times_seconds=stimulus_event_times,
        bin_width_ms=bin_width_ms,
        psth_start_time_ms=psth_start_time_ms,
        psth_end_time_ms=psth_end_time_ms,
        number_of_events=number_of_events,
    )

    return event_psth




def build_stimulus_psth_from_nwbfile(
    nwbfile: NWBFile,
    bin_width_ms: float = 10.0,
    psth_start_time_ms: float = -200.0,
    psth_end_time_ms: float = 400.0,
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
    trial_psth_dict : dict
        Dictionary where keys are stimulus IDs and values are arrays of PSTH counts.
    stimulus_presentation_times_dict : dict
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

    mworks_trials_dataframe = nwbfile.trials.to_dataframe()
    mworks_presentation_times_seconds = mworks_trials_dataframe["start_time"]
    mworks_presentation_id = mworks_trials_dataframe["stimulus_presented"]
    mworks_stimuli_ids = mworks_presentation_id.unique()

    ## associated stimuli meta are stored in the stimulus table in the following way
    # image_obj_list = nwbfile.stimulus["stimuli"].order_of_images
    # image_data = image_obj_list[i].data
    # image_filenames = image_obj_list.name
    # hash = image_obj_list[i].description

    # We also sort the stimuli by their id's used in MWorks
    stimuli_ids_sorted = sorted(mworks_stimuli_ids)

    stimulus_presentation_times_dict = {
        stimulus_id: mworks_presentation_times_seconds[mworks_presentation_id == stimulus_id].values
        for stimulus_id in stimuli_ids_sorted
    }

    stimulus_psth_dict = {}
    desc = "Calculating PSTH for stimuli"
    for stimuli_id in tqdm(stimuli_ids_sorted, desc=desc, unit=" stimuli processed", disable=not verbose):
        stimulus_presentation_times = stimulus_presentation_times_dict[stimuli_id]
        psth_per_stimuli = calculate_event_psth(
            spike_times_list=spike_times_list,
            stimulus_event_times=stimulus_presentation_times,
            bin_width_ms=bin_width_ms,
            psth_start_time_ms=psth_start_time_ms,
            psth_end_time_ms=psth_end_time_ms,
        )
        stimulus_psth_dict[stimuli_id] = psth_per_stimuli

    return stimulus_psth_dict, stimulus_presentation_times_dict


def add_session_psth_to_nwbfile(
    nwbfile: NWBFile,
    stimulus_psth_dict: dict,
    stimulus_presentation_times_dict: dict,
    bin_width_ms: float = 10.0,
    psth_start_time_ms: float = -200.0,
    psth_end_time_ms: float = 400.0,
) -> np.ndarray:
    
    ## Create session PSTH
    psth_duration_ms = psth_end_time_ms - psth_start_time_ms
    number_of_bins = int(psth_duration_ms / bin_width_ms)
    # find maximum number of values associated to each key value in a dict, 'stimulus_presentation_times_dict'
    max_repetitions = max(len(times) for times in stimulus_presentation_times_dict.values())

    units_table = nwbfile.units
    number_of_units = units_table["id"].shape[0]
    number_of_stimuli = len(stimulus_presentation_times_dict)

    session_psth = np.full(
        shape=(number_of_units, number_of_stimuli, max_repetitions, number_of_bins), fill_value=np.nan
    )
    for stimuli_index, stimuli_psth in enumerate(tqdm(stimulus_psth_dict.values(), desc="Packaging PSTH")):
        reps_for_this_stim = stimuli_psth.shape[1]
        session_psth[:, stimuli_index, :reps_for_this_stim, ...] = stimuli_psth # units x reps x bins

    # Question: how to associate dimension labels to the session_psth array?
    psth_ts = TimeSeries(
        name="PSTH",
        data=session_psth,
        unit='spikes/bin',
        description="Peri-Stimulus Time Histogram (PSTH): sites (or units) x stimuli x repeats x timebins",
        timestamps=np.arange(psth_start_time_ms, psth_end_time_ms, bin_width_ms) / 1000, # in seconds
    )
    ## Unit labels and stimulus filenames can be retrieved in the following manner
    # # unit ID
    # units_table = nwbfile.units
    # df = units_table.to_dataframe()
    # unit_labels = df["unit_name"].values
    # print(f"units: {unit_labels}")

    # # stimulus ID
    # image_obj_list = nwbfile.stimulus["stimuli"].order_of_images
    # image_filenames = [image_obj.name for image_obj in image_obj_list]
    # print(f"stimuli: {image_filenames}")

    nwbfile.add_acquisition(psth_ts)

   

def build_binned_aligned_spikes_from_nwbfile(
    nwbfile: NWBFile,
    stimulus_psth_dict: dict,
    stimulus_presentation_times_dict: dict,
    bin_width_ms: float = 10.0,
    psth_start_time_ms: float = -200.0,
    psth_end_time_ms: float = 400.0,
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

    # psth_dict, stimuli_presentation_times_dict = build_trial_psth_from_nwbfile_2(
    #     nwbfile=nwbfile,
    #     bin_width_ms=bin_width_ms,
    #     psth_start_time_ms=psth_start_time_ms,
    #     psth_end_time_ms=psth_end_time_ms,
    #     verbose=verbose,
    # )

    units_table = nwbfile.units
    num_units = units_table["id"].shape[0]
    region_indices = [i for i in range(num_units)]
    units_region = DynamicTableRegion(
        data=region_indices, table=units_table, description="region of units table", name="units_region"
    )

    event_timestamps = [stimulus_presentation_times_dict[stimulus_id] for stimulus_id in stimulus_psth_dict.keys()]
    data = [stimulus_psth_dict[stimulus_id] for stimulus_id in stimulus_psth_dict.keys()]
    condition_labels = [stimulus_id for stimulus_id in stimulus_psth_dict.keys()]
    condition_indices = [
        [index] * len(stimulus_presentation_times_dict[stimulus_id])
        for index, stimulus_id in enumerate(stimulus_psth_dict.keys())
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
        bin_width_in_milliseconds=bin_width_ms,
        milliseconds_from_event_to_first_bin=psth_start_time_ms,
        units_region=units_region,
    )

    #  ## Create session PSTH
    # interfaces = nwbfile.processing["ecephys"].data_interfaces.values()
    # is_binned_spikes = lambda interface: interface.data_type == "BinnedAlignedSpikes"
    # valid_interfaces = [interface for interface in interfaces if is_binned_spikes(interface)]
    # psth_dict = {interface.name: interface.data for interface in valid_interfaces}

    # psth_duration_ms = psth_end_time_ms - psth_start_time_ms
    # number_of_bins = int(psth_duration_ms / bin_width_ms)
    # max_repetitions = mworks_presentation_id.value_counts().max()

    # number_of_units = len(unit_ids_sorted)
    # number_of_stimuli = len(stimuli_presentation_times_dict)

    # session_psth = np.full(
    #     shape=(number_of_units, number_of_stimuli, max_repetitions, number_of_bins), fill_value=np.nan
    # )
    # for stimuli_index, stimuli_psth in enumerate(tqdm(psth_dict.values(), desc="Packaging PSTH")):
    #     reps_for_this_stim = stimuli_psth.shape[1]
    #     session_psth[:, stimuli_index, :reps_for_this_stim, ...] = stimuli_psth # units x reps x bins

    return binned_aligned_spikes


def write_binned_spikes_to_nwbfile(
    nwbfile_path: Path | str,
    bin_width_ms: float,
    psth_start_time_ms: float = -200.0,
    psth_end_time_ms: float = 400.0,
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

        stim_psth_dict, stim_presentation_times_dict = build_stimulus_psth_from_nwbfile(
        nwbfile=nwbfile,
        bin_width_ms=bin_width_ms,
        psth_start_time_ms=psth_start_time_ms,
        psth_end_time_ms=psth_end_time_ms,
        verbose=verbose,
    )
        binned_aligned_spikes = build_binned_aligned_spikes_from_nwbfile(
            nwbfile=nwbfile,
            stimulus_psth_dict=stim_psth_dict,
            stimulus_presentation_times_dict=stim_presentation_times_dict,
            bin_width_ms=bin_width_ms,
            psth_start_time_ms=psth_start_time_ms,
            psth_end_time_ms=psth_end_time_ms,
            verbose=verbose,
        )

        ecephys_processing_module = nwbfile.create_processing_module(
            name="ecephys",
            description="Intermediate data derived from extracellular electrophysiology recordings such as PSTHs.",
        )

        ecephys_processing_module.add(binned_aligned_spikes)

        add_session_psth_to_nwbfile(
            nwbfile=nwbfile,
            stimulus_psth_dict=stim_psth_dict,
            stimulus_presentation_times_dict=stim_presentation_times_dict,
            bin_width_ms=bin_width_ms,
            psth_start_time_ms=psth_start_time_ms,
            psth_end_time_ms=psth_end_time_ms,
        )

        # nwbfile.add_scratch(
        #     name="session_psth",
        #     data=session_psth,
        #     description="Session psth [units x stimuli x reps x timebins]",
        # )
        
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
