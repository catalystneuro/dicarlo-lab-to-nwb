import numpy as np


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
    if number_of_events is None:
        number_of_events = len(event_times_seconds)

    # We do everything in seconds
    event_times_seconds = np.asarray(event_times_seconds)
    bin_width_in_seconds = bin_width_in_milliseconds / 1000.0
    seconds_from_event_to_first_bin = milliseconds_from_event_to_first_bin / 1000.0

    base_bins_end = bin_width_in_seconds * number_of_bins
    base_bins = np.arange(0, base_bins_end + bin_width_in_seconds, bin_width_in_seconds)

    bins_start = event_times_seconds[:, np.newaxis] + seconds_from_event_to_first_bin
    bins = bins_start + base_bins[np.newaxis, :]

    number_of_units = len(spike_times_list)
    event_psth = np.full(shape=(number_of_units, number_of_events, number_of_bins), fill_value=np.nan)
    for channel_index, spike_times in enumerate(spike_times_list):
        spikes_in_bins = np.zeros((base_bins.size * event_times_seconds.size), dtype="uint8")
        spikes_in_bins[:-1] = np.histogram(spike_times, bins=bins.ravel())[0]
        spikes_to_store = spikes_in_bins.reshape(event_times_seconds.size, base_bins.size)[:, :-1]
        event_psth[channel_index, : len(event_times_seconds)] = spikes_to_store

    return event_psth


def calculate_event_psth_numba(
    spike_times_list,
    event_times_seconds,
    bin_width_in_milliseconds,
    number_of_bins,
    milliseconds_from_event_to_first_bin=0.0,
    number_of_events=None,
):

    if hasattr(calculate_event_psth_numba, "_cached_function"):
        event_psth = calculate_event_psth_numba._cached_function(
            spike_times_list=spike_times_list,
            event_times_seconds=event_times_seconds,
            bin_width_in_milliseconds=bin_width_in_milliseconds,
            number_of_bins=number_of_bins,
            milliseconds_from_event_to_first_bin=milliseconds_from_event_to_first_bin,
            number_of_events=number_of_events,
        )

        return event_psth

    import numba

    @numba.jit(nopython=True)
    def _optimized_calculate_event_psth_numba(
        spike_times_list,
        event_times_seconds,
        bin_width_in_milliseconds,
        number_of_bins,
        milliseconds_from_event_to_first_bin,
        number_of_events,
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

        bins_start = event_times_seconds[:, np.newaxis] + seconds_from_event_to_first_bin
        bins = bins_start + base_bins[np.newaxis, :]

        number_of_units = len(spike_times_list)
        event_psth = np.full(shape=(number_of_units, number_of_events, number_of_bins), fill_value=np.nan)
        for channel_index, spike_times in enumerate(spike_times_list):
            spikes_in_bins = np.zeros((base_bins.size * event_times_seconds.size), dtype="uint8")
            spikes_in_bins[:-1] = np.histogram(spike_times, bins=bins.ravel())[0]
            spikes_to_store = spikes_in_bins.reshape(event_times_seconds.size, base_bins.size)[:, :-1]
            event_psth[channel_index, : len(event_times_seconds)] = spikes_to_store

        return event_psth

    # Cache the compiled function
    calculate_event_psth_numba._cached_function = _optimized_calculate_event_psth_numba

    event_psth = calculate_event_psth_numba._cached_function(
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


def calculate_event_psth_old(
    spike_trains_per_unit,
    event_times,
    bin_width_in_milliseconds,
    number_of_bins,
    milliseconds_from_event_to_first_bin=0.0,
    number_of_events=None,
):

    if number_of_events is None:
        number_of_events = len(event_times)

    event_times = np.asarray(event_times)

    bin_width_in_seconds = bin_width_in_milliseconds / 1000.0
    seconds_from_event_to_first_bin = milliseconds_from_event_to_first_bin / 1000.0
    base_bins_end = bin_width_in_seconds * number_of_bins
    base_bins = np.linspace(0, base_bins_end, number_of_bins + 1, endpoint=True)
    number_of_units = len(spike_trains_per_unit)

    bins_start = event_times[:, np.newaxis] + seconds_from_event_to_first_bin
    bins = bins_start + base_bins[np.newaxis, :]

    event_psth = np.full(shape=(number_of_units, number_of_events, number_of_bins), fill_value=np.nan)
    for channel_index, channel_id in enumerate(spike_trains_per_unit.keys()):
        spikes_times = spike_trains_per_unit[channel_id]
        for repetition, event_bins in enumerate(bins):
            spikes_in_bins = np.histogram(spikes_times, bins=event_bins)[0]
            event_psth[channel_index, repetition] = spikes_in_bins

    return event_psth
