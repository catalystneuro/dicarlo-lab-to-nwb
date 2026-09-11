import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from dicarlo_lab_to_nwb.mworks_library.mwk2reader import MWKFile


def equal_for_all_trials(events):
    return all(e.data == events[0].data for e in events)


def listify_events(events):
    return list(e.data for e in events)


def get_events(event_file, name):
    data = {
        "code": [],
        "name": [],
        "time": [],
        "data": [],
    }
    for event in event_file.get_events_iter(codes=name):
        data["code"].append(event.code)
        data["name"].append(event_file.codec[event.code])
        data["time"].append(event.time)
        data["data"].append(event.data)
    data = pd.DataFrame(data)
    data = data.sort_values(by="time").reset_index(drop=True)
    return data


def extract_images_from_sdu(data):
    images = []
    try:
        for item in data:
            try:
                if item["type"] == "image":
                    images.append(item)
            except:
                pass
    except:
        pass
    return images


def extract_image_from_sdu(data):
    images = extract_images_from_sdu(data)
    assert len(images) == 1
    return images.pop()


def get_display_updates(event_file) -> pd.DataFrame:
    """
    Return one row per stimulus that reached the screen, from the `#stimDisplayUpdate` events.

    An image is taken from every display update that contains it. A video spans many consecutive display
    updates, so only the first one of each run is taken.

    Returns
    -------
    pd.DataFrame
        Columns `time` (MWorks time in microseconds), `stimulus_type`, `filename` and `file_hash`.
    """
    rows = []
    in_video = False
    for event in event_file.get_events_iter(codes=["#stimDisplayUpdate"]):
        items = [item for item in event.data if isinstance(item, dict)]
        video = next((item for item in items if item.get("type") == "video"), None)
        if video is not None:
            if not in_video:
                # No hash for videos yet in MWorks 0.12
                rows.append((event.time, "video", Path(video["filename"]).name, ""))
            in_video = True
            continue
        in_video = False

        image = next((item for item in items if item.get("type") in ("image", "audio")), None)
        if image is not None:
            if image["type"] == "audio":
                print("Audio stimulus detected. Not supported yet")
            rows.append((event.time, image["type"], Path(image["filename"]).name, image.get("file_hash", "")))

    return pd.DataFrame(rows, columns=["time", "stimulus_type", "filename", "file_hash"])


def match_display_updates_to_presentations(
    presentation_times: np.ndarray, display_times: np.ndarray, max_lag_us: float = 100_000.0
) -> np.ndarray:
    """
    Pair every display update with the `stimulus_presented` event nearest to it in time.

    The display update usually follows its presentation event by a few milliseconds, but it is occasionally
    logged just before it, so the nearest event is used instead of the preceding one.

    Returns
    -------
    np.ndarray
        For each display update, the index of its presentation event.
    """
    order = np.searchsorted(presentation_times, display_times)
    previous = np.clip(order - 1, 0, len(presentation_times) - 1)
    following = np.clip(order, 0, len(presentation_times) - 1)
    following_is_nearer = np.abs(presentation_times[following] - display_times) < np.abs(
        presentation_times[previous] - display_times
    )
    nearest = np.where(following_is_nearer, following, previous)

    lags_us = np.abs(presentation_times[nearest] - display_times)
    if (lags_us > max_lag_us).any():
        raise ValueError(
            f"{int((lags_us > max_lag_us).sum())} display updates have no stimulus_presented event within "
            f"{max_lag_us / 1000:.0f} ms."
        )
    if len(np.unique(nearest)) != len(nearest):
        raise ValueError("Two display updates were paired with the same stimulus_presented event.")

    return nearest


def pair_pulses_with_display_updates(
    display_times_us: np.ndarray, pulse_frames: np.ndarray, sampling_frequency: float, tolerance_ms: float = 50.0
) -> np.ndarray:
    """
    Pair every sample-on pulse (Intan clock) with the display update it belongs to (MWorks clock).

    The offset between the two clocks is taken from the differences between the first pulses and the display
    updates: the candidate under which most of the first pulses land on a display update wins. Picking the
    most common difference alone is not enough, because stimuli within a trial are evenly spaced and an
    offset shifted by one stimulus also lines up most of them. Each pulse is then paired with the display
    update nearest to its predicted MWorks time, and the prediction is refit with a line through the paired
    times to absorb the drift between the clocks (a few parts per million, tens of milliseconds over a
    session). A pulse with no display update within
    `tolerance_ms` (a stimulus that was announced but never drawn) is left unpaired, and so is a display update
    with no pulse (outside the Intan recording).

    Returns
    -------
    np.ndarray
        For each pulse, the index of its display update, or -1 when it has none.
    """
    display_times_s = np.asarray(display_times_us, dtype=float) / 1e6
    pulse_times_s = np.asarray(pulse_frames, dtype=float) / sampling_frequency
    tolerance_s = tolerance_ms / 1000.0

    def pair_nearest(predicted_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.clip(np.searchsorted(display_times_s, predicted_s), 1, len(display_times_s) - 1)
        previous_is_nearer = np.abs(display_times_s[order - 1] - predicted_s) < np.abs(
            display_times_s[order] - predicted_s
        )
        nearest = np.where(previous_is_nearer, order - 1, order)
        return nearest, np.abs(display_times_s[nearest] - predicted_s) <= tolerance_s

    first_pulses_s = pulse_times_s[:200]
    differences_s = (display_times_s[np.newaxis, :] - first_pulses_s[:, np.newaxis]).ravel()
    binned_differences, counts = np.unique(np.round(differences_s / 0.01), return_counts=True)
    candidate_offsets_s = binned_differences[np.argsort(counts)[::-1][:20]] * 0.01
    clock_offset_s = max(candidate_offsets_s, key=lambda offset_s: pair_nearest(first_pulses_s + offset_s)[1].sum())

    slope, intercept = 1.0, clock_offset_s
    for _ in range(2):
        nearest, is_paired = pair_nearest(slope * pulse_times_s + intercept)
        slope, intercept = np.polyfit(pulse_times_s[is_paired], display_times_s[nearest[is_paired]], 1)

    pulse_to_display = np.where(is_paired, nearest, -1)
    paired_displays = pulse_to_display[pulse_to_display >= 0]
    if len(np.unique(paired_displays)) != len(paired_displays):
        raise ValueError("Two sample-on pulses were paired with the same display update.")

    return pulse_to_display


def dump_events_rsvp(SAMPLING_FREQUENCY_HZ, filename, photodiode_filepath, digi_event_filepath, output_dir: str = "./"):
    print(f"Sampling rate: {SAMPLING_FREQUENCY_HZ}, {filename}")

    event_file = MWKFile(filename)
    event_file.open()

    # Variables we'd like to fetch data for
    names = [
        "trial_start_line",
        "correct_fixation",
        "stimulus_presented",
        "#stimDisplayUpdate",
        "stim_on_time",
        "stim_off_time",
        "stim_on_delay",
        "stimulus_size",  # for 'normalizers'
        "stimulus_size_deg",  # for 'gestalt_control' & 'SFM_*'
        "fixation_window_size",
        "fixation_point_size_min",
    ]
    data = get_events(event_file=event_file, name=names)

    ###########################################################################
    # Create a dict to store output information
    ###########################################################################
    # Check if each entry in 'names' is in data.name and filter out entries with empty lists
    filtered_names = [name for name in names if not data[data.name == name].empty]
    # check if stimulus_size_deg is in the list of names
    if "stimulus_size_deg" in filtered_names:
        output = {
            "stim_on_time_ms": data[data.name == "stim_on_time"]["data"].values[-1] / 1000.0,
            "stim_off_time_ms": data[data.name == "stim_off_time"]["data"].values[-1] / 1000.0,
            "stim_on_delay_ms": data[data.name == "stim_on_delay"]["data"].values[-1] / 1000.0,
            "stimulus_size_degrees": data[data.name == "stimulus_size_deg"]["data"].values[
                -1
            ],  # for 'gestalt_control' & 'SFM_*'
            "fixation_window_size_degrees": data[data.name == "fixation_window_size"]["data"].values[-1],
            "fixation_point_size_degrees": data[data.name == "fixation_point_size_min"]["data"].values[-1],
        }
    elif "stimulus_size" in filtered_names:
        output = {
            "stim_on_time_ms": data[data.name == "stim_on_time"]["data"].values[-1] / 1000.0,
            "stim_off_time_ms": data[data.name == "stim_off_time"]["data"].values[-1] / 1000.0,
            "stim_on_delay_ms": data[data.name == "stim_on_delay"]["data"].values[-1] / 1000.0,
            "stimulus_size_degrees": data[data.name == "stimulus_size"]["data"].values[-1],  # for 'normalizers'
            "fixation_window_size_degrees": data[data.name == "fixation_window_size"]["data"].values[-1],
            "fixation_point_size_degrees": data[data.name == "fixation_point_size_min"]["data"].values[-1],
        }
    else:
        output = {
            "stim_on_time_ms": data[data.name == "stim_on_time"]["data"].values[-1] / 1000.0,
            "stim_off_time_ms": data[data.name == "stim_off_time"]["data"].values[-1] / 1000.0,
            "stim_on_delay_ms": data[data.name == "stim_on_delay"]["data"].values[-1] / 1000.0,
            "stimulus_size_degrees": 8.0,  # for 'normalizers'
            "fixation_window_size_degrees": data[data.name == "fixation_window_size"]["data"].values[-1],
            "fixation_point_size_degrees": data[data.name == "fixation_point_size_min"]["data"].values[-1],
        }

    ###########################################################################
    # Add column in data to indicate whether stimulus was first in trial or not
    ###########################################################################
    data["first_in_trial"] = False
    # Filter data to only get `trial_start_line` and `stimulus_presented` information
    df = data[(data.name == "trial_start_line") | ((data.name == "stimulus_presented") & (data.data != -1))]
    # Extract `time` for the first `stimulus_presented` (which is right after `trial_start_line` has been pulsed)
    first_in_trial_times = [
        df.time.values[i]
        for i in range(1, len(df))
        if ((df.name.values[i - 1] == "trial_start_line") and (df.name.values[i] == "stimulus_presented"))
    ]
    data["first_in_trial"] = data["time"].apply(lambda x: True if x in first_in_trial_times else False)

    ###########################################################################
    # Extract stimulus presentation order and fixation information
    ###########################################################################
    stimulus_presented_df = data[data.name == "stimulus_presented"].reset_index(drop=True)
    correct_fixation_df = data[data.name == "correct_fixation"].reset_index(drop=True)

    # In case one there is one extra stimulus event but not fixation, use this
    if len(correct_fixation_df) < len(stimulus_presented_df):
        stimulus_presented_df = stimulus_presented_df[: len(correct_fixation_df)]

    assert len(stimulus_presented_df) == len(correct_fixation_df)

    # Drop `empty` data (i.e. -1) before the experiment actually began and after it had already ended
    correct_fixation_df = correct_fixation_df[stimulus_presented_df.data != -1].reset_index(drop=True)
    stimulus_presented_df = stimulus_presented_df[stimulus_presented_df.data != -1].reset_index(drop=True)
    # Add `first_in_trial` info to other data frame too
    correct_fixation_df["first_in_trial"] = stimulus_presented_df["first_in_trial"]

    ###########################################################################
    # Add column to indicate order in trial (1 2 3 1 2 3 etc.)
    ###########################################################################
    assert stimulus_presented_df.iloc[0].first_in_trial
    stimulus_presented_df["stimulus_order_in_trial"] = 0
    counter = 1
    for index, row in stimulus_presented_df.iterrows():
        if row["first_in_trial"]:
            counter = 1
        stimulus_presented_df.at[index, "stimulus_order_in_trial"] = counter
        counter += 1
    correct_fixation_df["stimulus_order_in_trial"] = stimulus_presented_df["stimulus_order_in_trial"]

    ###########################################################################
    # Pair the presentations with the stimuli that reached the screen
    ###########################################################################
    # Presentation events that were never displayed (e.g. logged before the task started) are dropped. Pairing by
    # time instead of by position keeps the stimulus index, the file name and the times of each row together
    display_updates_df = get_display_updates(event_file)
    displayed_indices = match_display_updates_to_presentations(
        presentation_times=stimulus_presented_df.time.to_numpy(), display_times=display_updates_df.time.to_numpy()
    )
    never_displayed = np.setdiff1d(np.arange(len(stimulus_presented_df)), displayed_indices)
    if len(never_displayed) > 0:
        print(f"Dropping {len(never_displayed)} stimulus_presented events that were never displayed: {never_displayed}")
    stimulus_presented_df = stimulus_presented_df.iloc[displayed_indices].reset_index(drop=True)
    correct_fixation_df = correct_fixation_df.iloc[displayed_indices].reset_index(drop=True)
    stimulus_presented_df["stimulus_type"] = display_updates_df["stimulus_type"].to_numpy()
    stimulus_presented_df["filename"] = display_updates_df["filename"].to_numpy()
    stimulus_presented_df["image_hash"] = display_updates_df["file_hash"].to_numpy()

    ###########################################################################
    # Read sample on file
    ###########################################################################
    fid = open(digi_event_filepath, "r")
    filesize = os.path.getsize(digi_event_filepath)  # in bytes
    num_samples = filesize // 2  # uint16 = 2 bytes
    digital_in = np.fromfile(fid, "uint16", num_samples)
    fid.close()

    (samp_on,) = np.nonzero(digital_in[:-1] < digital_in[1:])  # Look for 0->1 transitions
    samp_on = samp_on + 1  # Previous line returns indexes of 0s seen before spikes, but we want indexes of first spikes

    # Pair each sample-on pulse with its displayed stimulus by time. Pulses of stimuli that were never drawn and
    # displayed stimuli outside the Intan recording are dropped
    pulse_to_display = pair_pulses_with_display_updates(
        display_times_us=display_updates_df.time.to_numpy(),
        pulse_frames=samp_on,
        sampling_frequency=SAMPLING_FREQUENCY_HZ,
    )
    if (pulse_to_display < 0).any():
        print(f"Warning: dropping {int((pulse_to_display < 0).sum())} sample-on pulses with no displayed stimulus")
    samp_on = samp_on[pulse_to_display >= 0]
    displays_with_pulse = pulse_to_display[pulse_to_display >= 0]
    if len(displays_with_pulse) < len(stimulus_presented_df):
        print(
            f"Warning: keeping the {len(displays_with_pulse)} of {len(stimulus_presented_df)} displayed stimuli that have a pulse"
        )
    stimulus_presented_df = stimulus_presented_df.iloc[displays_with_pulse].reset_index(drop=True)
    correct_fixation_df = correct_fixation_df.iloc[displays_with_pulse].reset_index(drop=True)

    assert len(samp_on) == len(stimulus_presented_df)

    ###########################################################################
    # Read photodiode file
    ###########################################################################
    fid = open(photodiode_filepath, "r")
    filesize = os.path.getsize(photodiode_filepath)  # in bytes
    num_samples = filesize // 2  # uint16 = 2 bytes
    v = np.fromfile(fid, "uint16", num_samples)
    fid.close()

    # Convert to volts (use this if the data file was generated by Recording Controller)
    # v = (v - 32768) * 0.0003125
    v = v * 0.195

    upper_quantile = np.quantile(v, 0.75)
    lower_quantile = np.quantile(v, 0.25)
    v_range = upper_quantile - lower_quantile

    thresh = v_range * 0.5 + lower_quantile

    v_digi = np.zeros(np.size(v))
    v_digi[v > thresh] = 1
    (v_on,) = np.nonzero(v_digi[:-1] < v_digi[1:])  # Look for 0->1 transitions
    v_on = v_on + 1  # Previous line returns indexes of 0s seen before spikes, but we want indexes of first spikes
    photodiode_on = np.asarray([min(v_on[(v_on >= s) & (v_on < (s + 100_000))]) for s in samp_on])

    assert len(photodiode_on) == len(stimulus_presented_df)

    # Convert both times to microseconds to match MWorks
    photodiode_on = photodiode_on * 1_000_000 / SAMPLING_FREQUENCY_HZ  # in us
    samp_on = samp_on * 1_000_000 / SAMPLING_FREQUENCY_HZ  # in us

    ###########################################################################
    # Correct the times
    ###########################################################################
    corrected_time = stimulus_presented_df.time.values.tolist() + (photodiode_on - samp_on)  # Both are in microseconds
    print(f"Delay recorded on photodiode is {np.mean(photodiode_on - samp_on) / 1000.:.2f} ms on average")

    stimulus_presented_df["time"] = corrected_time
    correct_fixation_df["time"] = corrected_time

    # Print any times differences between digital signal and photodiode that are atrociously huge (>40ms)
    for i, x in enumerate(photodiode_on - samp_on):
        if x / 1000.0 > 40:
            print(f"Warning: Sample {i} has delay of {x / 1000.} ms")

    ###########################################################################
    # Get eye data
    ###########################################################################
    eye_h, eye_v, eye_time = [], [], []
    pupil_size, pupil_time = [], []
    for t in stimulus_presented_df.time.values:
        t1 = int(t - 50 * 1000.0)  # Start time (ms)
        t2 = int(t + (output["stim_on_time_ms"] + 50) * 1000.0)  # Stop time (ms)
        h = [event.data for event in event_file.get_events_iter(codes=["eye_h"], time_range=[t1, t2])]
        v = [event.data for event in event_file.get_events_iter(codes=["eye_v"], time_range=[t1, t2])]
        time = [(event.time - t) / 1000.0 for event in event_file.get_events_iter(codes=["eye_v"], time_range=[t1, t2])]
        assert len(h) == len(v)
        assert len(time) == len(h)
        eye_h.append(h)
        eye_v.append(v)
        eye_time.append(time)

    assert len(eye_h) == len(stimulus_presented_df)
    event_file.close()

    ###########################################################################
    # Save output
    ###########################################################################
    output["stimulus_presented"] = stimulus_presented_df.data.values.tolist()
    output["fixation_correct"] = correct_fixation_df.data.values.tolist()
    output["stimulus_order_in_trial"] = stimulus_presented_df.stimulus_order_in_trial.values.tolist()
    output["samp_on_us"] = np.array(samp_on).astype(int)  # Convert list to numpy array first
    output["photodiode_on_us"] = np.array(photodiode_on).astype(int)  # This is already a numpy array
    output["rig_delays_us"] = (photodiode_on - np.array(samp_on)).astype(int)
    output["stimulus_type"] = stimulus_presented_df["stimulus_type"]
    output["stimulus_filename"] = stimulus_presented_df["filename"]
    output["image_hash"] = stimulus_presented_df["image_hash"]

    # save to processed folder
    output = pd.DataFrame(output)
    output_filepath = os.path.join(
        output_dir, str(filename).split("/")[-1][:-5] + "_mwk.csv"
    )  # -5 in filename to delete the .mwk2 extension
    output.to_csv(output_filepath, index=False)

    ###########################################################################
    # Repetitions
    ###########################################################################
    selected_indexes = correct_fixation_df[correct_fixation_df.data == 1]["data"].index.tolist()
    correct_trials = np.asarray(stimulus_presented_df.data.values.tolist())[selected_indexes]
    num_repetitions = np.asarray(
        [
            len(correct_trials[correct_trials == stimulus])
            for stimulus in np.unique(stimulus_presented_df.data.values.tolist())
        ]
    )
    print(f"... {min(num_repetitions)} repeats, range is {np.unique(num_repetitions)}")

    return output_filepath


if __name__ == "__main__":
    # folder structure
    # todays_path = Path(root_dir) / subjectName / stimulusSet / sessionDate
    todays_path = Path("/Users/yoon/raw_data/Apollo/test/20240701/normalizers_240701_131011")

    print(f"Processing {todays_path}...\n")
    assert todays_path.is_dir(), f"Folder {todays_path} does not exist"

    # iterate through each folder in the session folder. There should be one folder per experiment
    if todays_path.is_dir() and not todays_path.name.startswith(".") and not todays_path.name.startswith("processed"):

        # sampling_freq = 20000
        sampling_freq = 30000

        # get the mworks file ending with .mwk2 (take the first in list)
        mworks_file = list(todays_path.glob("*.mwk2"))[0]
        photodiode_file = todays_path / "board-ANALOG-IN-1.dat"
        digi_event_file = todays_path / "board-DIGITAL-IN-02.dat"

        # print(f"MWorks parsing : {mworks_file}")

        # save spike times as a .npy file at 'output_dir'
        output_dir = todays_path / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        # run parser
        from dicarlo_lab_to_nwb.mworks_library import mwk_rsvp

        mwk_rsvp.dump_events_rsvp(sampling_freq, mworks_file, photodiode_file, digi_event_file, output_dir)
