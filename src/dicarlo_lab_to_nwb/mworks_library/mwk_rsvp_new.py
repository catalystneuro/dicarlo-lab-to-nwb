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


def dump_events_rsvp(SAMPLING_FREQUENCY_HZ, filename, photodiode_filepath, digi_event_filepath, output_dir: str = "./"):
    print(f"Sampling rate: {SAMPLING_FREQUENCY_HZ}, {filename}")

    event_file = MWKFile(filename)
    event_file.open()
    names = [
        "trial_start_line",
        "correct_fixation",
        "stimulus_presented",
        "stim_on_time",
        "stim_off_time",
        "stim_on_delay",
        "stimulus_size_deg",
        "fixation_window_size",
        "fixation_point_size_min",
    ]

    ###########################################################################
    # Create a dict to store output information
    ###########################################################################
    data = get_events(event_file=event_file, name=names)
    # if there is no stimulus_size_deg, then it is an older version of the RSVP mworks code
    if "stimulus_size_deg" not in data.name.values:
        print("Warning: Using older version of RSVP mworks code")
        # Variables we'd like to fetch data for
        names = [
            "trial_start_line",
            "correct_fixation",
            "stimulus_presented",
            "stim_on_time",
            "stim_off_time",
            "stim_on_delay",
            "stimulus_size",  # for older versions of RSVP mworks code
            "fixation_window_size",
            "fixation_point_size_min",
        ]
        # data = get_events(event_file=event_file, name=names)
        # event_file.close()

        output = {
            "stim_on_time_ms": data[data.name == "stim_on_time"]["data"].values[-1] / 1000.0,
            "stim_off_time_ms": data[data.name == "stim_off_time"]["data"].values[-1] / 1000.0,
            "stim_on_delay_ms": data[data.name == "stim_on_delay"]["data"].values[-1] / 1000.0,
            "stimulus_size_degrees": data[data.name == "stimulus_size"]["data"].values[-1],  # for older versions
            # 'stimulus_size_degrees': data[data.name == 'stimulus_size']['data'].values[-1], # for 'normalizers'
            # 'fixation_window_size_degrees': data[data.name == 'fixation_window_size']['data'].values[-1],
            # 'fixation_point_size_degrees': data[data.name == 'fixation_point_size_min']['data'].values[-1],
        }
    else:

        output = {
            "stim_on_time_ms": data[data.name == "stim_on_time"]["data"].values[-1] / 1000.0,
            "stim_off_time_ms": data[data.name == "stim_off_time"]["data"].values[-1] / 1000.0,
            "stim_on_delay_ms": data[data.name == "stim_on_delay"]["data"].values[-1] / 1000.0,
            "stimulus_size_degrees": data[data.name == "stimulus_size_deg"]["data"].values[
                -1
            ],  # for 'gestalt_control' & 'SFM_*'
            # 'fixation_window_size_degrees': data[data.name == 'fixation_window_size']['data'].values[-1],
            # 'fixation_point_size_degrees': data[data.name == 'fixation_point_size_min']['data'].values[-1],
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

    # If you have one extra stimulus event but not fixation, use this
    if len(correct_fixation_df) < len(stimulus_presented_df):
        stimulus_presented_df = stimulus_presented_df[: len(correct_fixation_df)]
    # stimulus_presented_df = stimulus_presented_df[:len(correct_fixation_df)]

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
    stimulus_presented_df["stimulus_order_in_trial"] = ""
    counter = 1
    for index, row in stimulus_presented_df.iterrows():
        if row["first_in_trial"]:
            counter = 1
        stimulus_presented_df.at[index, "stimulus_order_in_trial"] = counter
        counter += 1
    correct_fixation_df["stimulus_order_in_trial"] = stimulus_presented_df["stimulus_order_in_trial"]

    ###########################################################################
    # Read sample on file
    ###########################################################################
    fid = open(digi_event_filepath, "rb")
    filesize = os.path.getsize(filename)  # in bytes
    num_samples = filesize // 2  # uint16 = 2 bytes
    digital_in = np.fromfile(fid, "uint16", num_samples)
    fid.close()

    (samp_on,) = np.nonzero(digital_in[:-1] < digital_in[1:])  # Look for 0->1 transitions
    samp_on = samp_on + 1  # Previous line returns indexes of 0s seen before spikes, but we want indexes of first spikes

    if len(stimulus_presented_df) > len(samp_on):
        print(f"Warning: Trimming MWorks files as ({len(stimulus_presented_df)} > {len(samp_on)})")
        stimulus_presented_df = stimulus_presented_df[: len(samp_on)]
        correct_fixation_df = correct_fixation_df[: len(samp_on)]

    # print(f"samp_on: {len(samp_on)}")
    # print(f"stimulus_presented_df: {len(stimulus_presented_df)}")
    # samp_on = samp_on[:len(correct_fixation_df)]   # If you have one extra stimulus event but not fixation, use this
    if len(correct_fixation_df) < len(samp_on):
        samp_on = samp_on[: len(correct_fixation_df)]

    assert len(samp_on) == len(stimulus_presented_df)

    ###########################################################################
    # Read photodiode file
    ###########################################################################
    fid = open(photodiode_filepath, "rb")
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

    # plot the photodiode signal for the first 10 seconds
    # import matplotlib.pyplot as plt
    # plt.plot(v[int(10*SAMPLING_FREQUENCY_HZ) : int(60*SAMPLING_FREQUENCY_HZ)])
    # plt.plot([int(10*SAMPLING_FREQUENCY_HZ), int(60*SAMPLING_FREQUENCY_HZ)], [thresh, thresh], 'r--')
    # plt.show()

    v_digi = np.zeros(np.size(v))
    v_digi[v > thresh] = 1
    (v_on,) = np.nonzero(v_digi[:-1] < v_digi[1:])  # Look for 0->1 transitions
    v_on = v_on + 1  # Previous line returns indexes of 0s seen before spikes, but we want indexes of first spikes
    photodiode_on = np.asarray([min(v_on[(v_on >= s) & (v_on < (s + 100_000))]) for s in samp_on])

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

    # # Print any times differences between digital signal and photodiode that are atrociously huge (>40ms)
    # for i, x in enumerate(photodiode_on - samp_on):
    #     if x / 1000. > 40:
    #         print(f'Warning: Sample {i} has delay of {x / 1000.} ms')

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
        # t1 = int(t - 1000 * 1000.)  # Start time (ms)
        # t2 = int(t + (output['stim_on_time_ms'] + 2000) * 1000.)  # Stop time (ms)
        # p = [event.data for event in event_file.get_events_iter(codes=['pupil_size_r'], time_range=[t1, t2])]
        # p_time = [(event.time - t) / 1000. for event in event_file.get_events_iter(codes=['pupil_size_r'], time_range=[t1, t2])]
        # assert len(p_time) == len(p)
        # pupil_size.append(p)
        # pupil_time.append(p_time)
    assert len(eye_h) == len(stimulus_presented_df)
    # assert len(pupil_size) == len(stimulus_presented_df)
    event_file.close()

    ###########################################################################
    # Double-check `correct_fixation` is actually correct by analyzing the
    # `eye_h` and `eye_v` data
    ###########################################################################
    # # Threshold to check against to determine if we have enough eye data for given stimulus presentation
    # threshold = output['stim_on_time_ms'] // 2
    #
    # for i in range(len(eye_h)):
    #     if correct_fixation_df.iloc[i]['data'] == 0:  # Skip if already marked incorrect
    #         continue
    #
    #     if len(eye_h[i]) < threshold or len(eye_v[i]) < threshold:
    #         correct_fixation_df.at[i, 'data'] = 0
    #     elif np.any([np.abs(_) > output['fixation_window_size_degrees'] for _ in eye_h[i]]) or\
    #             np.any([np.abs(_) > output['fixation_window_size_degrees'] for _ in eye_v[i]]):
    #         correct_fixation_df.at[i, 'data'] = 0

    ###########################################################################
    # Save output
    ###########################################################################
    output["stimulus_presented"] = stimulus_presented_df.data.values.tolist()
    output["fixation_correct"] = correct_fixation_df.data.values.tolist()
    output["stimulus_order_in_trial"] = stimulus_presented_df.stimulus_order_in_trial.values.tolist()
    # output['eye_h_degrees'] = eye_h
    # output['eye_v_degrees'] = eye_v
    # output['eye_time_ms'] = eye_time
    output["samp_on_us"] = samp_on.astype(int)  # Convert to int okay only if times are in microseconds
    output["photodiode_on_us"] = photodiode_on.astype(int)  # Convert to int okay only if times are in microseconds
    output["rig_delays_us"] = (photodiode_on - samp_on).astype(int)
    # output['pupil_size_degrees'] = pupil_size
    # output['pupil_time_ms'] = pupil_time

    # save to processed folder
    output = pd.DataFrame(output)
    output_filepath = os.path.join(
        output_dir, str(filename).split("/")[-1][:-5] + "_mwk.csv"
    )  # -5 in filename to delete the .mwk2 extension
    # output.to_csv(filename.split('/')[-1][:-5] + '_mwk.csv', index=False)  # -5 in filename to delete the .mwk2 extension
    output.to_csv(output_filepath, index=False)

    # # save another to raw folder (for buildign psths)
    # data_folder = Path(photodiode_filepath).parent
    # output_filepath_raw = os.path.join(data_folder, str(filename).split('/')[-1][:-5] + '_mwk.csv')  # -5 in filename to delete the .mwk2 extension
    # output.to_csv(output_filepath_raw, index=False)

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


# if __name__ == '__main__':
#     dump_events(sys.argv[1], sys.argv[2], sys.argv[3])
