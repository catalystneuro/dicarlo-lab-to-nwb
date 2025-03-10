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
            "stimulus_size_degrees": data[data.name == "stimulus_size_deg"]["data"].values[-1],  # for 'gestalt_control' & 'SFM_*'
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
    fid = open(digi_event_filepath, "r")
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

    if len(correct_fixation_df) < len(samp_on):
        samp_on = samp_on[: len(correct_fixation_df)]

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
    # Extract image file hash from #stimDisplayUpdate events
    ###########################################################################
    stimulus_type_list = []
    filepath_list = []
    file_hash_list = []
    
    # track video events separately in case there are video stimuli
    video_sdu_times = []
    sdu_times = []
    in_video = False
    video_info = None  # Store current video info
    
    for e_i in event_file.get_events_iter(codes=["#stimDisplayUpdate"]):
        has_video = any(data_i.get('type')=='video' for data_i in e_i.data)
        
        if has_video:
            if not in_video:
                # New video starting
                video_sdu_times.append(e_i.time)
                in_video = True
                # Store video info
                for d_i in e_i.data:
                    if d_i.get("type") == "video":
                        video_info = {
                            "type": "video",
                            "filename": Path(d_i["filename"]).name,
                            "file_hash": ""  # No hash for videos yet for MWorks 0.12
                        }
                        # Add video info only once at start
                        stimulus_type_list.append(video_info["type"])
                        filepath_list.append(video_info["filename"])
                        file_hash_list.append(video_info["file_hash"])
                        break
            # Skip adding info for subsequent video frames
            continue
            
        else: # non-video stimuli
            # Reset video state
            in_video = False
            video_info = None
            
        # Handle non-video stimuli
        for d_i in e_i.data:
            sdu_times.append(e_i.time)
            if d_i.get("type") == "image":
                stimulus_type_list.append("image")
                filepath_list.append(Path(d_i["filename"]).name)
                file_hash_list.append(d_i["file_hash"])
                break
            elif d_i.get("type") == "audio":
                stimulus_type_list.append("audio")
                filepath_list.append(Path(d_i["filename"]).name)
                file_hash_list.append("")
                print("Audio stimulus detected. Not supported yet")
                break
            
    
    # (YB) 02/26/2025: patch for VIDEO only experiments.
    # Detected cases where there are more stimulus presentations than #stimDisplayUpdate events with videos.
    # Video SDU events typically occur after the stimulus presentation with a delay ranging from 9ms to 30ms, but this is not always the case.
    if 'video' in stimulus_type_list:
        # Get timestamps from stimulus_presented_df
        stim_times = stimulus_presented_df['time'].values
        stimulus_presented_df_skip_idx = []

        # Align timestamps and adjust lists accordingly
        aligned_type_list = []
        aligned_filepath_list = []
        aligned_hash_list = []
        aligned_stim_times = [] # for debugging
        aligned_sdu_times = [] # for debugging
        aligned_samp_on_us = [] 
        aligned_photodiode_on_us = []
        stim_idx = 0
        sdu_idx = 0
        
        while stim_idx < len(stim_times) and sdu_idx < len(video_sdu_times):
            time_diff = video_sdu_times[sdu_idx] - stim_times[stim_idx]
            
            if time_diff > 0:
                # If video SDU time is later than its own stimulus time, skip this stimulus time
                print(f"Skipping trial time {stim_times[stim_idx]} with no video (diff w/ next video: {time_diff/1000:.2f}ms)")
                # adjust stimulus_presented_df to match the aligned length
                stimulus_presented_df_skip_idx.append(stim_idx)
                stim_idx += 1
            else:
                # Add the SDU info and advance both indices
                aligned_type_list.append(stimulus_type_list[sdu_idx])
                aligned_filepath_list.append(filepath_list[sdu_idx])
                aligned_hash_list.append(file_hash_list[sdu_idx])
                aligned_stim_times.append(stim_times[stim_idx])
                aligned_sdu_times.append(sdu_times[sdu_idx])

                aligned_samp_on_us.append(samp_on[stim_idx])
                aligned_photodiode_on_us.append(photodiode_on[stim_idx])

                stim_idx += 1
                sdu_idx += 1
        
        # Update the lists with aligned versions
        stimulus_type_list = aligned_type_list
        filepath_list = aligned_filepath_list
        file_hash_list = aligned_hash_list
        samp_on = aligned_samp_on_us
        photodiode_on = aligned_photodiode_on_us
        # Trim stimulus_presented_df to match the aligned length
        stimulus_presented_df = stimulus_presented_df[~stimulus_presented_df.index.isin(stimulus_presented_df_skip_idx)].reset_index(drop=True)
        correct_fixation_df = correct_fixation_df[~correct_fixation_df.index.isin(stimulus_presented_df_skip_idx)].reset_index(drop=True)
        # stimulus_presented_df = stimulus_presented_df.iloc[:len(stimulus_type_list)].reset_index(drop=True)
        # correct_fixation_df = correct_fixation_df.iloc[:len(stimulus_type_list)].reset_index(drop=True)
        
        # Verify lengths match
        assert len(stimulus_type_list) == len(stimulus_presented_df), \
            f"Stimulus type list length ({len(stimulus_type_list)}) doesn't match presented stimuli ({len(stimulus_presented_df)})"
        
        print(f"\nFinal alignment: {len(stimulus_type_list)} stimuli")
    
    stimulus_presented_df["stimulus_type"] = stimulus_type_list
    stimulus_presented_df["filename"] = filepath_list
    stimulus_presented_df["image_hash"] = file_hash_list


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
