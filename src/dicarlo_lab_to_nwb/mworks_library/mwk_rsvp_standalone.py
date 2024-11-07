import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# debugging
from dicarlo_lab_to_nwb.mworks_library.mwk2reader import MWKFile

_logger = logging.getLogger(__name__)


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


def dump_events_rsvp_standalone(filename, output_dir: str = "./"):

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
        }
    else:
        output = {
            "stim_on_time_ms": data[data.name == "stim_on_time"]["data"].values[-1] / 1000.0,
            "stim_off_time_ms": data[data.name == "stim_off_time"]["data"].values[-1] / 1000.0,
            "stim_on_delay_ms": data[data.name == "stim_on_delay"]["data"].values[-1] / 1000.0,
            "stimulus_size_degrees": data[data.name == "stimulus_size"]["data"].values[-1],  # for 'normalizers'
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
    # Extract image file hash from #stimDisplayUpdate events
    ###########################################################################
    stimulus_type_list = []
    filepath_list = []
    file_hash_list = []
    sdu_events = []
    for e_i in event_file.get_events_iter(codes=["#stimDisplayUpdate"]):
        for d_i in e_i.data:
            if d_i.get("type") == "image":
                sdu_events.append(e_i)
                filepath = d_i["filename"]
                filepath_list.append(Path(filepath).name)
                file_hash_list.append(d_i["file_hash"])
                stimulus_type_list.append("image")
                break
            elif d_i.get("type") == "video":
                if d_i.get("current_video_time_seconds") == 0.0:
                    sdu_events.append(e_i)
                    filepath = d_i["filename"]
                    filepath_list.append(Path(filepath).name)
                    file_hash_list.append("")  # No hash for videos
                    stimulus_type_list.append("video")
                break
            elif d_i.get("type") == "audio":
                stimulus_type = "audio"
                stimulus_type_list.append("audio")
                filepath = d_i["filename"]
                filepath_list.append(Path(filepath).name)
                print("Audio stimulus detected. Not supported yet")
                break

    stimulus_presented_df["stimulus_type"] = stimulus_type_list
    stimulus_presented_df["filename"] = filepath_list
    stimulus_presented_df["image_hash"] = file_hash_list

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
    # Save output
    ###########################################################################
    output["stimulus_presented"] = stimulus_presented_df.data.values.tolist()
    output["fixation_correct"] = correct_fixation_df.data.values.tolist()
    output["stimulus_order_in_trial"] = stimulus_presented_df.stimulus_order_in_trial.values.tolist()
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
    print(f"{min(num_repetitions)} repeats, range is {np.unique(num_repetitions)}")


if __name__ == "__main__":

    root_path = Path("/Users/yoon/Downloads")
    mwk2_path = root_path / "apollo_template_241007_154729.mwk2"
    output_dir = root_path / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {mwk2_path}...\n")
    dump_events_rsvp_standalone(mwk2_path, output_dir)
    print(f"Finished parsing {mwk2_path.name}!\n")
