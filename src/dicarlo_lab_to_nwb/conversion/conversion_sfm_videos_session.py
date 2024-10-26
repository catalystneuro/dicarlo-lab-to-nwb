from pathlib import Path

from dicarlo_lab_to_nwb.conversion.convert_session import convert_session_to_nwb

data_folder = Path("/media/heberto/One Touch/DiCarlo-CN-data-share")
stimuli_folder = data_folder / "StimulusSets/SFM_foveal_videos_2/stimuli/SFM_videos/"

project_name = "SFM_foveal_videos_2"
stimulus_data_folder_path = data_folder / project_name
assert stimulus_data_folder_path.exists(), f"{stimulus_data_folder_path} does not exist"

session_date = "20240715"
session_date = "20240716"
subject = "Apollo"  # Confirm this?
pipeline_version = "DiLorean"

output_dir_path = data_folder / "nwb_files"
stub_test = False
verbose = True
add_thresholding_events = True
add_psth = True
stimuli_are_video = True
add_raw_amplifier_data = False

thresholindg_pipeline_kwargs = {
    "f_notch": 60.0,  # Frequency for the notch filter
    "bandwidth": 10.0,  # Bandwidth for the notch filter
    "f_low": 300.0,  # Low cutoff frequency for the bandpass filter
    "f_high": 6000.0,  # High cutoff frequency for the bandpass filter
    "noise_threshold": 3,  # Threshold for detection in the thresholding algorithm
}

# Ten bins starting 200 ms before the stimulus and spanning 400 ms
psth_kwargs = {"bins_span_milliseconds": 400, "num_bins": 10, "milliseconds_from_event_to_first_bin": -200.0}

# This is the ground truth time column for the stimuli in the mworks csv file
ground_truth_time_column = "samp_on_us"


session_folder_path = stimulus_data_folder_path / session_date
folders_in_session_date = [folder for folder in session_folder_path.iterdir() if folder.is_dir()]
session_data_folder_path = next(path for path in folders_in_session_date if project_name in path.name)
normalizers = [folder for folder in folders_in_session_date if "normalizers" in folder.name]

folder_paths_to_convert = [session_data_folder_path] + normalizers

for folder_with_data_path in folder_paths_to_convert:
    intan_file_path = folder_with_data_path / "info.rhd"
    assert intan_file_path.exists(), f"{intan_file_path} does not exist"

    # This is a csv file that contains stimuli time
    mworks_processed_file_path = [path for path in folder_with_data_path.iterdir() if path.suffix == ".csv"][0]
    assert mworks_processed_file_path.exists(), f"{mworks_processed_file_path} does not exist"

    # Folders are named {something}_{session_date}_{session_time}
    session_time = folder_with_data_path.name.split("_")[-1]

    is_normalizer = "normalizer" in folder_with_data_path.name
    type_of_data = "session_data" if not is_normalizer else "normalizer_data"
    session_metadata = {
        "project_name": project_name,
        "session_date": session_date,
        "session_time": session_time,
        "subject": subject,
        "type_of_data": type_of_data,
        "pipeline_version": pipeline_version,
    }

    convert_session_to_nwb(
        session_metadata=session_metadata,
        intan_file_path=intan_file_path,
        mworks_processed_file_path=mworks_processed_file_path,
        stimuli_folder=stimuli_folder,
        thresholindg_pipeline_kwargs=thresholindg_pipeline_kwargs,
        psth_kwargs=psth_kwargs,
        output_dir_path=output_dir_path,
        stub_test=stub_test,
        verbose=verbose,
        add_thresholding_events=add_thresholding_events,
        add_psth=add_psth,
        stimuli_are_video=stimuli_are_video,
        ground_truth_time_column=ground_truth_time_column,
        add_raw_amplifier_data=add_raw_amplifier_data,
    )
