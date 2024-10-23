from pathlib import Path

data_folder = Path("/media/heberto/One Touch/DiCarlo-CN-data-share")

image_set_name = "SFM_foveal_videos_2"

stimulus_data_folder_path = data_folder / image_set_name
assert stimulus_data_folder_path.exists(), f"{stimulus_data_folder_path} does not exist"

session_date = "20240715"
session_time = "144848"

session_date = "20240716"
session_time = "142554"

session_date = "20240717"
session_time = "151839"

# Normalizers
session_date = "20240715"
session_time = "144848"
normalizer = True

subject = "Apollo"  # Confirm this?

session_folder_path = stimulus_data_folder_path / session_date

assert session_folder_path.exists(), f"{session_folder_path} does not exist"

if normalizer:
    session_data_folder = session_folder_path / f"{image_set_name}_{session_date[2:]}_{session_time}"
else:
    sesssion_data_folder = session_folder_path / f"normalizers_{session_date[2:]}_{session_time}_raw"
assert session_data_folder.exists(), f"{session_data_folder} does not exist"

session_data_intan_file_path = session_data_folder / "info.rhd"
assert session_data_intan_file_path.exists(), f"{session_data_intan_file_path} does not exist"

session_data_mworks_processed_file_path = (
    session_data_folder / f"rig1-SFM.foveal.videos_2_rig_1-{session_date}-{session_time}_mwk.csv"
)
session_data_mworks_processed_file_path = [path for path in session_data_folder.iterdir() if path.suffix == ".csv"][0]
assert session_data_mworks_processed_file_path.exists(), f"{session_data_mworks_processed_file_path} does not exist"

from dicarlo_lab_to_nwb.conversion.convert_session import convert_session_to_nwb

session_metadata = {
    "image_set_name": image_set_name,
    "session_date": session_date,
    "session_time": session_time,
    "subject": subject,
}

intan_file_path = session_data_intan_file_path
mworks_processed_file_path = session_data_mworks_processed_file_path

stimuli_folder = Path(
    "/media/heberto/One Touch/DiCarlo-CN-data-share/StimulusSets/SFM_foveal_videos_2/stimuli/SFM_videos/"
)

output_dir_path = Path.home() / "conversion_nwb"
stub_test = False
verbose = True
add_thresholding_events = True
add_psth = True
stimuli_are_video = True

thresholindg_pipeline_kwargs = {
    "f_notch": 60.0,  # Frequency for the notch filter
    "bandwidth": 10.0,  # Bandwidth for the notch filter
    "f_low": 300.0,  # Low cutoff frequency for the bandpass filter
    "f_high": 6000.0,  # High cutoff frequency for the bandpass filter
    "noise_threshold": 3,  # Threshold for detection in the thresholding algorithm
}

# Ten bins starting 200 ms before the stimulus and spanning 400 ms
psth_kwargs = {"bins_span_milliseconds": 400, "num_bins": 10, "milliseconds_from_event_to_first_bin": -200.0}


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
)
