from pathlib import Path

from dicarlo_lab_to_nwb.conversion.convert_session import convert_session_to_nwb
from dicarlo_lab_to_nwb.conversion.parse_mworks_RSVP import parse_mworks_file

data_folder = Path("/media/heberto/One Touch/DiCarlo-CN-data-share")
stimuli_folder = data_folder / "StimulusSets/SFM_foveal_contrast_images2/mworks_contrast_images2/"

project_name = "SFM_foveal_contrast_images2"
stimulus_data_folder_path = data_folder / project_name
assert stimulus_data_folder_path.exists(), f"{stimulus_data_folder_path} does not exist"

session_date = "20240814"
session_date = "20240815"
subject = "Apollo"
pipeline_version = "DiLorean"

output_dir_path = data_folder / "nwb_files"
stub_test = True
verbose = True
add_amplifier_data_to_nwb = True
add_thresholding_events = True
add_stimuli_media_to_nwb = True
process_mworks = False
add_psth = False
add_psth_in_pipeline_format_to_nwb = False
stimuli_are_video = False
is_stimuli_one_indexed = True

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


intan_file_path = session_data_folder_path / "info.rhd"
assert intan_file_path.exists(), f"{intan_file_path} does not exist"

intan_data_folder = intan_file_path.parent


if process_mworks:
    mkworks_procssed_folder = intan_data_folder / "mworks_processed"
    mkworks_procssed_folder.mkdir(exist_ok=True)
    mworks_processed_file_path = parse_mworks_file(
        mworks_folder=intan_data_folder,
        raw_data_folder=intan_data_folder,
        output_folder=mkworks_procssed_folder,
    )
else:
    mworks_processed_file_path = [path for path in intan_data_folder.iterdir() if path.suffix == ".csv"][0]
    assert (
        mworks_processed_file_path.exists()
    ), f"{mworks_processed_file_path} does not exist"  # This is a csv file that contains stimuli time

from pathlib import Path

folder_path = Path("/media/heberto/One Touch/DiCarlo-CN-data-share/StimulusSets/SFM_foveal_contrast_images2/meta")
file_path = folder_path / "SFM.shaded.contrast.images2_stratified.csv"
assert file_path.is_file(), f"{file_path} does not exist"

# Folders are named {something}_{session_date}_{session_time}
session_time = session_data_folder_path.name.split("_")[-1]

is_normalizer = "normalizer" in session_data_folder_path.name
type_of_data = "session" if not is_normalizer else "normalizer"
session_metadata = {
    "project_name": project_name,
    "session_date": session_date[2:],
    "session_time": session_time,
    "stimulus_name": "SFM_foveal_contrast_images2",
    "subject": subject,
    "type_of_data": type_of_data,
    "pipeline_version": pipeline_version,
}

# Training-test division
folder_path = Path("/media/heberto/One Touch/DiCarlo-CN-data-share/StimulusSets/SFM_foveal_contrast_images2/meta")
train_test_split_data_file_path = folder_path / "SFM.shaded.contrast.images2_stratified.csv"
assert train_test_split_data_file_path.is_file()


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
    add_stimuli_media_to_nwb=add_stimuli_media_to_nwb,
    ground_truth_time_column=ground_truth_time_column,
    add_amplifier_data_to_nwb=add_amplifier_data_to_nwb,
    add_psth_in_pipeline_format_to_nwb=add_psth_in_pipeline_format_to_nwb,
    train_test_split_data_file_path=train_test_split_data_file_path,
    is_stimuli_one_indexed=is_stimuli_one_indexed,
)