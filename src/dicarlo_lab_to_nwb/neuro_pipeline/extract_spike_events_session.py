import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import yaml

from neuroconv import ConverterPipe
from neuroconv.datainterfaces import IntanRecordingInterface
from neuroconv.utils import dict_deep_update, load_dict_from_file

from dicarlo_lab_to_nwb.neuro_pipeline.behaviorinterface import BehavioralTrialsInterface
from dicarlo_lab_to_nwb.neuro_pipeline.data_locator import locate_session_paths

from dicarlo_lab_to_nwb.neuro_pipeline.pipeline import (
    calculate_thresholding_events_from_nwb,
    write_thresholding_events_to_nwb,
)
from dicarlo_lab_to_nwb.neuro_pipeline.probe import attach_probe_to_recording
from dicarlo_lab_to_nwb.neuro_pipeline.psth import write_binned_spikes_to_nwbfile
from dicarlo_lab_to_nwb.neuro_pipeline.stimuli_interface import (
    StimuliImagesInterface,
    StimuliVideoInterface,
)
from dicarlo_lab_to_nwb.neuro_pipeline.parse_mworks_RSVP import parse_mworks_file


def parse_yaml(yaml_content: str | Path, prefix: str = "") -> dict:
    
    meta_dict = {}
    for key, value in yaml_content.items():
        # Construct a full key (e.g., session.session_sets or session.subject)
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            # recursive call for nested dictionaries
            meta_dict.update(parse_yaml(value, prefix=full_key))
        else:
            # add leaf key-value pair
            meta_dict[full_key] = value
    
    return meta_dict


    return meta_dict
# Get spike timestamps from a single session
# session: recording from one particular stimulus set, e.g., normalizers.V3
def extract_spikes_from_session(
    session_metadata: dict,
    intan_file_path: str | Path,
    mworks_processed_file_path: str | Path,
    stimuli_folder: str | Path,
    output_dir_path: str | Path,
    verbose: bool = True,
    stimuli_are_video: bool = False,
    thresholding_kwargs: dict = None,
    stub_test: bool = False,
):
    
    if verbose:
        start_time = time.time()
    
    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # components for creating a session identifier
    project_name = session_metadata["project_name"]
    stimulus_set_name = session_metadata["stimulus_set_name"]
    session_date = session_metadata["session_date"]
    session_time = session_metadata["session_time"]
    subject = session_metadata["subject"]

    # create a session identifier
    session_id = f"{subject}_{project_name}_{stimulus_set_name}_{session_date}_{session_time}"
    nwbfile_path = output_dir_path / f"{session_id}.nwb"
    print(f"Converting to NWB: session_id: {session_id}")
    print(f"... converted NWB will be saved to: {nwbfile_path}")

    ### Build the three necessary data interfaces for the NWB conversion ============================================
    # 1. Add Intan Interface
    intan_recording_interface = IntanRecordingInterface(file_path=intan_file_path, ignore_integrity_checks=False)
    attach_probe_to_recording(recording=intan_recording_interface.recording_extractor)
    if stub_test:
        intan_recording = intan_recording_interface.recording_extractor
        duration = intan_recording.get_duration()
        end_time = min(10.0, duration)
        stubed_recording = intan_recording_interface.recording_extractor.time_slice(start_time=0, end_time=end_time)
        intan_recording_interface.recording_extractor = stubed_recording

    # 2. Add Behavioral Trials Interface
    behavioral_trials_interface = BehavioralTrialsInterface(file_path=mworks_processed_file_path)

    # 3. Add Stimuli Interface
    # Add Stimuli Interface
    if stimuli_are_video:
        stimuli_interface = StimuliVideoInterface(
            file_path=mworks_processed_file_path,
            folder_path=stimuli_folder,
            image_set_name=project_name,
            video_copy_path=output_dir_path / "videos",
            verbose=verbose,
        )
    else:
        stimuli_interface = StimuliImagesInterface(
            mworks_file_path=mworks_processed_file_path,
            folder_path=stimuli_folder,
            image_set_name=project_name,
            verbose=verbose,
        )

    # Add datetime to conversion
    # Build the converter pipe with the previously defined data interfaces
    data_interfaces_dict = {
        "Recording": intan_recording_interface,
        "Behavior": behavioral_trials_interface,
        "Stimuli": stimuli_interface,
    }
    converter_pipe = ConverterPipe(data_interfaces=data_interfaces_dict, verbose=verbose)

    ### Meta data for the NWB file ====================================================================================
    # Parse the string into a datetime object
    datetime_str = f"{session_date} {session_time}"
    datetime_format = "%Y%m%d %H%M%S"
    session_start_time = datetime.strptime(datetime_str, datetime_format).replace(tzinfo=ZoneInfo("US/Eastern"))

    metadata = converter_pipe.get_metadata()
    metadata["NWBFile"]["session_start_time"] = session_start_time
    metadata["NWBFile"]["session_id"] = session_id

    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent / "metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(metadata, editable_metadata)

    # subject_metadata = metadata["Subject"]
    # subject_metadata["subject_id"] = f"{subject}"

    # NWB conversion options
    conversion_options = dict()
    conversion_options["Recording"] = dict(
        iterator_opts={"display_progress": True, "buffer_gb": 5},
    )
    
    # Run conversion, this adds all the recording descriptions to the NWBFile
    converter_pipe.run_conversion(
        metadata=metadata,
        nwbfile_path=nwbfile_path,
        conversion_options=conversion_options,
        overwrite=True,
    )

    ### Spike timestamps from thresholding====================================================================================
    spike_events = calculate_thresholding_events_from_nwb(
        nwbfile_path=nwbfile_path,
        f_notch=thresholding_kwargs.get("f_notch", None),
        bandwidth=thresholding_kwargs.get("bandwidth", None),
        f_low=thresholding_kwargs.get("f_low", None),
        f_high=thresholding_kwargs.get("f_high", None),
        noise_threshold=thresholding_kwargs.get("noise_threshold", None),
        job_kwargs=thresholding_kwargs.get("job_kwargs", None),
        stub_test=stub_test,
        verbose=verbose,
    )
    nwbfile_path = write_thresholding_events_to_nwb(
        sorting=spike_events,
        nwbfile_path=nwbfile_path,
        append=True,
        verbose=verbose,
        thresholding_pipeline_kwargs=thresholding_kwargs,
    )

    if verbose:
            stop_time = time.time()
            thresholding_time = stop_time - start_time
            if thresholding_time <= 60 * 3:
                print(f"Thresholding (extract_spikes_from_session) took {thresholding_time:.2f} seconds")
            elif thresholding_time <= 60 * 60:
                print(f"Thresholding (extract_spikes_from_session) took {thresholding_time / 60:.2f} minutes")
            else:
                print(f"Thresholding (extract_spikes_from_session) took {thresholding_time / 60 / 60:.2f} hours")

    return nwbfile_path


if __name__ == "__main__":
    # input parameters for recording session (day): use YAML file
    session_yaml_path = Path(__file__).parent / "session_meta_example.yaml"
    with open(session_yaml_path, "r") as file:
        try:
            yaml_content = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    session_dict = parse_yaml(yaml_content, prefix="")
    stim_set_list = yaml_content.get('session', {}).get('stimulus_sets', [])
    print(f"Detected {len(stim_set_list)} stimulus sets: ")
    [print(f" ... {stim_set['name']}") for stim_set in stim_set_list]
            
    projectName = session_dict['session.project']
    subjectName = session_dict['session.subject']
    sessionDate =session_dict['session.date']
    root_dir = session_dict['session.data_folder']
    output_folder = session_dict['session.output_folder']
    pipeline_version = session_dict['session.version']

    # Load pipeline parameters from the pipeline version
    pipeline_yaml_path = Path(__file__).parent / "assets" / f"{pipeline_version}.yaml"
    with open(pipeline_yaml_path, "r") as file:
        try:
            pipeline_yaml = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    pipeline_params_dict = parse_yaml(pipeline_yaml, prefix="")
    thresholding_kwargs = {
        "f_notch": pipeline_params_dict['pipeline.thresholding.f_notch'],  # Frequency for the notch filter
        "bandwidth": pipeline_params_dict['pipeline.thresholding.bandwidth'],  # Bandwidth for the notch filter
        "f_low": pipeline_params_dict['pipeline.thresholding.f_low'],  # Low cutoff frequency for the bandpass filter
        "f_high": pipeline_params_dict['pipeline.thresholding.f_high'],  # High cutoff frequency for the bandpass filter
        "noise_threshold": pipeline_params_dict['pipeline.thresholding.noise_threshold'],  # Threshold for detection in the thresholding algorithm
    }

    data_folders = locate_session_paths(root_dir, subjectName, projectName, sessionDate)
    assert len(stim_set_list) == len(data_folders), "Number of stimulus sets and data folders do not match."
    
    stub_test = True
    verbose = True

    # sort data_folders by session time
    data_folders = sorted(data_folders, key=lambda x: Path(x).name.split('_')[-1])
    print(f"{len(data_folders)} data folders: \n")
    for i, data_folder in enumerate(data_folders):
        print(f"... processing data folder: { data_folder}")
        assert data_folder.is_dir(), f"Data directory not found: {data_folder}"
        
        # meta info for ePhys 
        foldername = Path(data_folder).name
        tokens = foldername.split("_")
        stimulus_set_name = tokens[0]
        stimulus_folder = stim_set_list[i]['stimulus_folder']
        folder_sessionDate = tokens[1]
        folder_sessionTime = tokens[2]
        print(f"... stimulus_set: {stimulus_set_name} | session_date: {folder_sessionDate} | session_time: {folder_sessionTime}")

        session_metadata = {
            "project_name": projectName, # SFM_images
            "stimulus_set_name": stimulus_set_name, # e.g., normalizers or SFM_images
            "session_date": folder_sessionDate, # same dates
            "session_time": folder_sessionTime, # different session times
            "subject": subjectName,
        }
        intan_file_path = data_folder / "info.rhd"

        # Process behavior data associated to each stimulus set recording (MWorks 0.12 and up)
        # mworks_processed_file_path = Path("/Users/yoon/raw_data/Apollo/test/20240701/normalizers_240701_131011/apollo_normalizersV3_240701_131011_mwk.csv")
        mworks_folder = stim_set_list[i]['behavior_folder']
        mworks_processed_file_path = parse_mworks_file(mworks_folder, data_folder, mworks_folder) # save MWorks results in the same folder

        nwbfile_path = extract_spikes_from_session(
            session_metadata=session_metadata,
            intan_file_path=intan_file_path,
            mworks_processed_file_path=mworks_processed_file_path,
            stimuli_folder=stimulus_folder,
            output_dir_path=output_folder,
            verbose=verbose,
            stimuli_are_video=False,
            thresholding_kwargs=thresholding_kwargs,
            stub_test=stub_test,
        )

