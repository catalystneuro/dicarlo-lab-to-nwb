import yaml
import re
from datetime import datetime
from pathlib import Path
from dicarlo_lab_to_nwb.thresholding_pipeline.data_locator import locate_session_paths
from dicarlo_lab_to_nwb.conversion.parse_mworks_RSVP import parse_mworks_file
from dicarlo_lab_to_nwb.conversion.convert_session import convert_session_to_nwb, calculate_quality_metrics_from_nwb
from pynwb import NWBHDF5IO
import numpy as np
from dicarlo_lab_to_nwb.conversion.quality_control.latency import get_unit_latencies_from_reliabilities
from dicarlo_lab_to_nwb.conversion.quality_control.reliability import get_NRR, get_p_values
import pandas as pd

# Recursive function to parse YAML content
def parse_yaml_recursively(data):
    # Check if the data is a dictionary
    if isinstance(data, dict):
        parsed_data = {}
        for key, value in data.items():
            parsed_data[key] = parse_yaml_recursively(value)
        return parsed_data
    
    # Check if the data is a list
    elif isinstance(data, list):
        return [parse_yaml_recursively(item) for item in data]
    
    # If it's neither dict nor list, it's a base case (string, int, etc.)
    else:
        return data


def convert_project_sessions(project_config_path: str | Path):
    project_config_path = Path(project_config_path)
    # assert if project_config_path is not a .yaml file
    assert project_config_path.exists(), f"File {project_config_path} does not exist"

    with open(project_config_path, "r") as file:
        try:
            yaml_content = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    parsed_result = parse_yaml_recursively(yaml_content)
        
    # GLOBAL VARIABLES
    # TODO: put this in a config file
    subject = "Apollo"
    pipeline_version = "DiLorean"

    stub_test = False
    verbose = True
    add_thresholding_events = True
    add_psth = True
    stimuli_are_video = False
    add_amplifier_data_to_nwb = False
    add_psth_in_pipeline_format_to_nwb = True

    thresholding_pipeline_kwargs = {
        "f_notch": 60.0,  # Frequency for the notch filter
        "bandwidth": 10.0,  # Bandwidth for the notch filter
        "f_low": 300.0,  # Low cutoff frequency for the bandpass filter
        "f_high": 6000.0,  # High cutoff frequency for the bandpass filter
        "noise_threshold": 3,  # Threshold for detection in the thresholding algorithm
    }

    # Ten bins starting 200 ms before the stimulus and spanning 400 ms
    psth_kwargs = {"bins_span_milliseconds": 500, "num_bins": 50, "milliseconds_from_event_to_first_bin": -200.0}
    # psth_kwargs = {"bins_span_milliseconds": 400, "num_bins": 10, "milliseconds_from_event_to_first_bin": -200.0}

    # This is the ground truth time column for the stimuli in the mworks csv file
    ground_truth_time_column = "samp_on_us"


    # project-level information
    project_info = parsed_result.get("project", {})
    project = project_info.get("name")
    project = "".join([word.capitalize() for word in project.split("_")])
    
    subject = project_info.get("subject")
    nwb_output_folder = project_info.get("nwb_output_folder")
    project_version = project_info.get("project_version")
    pipeline_version = project_info.get("pipeline_version")
    mworks_version = project_info.get("mworks_version")

    # Merged NWB file will be saved in the nwb_output_folder
    nwb_output_folder = Path(nwb_output_folder)
    nwb_output_folder.mkdir(parents=True, exist_ok=True)

    # experiment-level information
    experiments = project_info.get("experiments", [])
    for experiment in experiments:
        experiment_date = experiment.get("date")
        # intermediate NWB files from each day will be saved in the nwb_output_folder
        experiment_output_folder = nwb_output_folder / f"{experiment_date}"
        experiment_output_folder.mkdir(parents=True, exist_ok=True)
        print(f"\nExperiment Date: {experiment_date}-------------------\n")

        # session-level information
        sessions = experiment.get("sessions", [])
        for session in sessions:
            stimulus_name = session.get("stimulus_name")
            stimulus_folder = Path(session.get("stimulus_folder"))
            mworks_folder = Path(session.get("mworks_folder"))
            data_folder = Path(session.get("data_folder"))
            intan_file_path = data_folder / "info.rhd"
            assert intan_file_path.exists(), f"{intan_file_path} does not exist"
            notes = session.get("notes")

            print(f"  Session Stimulus Name: {stimulus_name}")
            print(f"  - Stimulus Folder: {stimulus_folder}")
            print(f"  - MWorks Folder: {mworks_folder}")
            print(f"  - Data Folder: {data_folder}")
            print(f"  - Notes: {notes}\n")
            
            # session info
            intan_recording_folder = data_folder.name
            match = re.match(r'(.+?)_(\d{6})_(\d{6})$', intan_recording_folder)
            if match:
                stimulus_name = match.group(1)
                session_date = match.group(2)
                session_time = match.group(3)
                stimulus_name_camel_case = "".join([word.capitalize() for word in stimulus_name.split("_")])

            else:
                assert False, f"intan_recording_folder name {intan_recording_folder} does not match the pattern"

            session_metadata = {
                "project_name": project,
                "subject": subject,
                "stimulus_name_camel_case": stimulus_name_camel_case,
                "session_date": session_date,
                "session_time": session_time,
                "pipeline_version": pipeline_version,
            }

            # Extract stimulus/behavioral events from MWorks files
            mworks_processed_file_path = parse_mworks_file(mworks_folder, data_folder, mworks_folder) # save MWorks results in the same folder
            print(f"  - MWorks processed file saved to: {mworks_processed_file_path}")

            session_nwb_filepath = convert_session_to_nwb(
                session_metadata=session_metadata,
                intan_file_path=intan_file_path,
                mworks_processed_file_path=mworks_processed_file_path,
                stimuli_folder=stimulus_folder,
                thresholindg_pipeline_kwargs=thresholding_pipeline_kwargs,
                psth_kwargs=psth_kwargs,
                output_dir_path=experiment_output_folder,
                stub_test=stub_test,
                verbose=verbose,
                add_thresholding_events=add_thresholding_events,
                add_psth=add_psth,
                stimuli_are_video=stimuli_are_video,
                add_stimuli_media_to_nwb=True,
                ground_truth_time_column=ground_truth_time_column,
                add_amplifier_data_to_nwb=add_amplifier_data_to_nwb,
                add_psth_in_pipeline_format_to_nwb=add_psth_in_pipeline_format_to_nwb,
            )

            # add quality control if stimulus_name contains "normalizer"
            if "normalizer" in stimulus_name:
                with NWBHDF5IO(session_nwb_filepath, mode='r') as io:
                    nwbfile = io.read()
                    calculate_quality_metrics_from_nwb(nwbfile, session_nwb_filepath.parent)

                    
if __name__ == '__main__':
    example_project_config = Path(__file__).parent / "project_config.yaml"
    convert_project_sessions(example_project_config)
        
    