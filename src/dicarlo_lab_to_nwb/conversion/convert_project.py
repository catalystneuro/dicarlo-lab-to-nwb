import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pynwb import NWBHDF5IO

from dicarlo_lab_to_nwb.conversion.aggregation import aggregate_nwbfiles
from dicarlo_lab_to_nwb.conversion.convert_session import (
    calculate_quality_metrics_from_nwb,
    convert_session_to_nwb,
)
from dicarlo_lab_to_nwb.conversion.parse_mworks_RSVP import parse_mworks_file
from dicarlo_lab_to_nwb.conversion.quality_control.latency import (
    get_unit_latencies_from_reliabilities,
)
from dicarlo_lab_to_nwb.conversion.quality_control.reliability import (
    get_NRR,
    get_p_values,
)


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


def convert_project_sessions(
    project_config_path: str | Path,
    valid_unit_metric: str = "p_value",
    valid_unit_threshold: float = 0.05,
):
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
    stub_test = False
    verbose = True
    add_thresholding_events = True
    add_psth = True
    stimuli_are_video = False
    add_amplifier_data_to_nwb = False
    add_psth_in_pipeline_format_to_nwb = True
    probe_info_path = Path(__file__).parent / "probe_info_dicarlo.csv"

    progress_bar = verbose
    # job_kwargs = dict(n_jobs=-1, progress_bar=progress_bar, chunk_duration=60.0)  # Fixed chunks to 60 seconds
    job_kwargs = dict(n_jobs=1, progress_bar=progress_bar, chunk_duration=10.0)  # Fixed chunks to 10 seconds

    thresholding_pipeline_kwargs = {
        "f_notch": 60.0,  # Frequency for the notch filter
        "bandwidth": 10.0,  # Bandwidth for the notch filter
        "f_low": 300.0,  # Low cutoff frequency for the bandpass filter
        "f_high": 6000.0,  # High cutoff frequency for the bandpass filter
        "noise_threshold": 3,  # Threshold for detection in the thresholding algorithm
        "job_kwargs": job_kwargs,
    }

    # This is the ground truth time column for the stimuli in the mworks csv file
    ground_truth_time_column = "photodiode_on_us"

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
    daily_experiments = project_info.get("experiments", [])

    # NWB filepaths for each individual session (recording)
    session_nwb_filepaths = []

    # quality-control dataframes across multiple sessions
    qc_dataframes = []
    for experiment in daily_experiments:
        experiment_date = experiment.get("date")
        # intermediate NWB files from each day will be saved in the nwb_output_folder
        experiment_output_folder = nwb_output_folder / f"{experiment_date}"
        experiment_output_folder.mkdir(parents=True, exist_ok=True)
        print(f"\nExperiment Date: {experiment_date}-------------------\n")

        # session-level information
        sessions = experiment.get("sessions", [])
        for session in sessions:
            # folder paths
            stimulus_name = session.get("stimulus_name")
            stimulus_folder = Path(session.get("stimulus_folder"))
            mworks_folder = Path(session.get("mworks_folder"))
            data_folder = Path(session.get("data_folder"))
            intan_file_path = data_folder / "info.rhd"
            assert intan_file_path.exists(), f"{intan_file_path} does not exist"

            # are the stimuli video or image?
            stimulus_type = str(session.get("stimulus_type"))
            stimuli_are_video = "video" in stimulus_type.lower()

            # psth time dimensions
            psth_start_s = session.get("psth_start_s")
            psth_end_s = session.get("psth_end_s")
            psth_bin_size_s = session.get("psth_bin_size_s")
            psth_timebins_s = np.round(np.arange(psth_start_s, psth_end_s, psth_bin_size_s), 3)
            # organize psth kwargs

            psth_kwargs = {
                "bins_span_milliseconds": (psth_end_s - psth_start_s) * 1000.0, 
                "num_bins": len(psth_timebins_s), 
                "milliseconds_from_event_to_first_bin": psth_start_s * 1000.0
                }
    
            notes = session.get("notes")

            print(f"  Session Stimulus Name: {stimulus_name}")
            print(f"  - Stimulus Folder: {stimulus_folder}")
            print(f"  - MWorks Folder: {mworks_folder}")
            print(f"  - Data Folder: {data_folder}")
            print(f"  - Notes: {notes}\n")

            # session info
            intan_recording_folder = data_folder.name
            match = re.match(r"(.+?)_(\d{6})_(\d{6})$", intan_recording_folder)
            if match:
                stimulus_name = match.group(1)
                session_date = match.group(2)
                session_time = match.group(3)
                data_collection = "normalizer" if "normalizer" in stimulus_name else "session"

            else:
                assert False, f"intan_recording_folder name {intan_recording_folder} does not match the pattern"

            session_metadata = {
                "project_name": project,
                "subject": subject,
                "stimulus_name": stimulus_name,
                "session_date": session_date,
                "session_time": session_time,
                "pipeline_version": pipeline_version,
                "data_collection": data_collection,
            }

            # Extract stimulus/behavioral events from MWorks files
            mworks_processed_file_path = parse_mworks_file(
                mworks_folder, data_folder, mworks_folder
            )  # save MWorks results in the same folder
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
                add_stimuli_media_to_nwb=False,
                ground_truth_time_column=ground_truth_time_column,
                add_amplifier_data_to_nwb=add_amplifier_data_to_nwb,
                add_psth_in_pipeline_format_to_nwb=add_psth_in_pipeline_format_to_nwb,
                probe_info_path=probe_info_path,
            )
            session_nwb_filepaths.append(session_nwb_filepath)
            
            # add quality control if stimulus_name contains "normalizer"
            if "normalizer" in stimulus_name:
                with NWBHDF5IO(session_nwb_filepath, mode="a") as io:
                    nwbfile = io.read()
                    qm_df = calculate_quality_metrics_from_nwb(nwbfile, session_nwb_filepath.parent)
                    from hdmf.common import DynamicTable

                    table = DynamicTable(description="quality_metrics", name="QualityMetricsSessionDateNormalizer")

                    for column in qm_df.columns:
                        table.add_column(name=column, description="")

                    for row in qm_df.iterrows():
                        table.add_row(row[1].to_dict())

                    nwbfile.add_scratch(table,
                                        name="quality_control_table",
                                        description="Quality control metrics from normalizer set",)
                    qc_dataframes.append(qm_df)

    stacked_metrics = pd.concat(
        {f"normalizer_{i}": df[valid_unit_metric] for i, df in enumerate(qc_dataframes)}, axis=1
    )
    if stub_test:
        valid_units = np.full(288, False)
        valid_units[:23] = True
    else:
        boolean_stacked = stacked_metrics < valid_unit_threshold
        valid_units = boolean_stacked.all(axis=1)

    aggregated_nwbfile_path = aggregate_nwbfiles(session_nwb_filepaths, 
                                                 nwb_output_folder, 
                                                 pipeline_version, 
                                                 valid_units,
                                                 qc_dataframes)


# if __name__ == "__main__":
#     project_config_folder = Path("/Users/yoon/Dropbox (MIT)/dorsal_ventral/pipeline_lab/dicarlo-lab-to-nwb-testing")
#     project_config_path = project_config_folder / "project_config_simulation.yaml"
#     example_project_config = project_config_path
#     convert_project_sessions(example_project_config)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python convert_project.py <config.yaml file path>")
        sys.exit(1)

    project_config_path = sys.argv[1]
    convert_project_sessions(project_config_path)

