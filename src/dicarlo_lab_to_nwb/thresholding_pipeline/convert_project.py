import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pynwb import NWBHDF5IO

from dicarlo_lab_to_nwb.conversion.convert_session import convert_session_to_nwb
from dicarlo_lab_to_nwb.conversion.quality_control.latency import (
    get_unit_latencies_from_reliabilities,
)
from dicarlo_lab_to_nwb.conversion.quality_control.reliability import (
    get_NRR,
    get_p_values,
)
from dicarlo_lab_to_nwb.thresholding_pipeline.parse_mworks_RSVP import parse_mworks_file


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

    thresholindg_pipeline_kwargs = {
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
            session_metadata = {
                "project_name": project,
                "subject": subject,
                "recording_id": data_folder.name,
                "pipeline_version": pipeline_version,
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
                thresholindg_pipeline_kwargs=thresholindg_pipeline_kwargs,
                psth_kwargs=psth_kwargs,
                output_dir_path=experiment_output_folder,
                stub_test=stub_test,
                verbose=verbose,
                add_thresholding_events=add_thresholding_events,
                add_psth=add_psth,
                stimuli_are_video=stimuli_are_video,
                ground_truth_time_column=ground_truth_time_column,
                add_amplifier_data_to_nwb=add_amplifier_data_to_nwb,
                add_psth_in_pipeline_format_to_nwb=add_psth_in_pipeline_format_to_nwb,
            )

            # add quality control if stimulus_name contains "normalizer"
            if "normalizer" in stimulus_name:
                with NWBHDF5IO(session_nwb_filepath, mode="r") as io:
                    nwbfile = io.read()
                    psth = nwbfile.scratch["psth_pipeline_format"].data[:]
                    n_units, n_stimuli, n_reps, n_timebins = psth.shape
                    df = nwbfile.electrodes.to_dataframe()
                    channel_names = df["channel_name"].values

                    binned_spikes = nwbfile.processing["ecephys"]["BinnedAlignedSpikesToStimulus"]
                    psth_timebin_ms = binned_spikes.bin_width_in_milliseconds
                    psth_0 = binned_spikes.milliseconds_from_event_to_first_bin
                    psth_1 = psth_0 + psth_timebin_ms * n_timebins - psth_timebin_ms
                    psth_timebins_s = np.linspace(psth_0, psth_1, n_timebins) / 1e3
                    latencies_s = get_unit_latencies_from_reliabilities(psth, psth_timebins_s)

                    trial_df = nwbfile.intervals["trials"].to_dataframe()
                    stim_duration_s = trial_df["stimuli_presentation_time_ms"].unique()[0] / 1e3
                    stim_size_deg = trial_df["stimulus_size_degrees"].unique()[0]

                    # integrate spike counts over stimulus presentation duration + response latency
                    psth_stim_onset_s = 0
                    rates = np.full((n_units, n_stimuli, n_reps), np.nan)
                    mean_rates = np.full(n_units, np.nan)
                    for i in range(n_units):
                        intg_window_s_0 = psth_stim_onset_s + latencies_s[i]
                        intg_window_s_1 = intg_window_s_0 + stim_duration_s
                        intg_window = [intg_window_s_0, intg_window_s_1]
                        intg_window_size_s = np.diff(intg_window)[0]
                        intg_window_idx = [np.argmin(np.abs(psth_timebins_s - t)) for t in intg_window]

                        psth_unit = psth[i]
                        rates[i, :, :] = (
                            np.sum(psth_unit[..., intg_window_idx[0] : intg_window_idx[1]], axis=-1)
                            / intg_window_size_s
                        )
                        # average rates across all dimensions except the first one
                        mean_rates[i] = np.nanmean(rates[i])

                    # p values
                    p_values = get_p_values(rates)
                    p_thresh = 0.05
                    valid_units = p_values < p_thresh
                    print(f"with p < {p_thresh}: N={np.sum(valid_units)} out of {len(p_values)} units")
                    print(f"mean latencies (valid units): {np.nanmean(latencies_s[valid_units])}")

                    # reliabilities
                    n_samples = n_reps // 2
                    rhos_samples = get_NRR(rates, n_reps=n_samples)
                    rhos_mean_values = np.nanmean(rhos_samples, axis=1)
                    print(f"half-split reliability (above 0.5) : {np.sum(rhos_mean_values>0.5)}")
                    srr_samples = get_NRR(rates, n_reps=2, correction=False)
                    srr_mean_values = np.nanmean(srr_samples, axis=-1)
                    print(f"SRR mean (valid units): {np.mean(srr_mean_values[valid_units])}")

                    # save results to a dataframe
                    df = pd.DataFrame(
                        {
                            "channel_name": channel_names,
                            "p_value": p_values,
                            "valid_unit": valid_units,
                            "mean_rate": mean_rates,
                            "response_latency_ms": latencies_s,
                            "half_split_reliability": rhos_mean_values,
                            "single_repeat_reliability": srr_mean_values,
                        }
                    )
                    csv_filepath = session_nwb_filepath.parent / f"{nwbfile.session_id}_QC.csv"
                    df.to_csv(csv_filepath, index=False)


if __name__ == "__main__":
    example_project_config = Path(__file__).parent / "project_config.yaml"
    convert_project_sessions(example_project_config)
