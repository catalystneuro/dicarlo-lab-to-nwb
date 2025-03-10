"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from neuroconv import ConverterPipe
from neuroconv.datainterfaces import IntanRecordingInterface
from neuroconv.utils import dict_deep_update, load_dict_from_file
from pynwb import NWBFile

from dicarlo_lab_to_nwb.conversion.behaviorinterface import BehavioralTrialsInterface
from dicarlo_lab_to_nwb.conversion.pipeline import (
    calculate_thresholding_events,
    write_thresholding_events_to_nwb,
)
from dicarlo_lab_to_nwb.conversion.probe import (
    UtahArrayProbeInterface,
    attach_probe_to_recording,
)
from dicarlo_lab_to_nwb.conversion.psth import (
    write_binned_spikes_to_nwbfile,
    write_psth_pipeline_format_to_nwbfile,
)
from dicarlo_lab_to_nwb.conversion.quality_control.latency import (
    get_unit_latencies_from_reliabilities,
)
from dicarlo_lab_to_nwb.conversion.quality_control.reliability import (
    get_NRR,
    get_p_values,
)
from dicarlo_lab_to_nwb.conversion.stimuli_interface import (
    SessionStimuliImagesInterface,
    SessionStimuliVideoInterface,
)


def convert_session_to_nwb(
    session_metadata: dict,
    intan_file_path: str | Path,
    mworks_processed_file_path: str | Path,
    stimuli_folder: str | Path,
    output_dir_path: str | Path,
    stub_test: bool = False,
    verbose: bool = False,
    add_thresholding_events: bool = False,
    add_psth: bool = False,
    stimuli_are_video: bool = False,
    add_stimuli_media_to_nwb: bool = False,
    thresholindg_pipeline_kwargs: Optional[dict] = None,
    psth_kwargs: Optional[dict] = None,
    ground_truth_time_column: str = "samp_on_us",
    add_amplifier_data_to_nwb: bool = False,
    probe_info_path: Optional[str | Path] = None,
    add_psth_in_pipeline_format_to_nwb: bool = True,
    train_test_split_data_file_path: Optional[str | Path] = None,
    is_stimuli_one_indexed: bool = False,
) -> Path:

    if verbose:
        total_start = time.time()
        start = time.time()

    project_name = session_metadata["project_name"]
    subject = session_metadata["subject"]
    stimulus_name = session_metadata["stimulus_name"]
    session_date = session_metadata["session_date"]
    session_time = session_metadata["session_time"]
    pipeline_version = session_metadata.get("pipeline_version", "")
    data_collection = session_metadata.get("data_collection", "")

    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # session_id defines the name of the NWB file
    # Break this down so recording_id is separated int its components, consider using a camel case veersion
    # Of all the components like this example:
    project_name_camel_case = "".join([word.capitalize() for word in project_name.split("_")])
    stimulus_name_camel_case = "".join([word.capitalize() for word in stimulus_name.split("_")])

    session_id = f"{project_name_camel_case}_{subject}_{stimulus_name_camel_case}_{session_date}_{session_time}_{pipeline_version}_thresholded"
    # Remove __ from the session_id for fields that are not included
    session_id = session_id.replace("__", "_")
    nwbfile_path = output_dir_path / f"{session_id}.nwb"

    if verbose:
        print(f"Converting session: {session_id}")

    conversion_options = dict()

    if add_amplifier_data_to_nwb:
        ecephys_interface = IntanRecordingInterface(file_path=intan_file_path, ignore_integrity_checks=True)
        attach_probe_to_recording(recording=ecephys_interface.recording_extractor)
        if stub_test:
            intan_recording = ecephys_interface.recording_extractor
            duration = intan_recording.get_duration()
            end_time = min(10.0, duration)
            stubed_recording = ecephys_interface.recording_extractor.time_slice(start_time=0, end_time=end_time)
            ecephys_interface.recording_extractor = stubed_recording

        conversion_options["Ecephys"] = dict(
            iterator_opts={"display_progress": True, "buffer_gb": 5},
        )
    else:
        # This path adds the geometry of the probe as the electrodes table so the units can be linked
        # Add Utah Array Probe Interface, pass a path if a different geometry is used
        ecephys_interface = UtahArrayProbeInterface(file_path=probe_info_path)

    # Behavioral Trials Interface
    behavioral_trials_interface = BehavioralTrialsInterface(
        file_path=mworks_processed_file_path,
        train_test_split_data_file_path=train_test_split_data_file_path,
    )
    conversion_options["Behavior"] = dict(
        stub_test=stub_test,
        ground_truth_time_column=ground_truth_time_column,
        is_stimuli_one_indexed=is_stimuli_one_indexed,
    )

    # Build the converter pipe with the previously defined data interfaces
    data_interfaces_dict = {
        "Ecephys": ecephys_interface,
        "Behavior": behavioral_trials_interface,
    }

    # Add Stimuli Interface
    if add_stimuli_media_to_nwb:
        if stimuli_are_video:
            stimuli_interface = SessionStimuliVideoInterface(
                file_path=mworks_processed_file_path,
                folder_path=stimuli_folder,
                image_set_name=project_name,
                # video_copy_path=output_dir_path / "videos",
                video_copy_path=None,  # Add a path if videos should be copied
                verbose=verbose,
            )
        else:
            stimuli_interface = SessionStimuliImagesInterface(
                file_path=mworks_processed_file_path,
                folder_path=stimuli_folder,
                image_set_name=project_name,
                verbose=verbose,
                train_test_split_data_file_path=train_test_split_data_file_path,
            )

        conversion_options["Stimuli"] = dict(stub_test=stub_test, ground_truth_time_column=ground_truth_time_column)
        data_interfaces_dict["Stimuli"] = stimuli_interface

    converter_pipe = ConverterPipe(data_interfaces=data_interfaces_dict, verbose=verbose)

    # Parse the string into a datetime object
    datetime_str = f"20{session_date} {session_time}"
    datetime_format = "%Y%m%d %H%M%S"
    session_start_time = datetime.strptime(datetime_str, datetime_format).replace(tzinfo=ZoneInfo("US/Eastern"))

    # Add datetime to conversion
    metadata = converter_pipe.get_metadata()
    metadata["NWBFile"]["session_start_time"] = session_start_time
    metadata["NWBFile"]["session_id"] = session_id
    metadata["NWBFile"]["data_collection"] = data_collection
    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent / "metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(metadata, editable_metadata)

    subject_metadata = metadata["Subject"]
    subject_metadata["subject_id"] = f"{subject}"

    # Run conversion, this adds the basic data to the NWBFile
    converter_pipe.run_conversion(
        metadata=metadata,
        nwbfile_path=nwbfile_path,
        conversion_options=conversion_options,
        overwrite=True,
    )

    if verbose:
        stop_time = time.time()
        conversion_time_seconds = stop_time - start
        if conversion_time_seconds <= 60 * 3:
            print(f"Conversion took {conversion_time_seconds:.2f} seconds")
        elif conversion_time_seconds <= 60 * 60:
            print(f"Conversion took {conversion_time_seconds / 60:.2f} minutes")
        else:
            print(f"Conversion took {conversion_time_seconds / 60 / 60:.2f} hours")

    # Calculate thresholding events
    if add_thresholding_events:
        if verbose:
            start_time = time.time()
            print("Calculating and storing thresholding events with parameters: ")
            print(thresholindg_pipeline_kwargs)

        f_notch = thresholindg_pipeline_kwargs.get("f_notch", None)
        bandwidth = thresholindg_pipeline_kwargs.get("bandwidth", None)
        f_low = thresholindg_pipeline_kwargs.get("f_low", None)
        f_high = thresholindg_pipeline_kwargs.get("f_high", None)
        noise_threshold = thresholindg_pipeline_kwargs.get("noise_threshold", None)
        job_kwargs = thresholindg_pipeline_kwargs.get("job_kwargs", None)

        if add_amplifier_data_to_nwb:
            file_path = nwbfile_path
        else:
            file_path = intan_file_path

        sorting = calculate_thresholding_events(
            file_path=file_path,
            f_notch=f_notch,
            bandwidth=bandwidth,
            f_low=f_low,
            f_high=f_high,
            noise_threshold=noise_threshold,
            job_kwargs=job_kwargs,
            stub_test=stub_test,
            verbose=verbose,
            probe_info_path=probe_info_path,
        )

        nwbfile_path = write_thresholding_events_to_nwb(
            sorting=sorting,
            nwbfile_path=nwbfile_path,
            verbose=verbose,
            thresholindg_pipeline_kwargs=thresholindg_pipeline_kwargs,
        )

        del sorting

        if verbose:
            stop_time = time.time()
            thresholding_time = stop_time - start_time
            if thresholding_time <= 60 * 3:
                print(f"Thresholding events took {thresholding_time:.2f} seconds")
            elif thresholding_time <= 60 * 60:
                print(f"Thresholding events took {thresholding_time / 60:.2f} minutes")
            else:
                print(f"Thresholding events took {thresholding_time / 60 / 60:.2f} hours")

    # Add PSTH
    if add_thresholding_events and add_psth:
        if verbose:
            start_time = time.time()
            print("Calculating and storing PSTH with parameters: ")
            print(psth_kwargs)

        number_of_bins = psth_kwargs.get("num_bins")
        bins_span_milliseconds = psth_kwargs.get("bins_span_milliseconds")
        bin_width_in_milliseconds = bins_span_milliseconds / number_of_bins
        milliseconds_from_event_to_first_bin = psth_kwargs.get("milliseconds_from_event_to_first_bin", None)

        write_binned_spikes_to_nwbfile(
            nwbfile_path=nwbfile_path,
            bin_width_in_milliseconds=bin_width_in_milliseconds,
            number_of_bins=number_of_bins,
            milliseconds_from_event_to_first_bin=milliseconds_from_event_to_first_bin,
            verbose=verbose,
        )

        if add_psth_in_pipeline_format_to_nwb:
            write_psth_pipeline_format_to_nwbfile(
                nwbfile_path=nwbfile_path,
                verbose=verbose,
            )

        if verbose:
            stop_time = time.time()
            psth_time = stop_time - start_time
            if psth_time <= 60 * 3:
                print(f"PSTH calculation took {psth_time:.2f} seconds")
            elif psth_time <= 60 * 60:
                print(f"PSTH calculation took {psth_time / 60:.2f} minutes")
            else:
                print(f"PSTH calculation took {psth_time / 60 / 60:.2f} hours")

    if verbose:
        total_stop = time.time()
        total_script_time = total_stop - total_start
        if total_script_time <= 60 * 3:
            print(f"Total script took {total_script_time:.2f} seconds")
        elif total_script_time <= 60 * 60:
            print(f"Total script took {total_script_time / 60:.2f} minutes")
        else:
            print(f"Total script took {total_script_time / 60 / 60:.2f} hours")

        print("\n \n")

    return nwbfile_path


def calculate_quality_metrics_from_nwb(nwbfile: NWBFile, session_nwb_folder: Path) -> pd.DataFrame:
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
        rates[i, :, :] = np.sum(psth_unit[..., intg_window_idx[0] : intg_window_idx[1]], axis=-1) / intg_window_size_s
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
    rhos_samples = get_NRR(rates, n_samples=n_samples)
    rhos_mean_values = np.nanmean(rhos_samples, axis=1)
    print(f"half-split reliability (above 0.5) : {np.sum(rhos_mean_values>0.5)}")
    srr_samples = get_NRR(rates, n_samples=2, correction=False)
    srr_mean_values = np.nanmean(srr_samples, axis=-1)
    print(f"SRR mean (valid units): {np.mean(srr_mean_values[valid_units])}")

    # save results to a dataframe
    df = pd.DataFrame(
        {
            "channel_name": channel_names,
            "p_value": p_values,
            "valid_unit": valid_units,
            "mean_rate": mean_rates,
            "response_latency_s": latencies_s,
            "half_split_reliability": rhos_mean_values,
            "single_repeat_reliability": srr_mean_values,
        }
    )
    csv_filepath = session_nwb_folder / f"{nwbfile.session_id}_QC.csv"
    df.to_csv(csv_filepath, index=False)
    return df
