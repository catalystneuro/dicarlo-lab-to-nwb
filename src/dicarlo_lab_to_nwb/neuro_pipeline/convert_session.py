"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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

from dicarlo_lab_to_nwb.quality_control.latency import get_unit_latencies_from_reliabilities
# from dicarlo_lab_to_nwb.quality_control.reliability import get_NRR, get_p_values
# from dicarlo_lab_to_nwb.quality_control.snr import compute_snr_resampled, compute_null_snr_resampled


def convert_session_to_nwb(
    session_metadata: dict,
    intan_file_path: str | Path,
    mworks_processed_file_path: str | Path,
    stimuli_folder: str | Path,
    output_dir_path: str | Path | None = None,
    stub_test: bool = False,
    verbose: bool = False,
    add_thresholding_events: bool = False,
    add_psth: bool = False,
    stimuli_are_video: bool = False,
    thresholindg_pipeline_kwargs: dict = None,
    psth_kwargs: dict = None,
):
    if verbose:
        total_start = time.time()
        start = time.time()

    if output_dir_path is None:
        output_dir_path = Path.home() / "conversion_nwb"

    project_name = session_metadata["project_name"]
    stimulus_set_name = session_metadata["stimulus_set_name"]
    session_date = session_metadata["session_date"]
    session_time = session_metadata["session_time"]
    subject = session_metadata["subject"]

    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    session_id = f"{subject}_{project_name}_{stimulus_set_name}_{session_date}_{session_time}"
    nwbfile_path = output_dir_path / f"{session_id}.nwb"

    conversion_options = dict()

    # Add Intan Interface
    intan_recording_interface = IntanRecordingInterface(file_path=intan_file_path, ignore_integrity_checks=True)
    attach_probe_to_recording(recording=intan_recording_interface.recording_extractor)
    if stub_test:
        intan_recording = intan_recording_interface.recording_extractor
        duration = intan_recording.get_duration()
        end_time = min(10.0, duration)
        stubed_recording = intan_recording_interface.recording_extractor.time_slice(start_time=0, end_time=end_time)
        intan_recording_interface.recording_extractor = stubed_recording

    conversion_options["Recording"] = dict(
        iterator_opts={"display_progress": True, "buffer_gb": 5},
    )

    # Behavioral Trials Interface
    behavioral_trials_interface = BehavioralTrialsInterface(file_path=mworks_processed_file_path)

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

    # Build the converter pipe with the previously defined data interfaces
    data_interfaces_dict = {
        "Recording": intan_recording_interface,
        "Behavior": behavioral_trials_interface,
        "Stimuli": stimuli_interface,
    }
    converter_pipe = ConverterPipe(data_interfaces=data_interfaces_dict, verbose=verbose)

    # Parse the string into a datetime object
    datetime_str = f"{session_date} {session_time}"
    datetime_format = "%Y%m%d %H%M%S"
    session_start_time = datetime.strptime(datetime_str, datetime_format).replace(tzinfo=ZoneInfo("US/Eastern"))

    # Add datetime to conversion
    metadata = converter_pipe.get_metadata()
    metadata["NWBFile"]["session_start_time"] = session_start_time
    metadata["NWBFile"]["session_id"] = session_id

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

        sorting = calculate_thresholding_events_from_nwb(
            nwbfile_path=nwbfile_path,
            f_notch=f_notch,
            bandwidth=bandwidth,
            f_low=f_low,
            f_high=f_high,
            noise_threshold=noise_threshold,
            job_kwargs=job_kwargs,
            stub_test=stub_test,
            verbose=verbose,
        )
        nwbfile_path = write_thresholding_events_to_nwb(
            sorting=sorting,
            nwbfile_path=nwbfile_path,
            append=True,
            verbose=verbose,
            thresholindg_pipeline_kwargs=thresholindg_pipeline_kwargs,
        )

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

        # number_of_bins = psth_kwargs.get("num_bins")
        # bins_span_milliseconds = psth_kwargs.get("bins_span_milliseconds")
        bin_width_ms = psth_kwargs.get("bin_width_ms")
        psth_start_time_ms = psth_kwargs.get("psth_start_time_ms")
        psth_end_time_ms = psth_kwargs.get("psth_end_time_ms")


        write_binned_spikes_to_nwbfile(
            nwbfile_path=nwbfile_path,
            bin_width_ms=bin_width_ms,
            psth_start_time_ms=psth_start_time_ms,
            psth_end_time_ms=psth_end_time_ms,
            append=True,
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
        total_scrip_time = total_stop - total_start
        if total_scrip_time <= 60 * 3:
            print(f"Total script took {total_scrip_time:.2f} seconds")
        elif total_scrip_time <= 60 * 60:
            print(f"Total script took {total_scrip_time / 60:.2f} minutes")
        else:
            print(f"Total script took {total_scrip_time / 60 / 60:.2f} hours")

    return nwbfile_path



if __name__ == "__main__":

    projectName = 'test'
    subjectName = 'Apollo'
    sessionDate = '20240701'
    sessionTime = '131011'
    # projectName = 'Monkeyvalence'
    # subjectName = 'Apollo'
    # sessionDate = '20240924'
    # sessionTime = '123021'
    root_dir = '/Users/yoon/raw_data/'

    data_folders = locate_session_paths(root_dir, subjectName, projectName, sessionDate)
    # sort data_folders by session time
    data_folders = sorted(data_folders, key=lambda x: Path(x).name.split('_')[-1])

    project_folder = Path(root_dir) / subjectName / projectName
    processed_folder = project_folder / "processed"
    processed_folder.mkdir(parents=True, exist_ok=True)

    print(f"{len(data_folders)} data folders")
    [print(f) for f in data_folders]


    data_folder = data_folders[0]
    assert data_folder.is_dir(), f"Data directory not found: {data_folder}"

    expt_folder = Path(data_folder).name
    tokens = expt_folder.split("_")
    stimulusSetName = tokens[0]
    folder_sessionDate = tokens[1]
    folder_sessionTime = tokens[2]
    print(f"stimulus_set: {stimulusSetName} | session_date: {folder_sessionDate} | session_time: {folder_sessionTime}")

    stimuli_folder = data_folders[0]
    stub_test = False
    verbose = True


    
    print(f"processing: {data_folder}")
    session_metadata = {
        "project_name": projectName, # SFM
        "stimulus_set_name": stimulusSetName, # normalizers -> SFM_images -> normalizers
        "session_date": sessionDate, # same dates
        "session_time": sessionTime, # different session times
        "subject": subjectName,
    }

    # These two functions is where we encode your data organization structure.
    intan_file_path = data_folder / "info.rhd"
    
    # mworks_processed_file_path = Path(list(data_folder.glob('*.csv'))[0])
    # mworks_processed_file_path = Path("/Users/yoon/raw_data/Apollo/Monkeyvalence/20240924/normalizers_240924_123021/apollo_normalizersV3_240924_123022_mwk.csv")
    mworks_processed_file_path = Path("/Users/yoon/raw_data/Apollo/test/20240701/normalizers_240701_131011/apollo_normalizersV3_240701_131011_mwk.csv")

    # stimulus folder
    mworks_folder_path = "/Users/yoon/Dropbox (MIT)/dorsal_ventral/pipeline_lab/stimuli/RSVP-normalizers-v3"
    stimuli_folder = "/Users/yoon/Dropbox (MIT)/dorsal_ventral/pipeline_lab/stimuli/RSVP-normalizers-v3/images"
    stimulus_ext = 'png'
    stimulus_labels_file_path = "/Users/yoon/Dropbox (MIT)/dorsal_ventral/pipeline_lab/stimuli/RSVP-normalizers-v3/RSVP-normalizers-v3_labels.csv"
    # stimulus_labels = pd.read_csv(get_stimulus_labels(mworks_folder_path))

    thresholindg_pipeline_kwargs = {
    "f_notch": 60.0,  # Frequency for the notch filter
    "bandwidth": 10.0,  # Bandwidth for the notch filter
    "f_low": 300.0,  # Low cutoff frequency for the bandpass filter
    "f_high": 6000.0,  # High cutoff frequency for the bandpass filter
    "noise_threshold": 3,  # Threshold for detection in the thresholding algorithm
    }

    # Ten bins starting 200 ms before the stimulus and spanning 400 ms
    psth_kwargs = {
        "bin_width_ms": 10.0,
        "psth_start_time_ms": -200.0,
        "psth_end_time_ms": 400.0,
    }

    nwbfile_path = convert_session_to_nwb(
        session_metadata=session_metadata,
        intan_file_path=intan_file_path,
        mworks_processed_file_path=mworks_processed_file_path,
        stimuli_folder=stimuli_folder,
        output_dir_path=processed_folder,
        add_thresholding_events=True,
        thresholindg_pipeline_kwargs=thresholindg_pipeline_kwargs,
        add_psth=True,
        psth_kwargs=psth_kwargs,
        stub_test=stub_test,
        verbose=verbose,
    )

    latencies_s = get_unit_latencies_from_reliabilities(nwbfile_path)
    # ### add quality control metrics ###
    # latencies_ms = np.full(n_units, np.nan)
    # latencies_idx = np.full(n_units, np.nan)
    # search_window = [0, 1.5*stim_dur_ms] 
    # search_window_idx = [np.argmin(np.abs(psth_t - t)) for t in search_window]
    # for i in range(n_units):
    #     psth_unit_half1 = psth[i, :, :n_reps_half, :]
    #     psth_unit_half2 = psth[i, :, n_reps_half:, :]
    #     for t in range(n_timebins):
    #         # compute consistency
    #         consistency_r = np.corrcoef(np.nanmean(psth_unit_half1[:,:,t], axis=1), np.nanmean(psth_unit_half2[:,:,t], axis=1))[0, 1]
    #         # Spearman-Brown correction for splitting into halves
    #         n_splits = 2
    #         consistency_r = (n_splits * consistency_r) / (1 + (n_splits-1)*consistency_r)
    #         consistency_samples[i, t] = consistency_r
    #     # find max consistency within first 200 ms
    #     latencies_idx[i] = np.argmax(consistency_samples[i, search_window_idx[0]:search_window_idx[-1]]).astype(int) + stim_onset_idx
    #     latencies_ms[i] = psth_t[int(latencies_idx[i])]   