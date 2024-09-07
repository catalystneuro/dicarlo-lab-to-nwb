"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from neuroconv import ConverterPipe
from neuroconv.datainterfaces import IntanRecordingInterface
from neuroconv.utils import dict_deep_update, load_dict_from_file

from dicarlo_lab_to_nwb.conversion.behaviorinterface import BehavioralTrialsInterface
from dicarlo_lab_to_nwb.conversion.data_locator import (
    locate_intan_file_path,
    locate_mworks_processed_file_path,
)
from dicarlo_lab_to_nwb.conversion.pipeline import (
    calculate_thresholding_events_from_nwb,
    write_thresholding_events_to_nwb,
)
from dicarlo_lab_to_nwb.conversion.probe import attach_probe_to_recording
from dicarlo_lab_to_nwb.conversion.psth import write_binned_spikes_to_nwbfile
from dicarlo_lab_to_nwb.conversion.stimuli_interface import (
    StimuliImagesInterface,
    StimuliVideoInterface,
)


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

    image_set_name = session_metadata["image_set_name"]
    session_date = session_metadata["session_date"]
    session_time = session_metadata["session_time"]
    subject = session_metadata["subject"]

    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    session_id = f"{subject}_{session_date}_{session_time}"
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
            image_set_name=image_set_name,
            video_copy_path=output_dir_path / "videos",
            verbose=verbose,
        )
    else:
        stimuli_interface = StimuliImagesInterface(
            file_path=mworks_processed_file_path,
            folder_path=stimuli_folder,
            image_set_name=image_set_name,
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

        number_of_bins = psth_kwargs.get("num_bins")
        bins_span_milliseconds = psth_kwargs.get("bins_span_milliseconds")
        bin_width_in_milliseconds = bins_span_milliseconds / number_of_bins
        milliseconds_from_event_to_first_bin = psth_kwargs.get("milliseconds_from_event_to_first_bin", None)

        write_binned_spikes_to_nwbfile(
            nwbfile_path=nwbfile_path,
            bin_width_in_milliseconds=bin_width_in_milliseconds,
            number_of_bins=number_of_bins,
            milliseconds_from_event_to_first_bin=milliseconds_from_event_to_first_bin,
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


if __name__ == "__main__":

    image_set_name = "domain-transfer-2023"
    subject = "pico"
    session_date = "20230215"
    session_time = "161322"

    # This one has a jump in time
    session_date = "20230214"
    session_time = "140610"

    # Third one
    # session_date = "20230216"
    # session_time = "150919"

    # Fourth one
    # session_date = "20230221"
    # session_time = "130510"

    # Video one (does not have intan)
    # image_set_name = "Co3D"
    # subject = "pico"
    # session_date = "230627"
    # session_time = "114317"

    data_folder = Path("/media/heberto/One Touch/DiCarlo-CN-data-share")
    assert data_folder.is_dir(), f"Data directory not found: {data_folder}"

    session_metadata = {
        "image_set_name": image_set_name,
        "session_date": session_date,
        "session_time": session_time,
        "subject": subject,
    }

    intan_file_path = locate_intan_file_path(data_folder=data_folder, **session_metadata)

    mworks_processed_file_path = locate_mworks_processed_file_path(
        data_folder=data_folder,
        image_set_name=image_set_name,
        subject=subject,
        session_date=session_date,
        session_time=session_time,
    )

    stimuli_folder = data_folder / "StimulusSets" / "RSVP-domain_transfer" / "images"
    # stimuli_folder = data_folder / "StimulusSets" / "Co3D" / "videos_mworks"
    # assert stimuli_folder.is_dir(), f"Stimuli folder not found: {stimuli_folder}"

    output_dir_path = Path.home() / "conversion_nwb"
    stub_test = True
    verbose = True
    add_thresholding_events = True
    add_psth = True

    thresholindg_pipeline_kwargs = {
        "f_notch": 60.0,  # Frequency for the notch filter
        "bandwidth": 10.0,  # Bandwidth for the notch filter
        "f_low": 300.0,  # Low cutoff frequency for the bandpass filter
        "f_high": 6000.0,  # High cutoff frequency for the bandpass filter
        "noise_threshold": 3,  # Threshold for detection in the thresholding algorithm
    }

    # Ten bins starting 200 ms before the stimulus and spanning 400 ms
    psth_kwargs = {"bins_span_milliseconds": 400, "num_bins": 10, "milliseconds_from_event_to_first_bin": -200.0}

    from dicarlo_lab_to_nwb.conversion.convert_session import convert_session_to_nwb

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
    )
