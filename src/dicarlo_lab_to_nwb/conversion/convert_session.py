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
from dicarlo_lab_to_nwb.conversion.psth import write_psth_to_nwbfile
from dicarlo_lab_to_nwb.conversion.stimuli_interface import (
    StimuliImagesInterface,
    StimuliVideoInterface,
)


def session_to_nwb(
    image_set_name: str,
    subject: str,
    session_date: str,
    session_time: str,
    intan_file_path: str | Path,
    mworks_processed_file_path: str | Path,
    stimuli_folder: str | Path,
    output_dir_path: str | Path,
    stub_test: bool = False,
    verbose: bool = False,
    add_thresholding_events: bool = True,
    add_psth: bool = True,
    stimuli_are_video: bool = False,
):
    if verbose:
        start = time.time()

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
    conversion_options["Recording"] = dict(
        stub_test=stub_test,
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
        )
    else:
        stimuli_interface = StimuliImagesInterface(
            file_path=mworks_processed_file_path,
            folder_path=stimuli_folder,
            image_set_name=image_set_name,
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

        chunk_duration = 10.0  # This is fixed
        job_kwargs = dict(n_jobs=-1, progress_bar=True, verbose=verbose, chunk_duration=chunk_duration)

        sorting = calculate_thresholding_events_from_nwb(
            nwbfile_path=nwbfile_path,
            job_kwargs=job_kwargs,
            stub_test=stub_test,
        )
        write_thresholding_events_to_nwb(sorting=sorting, nwbfile_path=nwbfile_path, append=True)

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

        # Make 10 bins of 0.200 seconds each
        number_of_bins = 10
        bin_width_in_milliseconds = 400.0 / number_of_bins
        # This means the first bin starts 200 ms before the image presentation
        milliseconds_from_event_to_first_bin = -200.0  #

        write_psth_to_nwbfile(
            nwbfile_path=nwbfile_path,
            bin_width_in_milliseconds=bin_width_in_milliseconds,
            number_of_bins=number_of_bins,
            milliseconds_from_event_to_first_bin=milliseconds_from_event_to_first_bin,
            append=True,
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


if __name__ == "__main__":

    image_set_name = "domain-transfer-2023"
    subject = "pico"
    session_date = "20230215"
    session_time = "161322"

    # This one has a jump in time
    session_date = "20230214"
    session_time = "140610"

    # Third one
    session_date = "20230216"
    session_time = "150919"

    # Fourth one
    session_date = "20230221"
    session_time = "130510"

    # Video one (does not have intan)
    # image_set_name = "Co3D"
    # subject = "pico"
    # session_date = "230627"
    # session_time = "114317"

    data_folder = Path("/media/heberto/One Touch/DiCarlo-CN-data-share")
    assert data_folder.is_dir(), f"Data directory not found: {data_folder}"

    intan_file_path = None
    intan_file_path = locate_intan_file_path(
        data_folder=data_folder,
        image_set_name=image_set_name,
        subject=subject,
        session_date=session_date,
        session_time=session_time,
    )

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
    stub_test = False
    verbose = True
    add_thresholding_events = True
    add_psht = True

    session_to_nwb(
        image_set_name=image_set_name,
        subject=subject,
        session_date=session_date,
        session_time=session_time,
        intan_file_path=intan_file_path,
        mworks_processed_file_path=mworks_processed_file_path,
        stimuli_folder=stimuli_folder,
        output_dir_path=output_dir_path,
        stub_test=stub_test,
        verbose=verbose,
        add_thresholding_events=add_thresholding_events,
        add_psth=True,
    )
