"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from neuroconv.utils import dict_deep_update, load_dict_from_file

from dicarlo_lab_to_nwb.conversion import ConversionNWBConverter
from dicarlo_lab_to_nwb.conversion.data_locator import (
    locate_intan_file_path,
    locate_mworks_processed_file_path,
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
):

    start = time.time()

    intan_file_path = Path(intan_file_path)
    mworks_processed_file_path = Path(mworks_processed_file_path)
    stimuli_folder = Path(stimuli_folder)

    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    session_id = f"{subject}_{session_date}_{session_time}"
    nwbfile_path = output_dir_path / f"{session_id}.nwb"

    source_data = dict()
    conversion_options = dict()

    # Add Recording
    source_data["Recording"] = dict(file_path=intan_file_path, ignore_integrity_checks=True)
    conversion_options["Recording"] = dict(
        stub_test=stub_test,
        iterator_opts={"display_progress": True, "buffer_gb": 5},
    )

    # Add behavior
    source_data["Behavior"] = dict(file_path=mworks_processed_file_path)

    # Add stimuli
    source_data["Stimuli"] = dict(
        file_path=mworks_processed_file_path,
        folder_path=stimuli_folder,
        image_set_name=image_set_name,
    )

    converter = ConversionNWBConverter(source_data=source_data)

    # Parse the string into a datetime object
    datetime_str = f"{session_date} {session_time}"
    datetime_format = "%Y%m%d %H%M%S"
    session_start_time = datetime.strptime(datetime_str, datetime_format).replace(tzinfo=ZoneInfo("US/Eastern"))

    # Add datetime to conversion
    metadata = converter.get_metadata()
    metadata["NWBFile"]["session_start_time"] = session_start_time

    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent / "metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(metadata, editable_metadata)

    subject_metadata = metadata["Subject"]
    subject_metadata["subject_id"] = f"{subject}"

    # Run conversion
    converter.run_conversion(
        metadata=metadata,
        nwbfile_path=nwbfile_path,
        conversion_options=conversion_options,
        overwrite=True,
    )

    stop_time = time.time()
    conversion_time_seconds = stop_time - start
    if conversion_time_seconds <= 60 * 3:
        print(f"Conversion took {conversion_time_seconds:.2f} seconds")
    elif conversion_time_seconds <= 60 * 60:
        print(f"Conversion took {conversion_time_seconds / 60:.2f} minutes")
    else:
        print(f"Conversion took {conversion_time_seconds / 60 / 60:.2f} hours")


if __name__ == "__main__":

    image_set_name = "domain-transfer-2023"
    subject = "pico"
    session_date = "20230215"
    session_time = "161322"

    # This one has a jump in time
    # session_date = "20230214"
    # session_time = "140610"

    data_folder = Path("/media/heberto/One Touch/DiCarlo-CN-data-share")
    assert data_folder.is_dir(), f"Data directory not found: {data_folder}"

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

    stimuli_folder = data_folder / "StimulusSets"
    assert stimuli_folder.is_dir(), f"Stimuli folder not found: {stimuli_folder}"

    output_dir_path = Path.home() / "conversion_nwb"
    stub_test = True

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
    )
