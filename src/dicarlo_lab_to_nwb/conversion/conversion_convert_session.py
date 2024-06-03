"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import datetime
from pathlib import Path
from typing import Union
from zoneinfo import ZoneInfo

from neuroconv.utils import dict_deep_update, load_dict_from_file

from dicarlo_lab_to_nwb.conversion import ConversionNWBConverter


def session_to_nwb(
    session_intan_raw: Union[str, Path],
    mworks_file_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    stub_test: bool = False,
):

    session_intan_raw = Path(session_intan_raw)
    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    session_id = "subject_identifier_usually"
    nwbfile_path = output_dir_path / f"{session_id}.nwb"

    source_data = dict()
    conversion_options = dict()

    # Add Recording
    # file_path = session_intan_raw / "info.rhd"
    # assert file_path.is_file(), f"Intan raw file not found: {file_path}"
    # source_data.update(dict(Recording=dict(file_path=str(file_path))))
    # conversion_options.update(dict(Recording=dict(stub_test=stub_test)))

    # Add behavior
    assert mworks_file_path.is_file(), f"Mworks file not found: {mworks_file_path}"
    source_data.update(dict(Behavior=dict(file_path=str(mworks_file_path))))

    converter = ConversionNWBConverter(source_data=source_data)

    # Add datetime to conversion
    metadata = converter.get_metadata()
    datetime.datetime(year=2020, month=1, day=1, tzinfo=ZoneInfo("US/Eastern"))
    date = datetime.datetime.today()  # TO-DO: Get this from author
    metadata["NWBFile"]["session_start_time"] = date

    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent / "conversion_metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(metadata, editable_metadata)

    subject_metadata = metadata["Subject"]
    subject_metadata["subject_id"] = "subject_id"

    # Run conversion
    converter.run_conversion(
        metadata=metadata,
        nwbfile_path=nwbfile_path,
        conversion_options=conversion_options,
        overwrite=True,
    )


if __name__ == "__main__":

    # Parameters for conversion
    base_directory = Path("/media/heberto/One Touch/DiCarlo-CN-data-share")
    subject = "pico"
    experiment_directory = base_directory / "exp_domain-transfer-2023" / f"exp_domain-transfer-2023.sub_{subject}"
    assert experiment_directory.is_dir(), f"Experiment directory not found: {experiment_directory}"
    raw_files_directory = experiment_directory / "raw_files"
    assert raw_files_directory.is_dir(), f"Raw files directory not found: {raw_files_directory}"

    session_intan_raw = raw_files_directory / "intanraw" / f"pico_domain-transfer-2023_230214_140610"
    assert session_intan_raw.is_dir(), f"Intan raw session directory not found: {session_intan_raw}"
    mworks_file_path = raw_files_directory / "mworksproc" / f"pico_domain-transfer-2023_230214_140610_mwk.csv"

    data_dir_path = Path("/Directory/With/Raw/Formats/")
    stimuli_dir_path = Path("/Directory/With/Stimuli/Files/")
    output_dir_path = Path.home() / "conversion_nwb"
    stub_test = True

    session_to_nwb(
        session_intan_raw=session_intan_raw,
        mworks_file_path=mworks_file_path,
        output_dir_path=output_dir_path,
        stub_test=stub_test,
    )
