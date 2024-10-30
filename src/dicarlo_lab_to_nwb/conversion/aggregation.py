import datetime
from pathlib import Path
from typing import List, Union

import ndx_binned_spikes
import numpy as np
import pynwb
from pynwb import NWBHDF5IO, NWBFile

from dicarlo_lab_to_nwb.conversion.probe import UtahArrayProbeInterface


def load_nwb_file(file_path: Union[str, Path]) -> NWBFile:
    """Load an NWB file.

    Parameters
    ----------
    file_path : str or Path
        Path to the NWB file to be loaded.

    Returns
    -------
    NWBFile
        The loaded NWB file object.

    Notes
    -----
    This function opens the NWB file in read-only mode using NWBHDF5IO.
    """
    io = NWBHDF5IO(file_path, mode="r")
    return io.read()


def validate_metadata_consistency(metadata_list: List[str], metadata_name: str) -> None:
    """Validate that all metadata values are consistent across files.

    Parameters
    ----------
    metadata_list : list of str
        List of metadata values to check.
    metadata_name : str
        Name of the metadata being checked (for error message).

    Raises
    ------
    ValueError
        If the metadata values are not consistent across all files.
    """
    unique_values = set(metadata_list)
    if len(unique_values) > 1:
        value_counts = {value: metadata_list.count(value) for value in unique_values}
        error_msg = f"Inconsistent {metadata_name} found across files:\n" + "\n".join(
            f"  - {value}: {count} files" for value, count in value_counts.items()
        )
        raise ValueError(error_msg)


def check_that_metadata_matches_across_files(
    file_paths: List[Path], subject_name_list: List[str], project_name_list: List[str], pipeline_version_list: List[str]
) -> None:
    """Check if metadata is consistent across all files.

    Parameters
    ----------
    file_paths : list of Path
        List of paths to the NWB files.
    subject_name_list : list of str
        List of subject names from all files.
    project_name_list : list of str
        List of project names from all files.
    pipeline_version_list : list of str
        List of pipeline versions from all files.

    Raises
    ------
    ValueError
        If any metadata is inconsistent across files, with detailed error message.

    Notes
    -----
    This function checks that subjects, project names, and pipeline versions
    are consistent across all files. If any inconsistencies are found,
    it raises a ValueError with a detailed message showing which files
    have which values.
    """
    try:
        validate_metadata_consistency(subject_name_list, "subject names")
        validate_metadata_consistency(project_name_list, "project names")
        validate_metadata_consistency(pipeline_version_list, "pipeline versions")
    except ValueError as e:
        # Add file information to the error message
        error_details = "\n\nFile details:"
        for i, path in enumerate(file_paths):
            error_details += f"\n{path}:"
            error_details += f"\n  - Subject: {subject_name_list[i]}"
            error_details += f"\n  - Project: {project_name_list[i]}"
            error_details += f"\n  - Pipeline Version: {pipeline_version_list[i]}"
        raise ValueError(str(e) + error_details)


def add_units_table(
    source_nwbfile: NWBFile,
    new_nwbfile: NWBFile,
    session_spike_times_module: pynwb.base.ProcessingModule,
    type_of_data: str,
):
    """Add units table from source file to the new file.

    Parameters
    ----------
    source_nwbfile : NWBFile
        Source NWB file containing the units table to copy
    new_nwbfile : NWBFile
        Target NWB file where the units table will be added
    session_spike_times_module : ProcessingModule
        Processing module where the units table will be stored
    type_of_data : str
        Type of data identifier for naming

    Returns
    -------
    pynwb.misc.Units
        The newly created units table
    """
    units_table = source_nwbfile.units
    session_start_time = source_nwbfile.session_start_time.replace(tzinfo=None)

    name_in_aggregated_table = f"{type_of_data}_{session_start_time}SpikeTimes"
    new_units_table = pynwb.misc.Units(name=name_in_aggregated_table, description=units_table.description)

    # Add to processing module
    session_spike_times_module.add(new_units_table)

    # Transfer columns and data
    units_table_df = units_table.to_dataframe()
    canonical_unit_columns = ["spike_times", "electrodes"]

    # Add non-canonical columns
    for column in units_table_df.columns:
        if column not in canonical_unit_columns:
            new_units_table.add_column(name=column, description="")

    # Add units
    for row in units_table_df.iterrows():
        row_dict = row[1].to_dict()
        row_dict["electrodes"] = row_dict["electrodes"].index.to_numpy()
        new_units_table.add_unit(**row_dict)


def create_destination_nwbfile(output_path: Path, session_id: str, pipeline_version: str = "DiLorean") -> None:
    """Create the initial NWB file with basic metadata.

    Parameters
    ----------
    output_path : Path
        Where to save the initial NWB file
    session_id : str
        Session ID for the file
    pipeline_version : str
        Version identifier for the pipeline
    """
    unit_epoch = int(datetime.datetime.now().timestamp())
    unit_epoch_as_datetime = datetime.datetime.fromtimestamp(unit_epoch)
    datetime_now = datetime.datetime.now()

    nwbfile = NWBFile(
        session_start_time=unit_epoch_as_datetime,
        session_description=pipeline_version,
        identifier=str(datetime_now),
        session_id=session_id,
    )

    # Create processing module for spike times
    nwbfile.create_processing_module(
        name="session_spike_times",
        description="session spike times data",
    )

    # Add Probe Information
    probe_interface = UtahArrayProbeInterface()
    probe_interface.add_to_nwbfile(nwbfile=nwbfile)

    # Save initial file
    with NWBHDF5IO(output_path, mode="w") as io:
        io.write(nwbfile)


def process_single_file(source_path: Path, destination_path: Path) -> dict:
    """Process a single source file and write its data to the destination.

    Parameters
    ----------
    source_path : Path
        Path to source NWB file
    destination_path : Path
        Path to destination NWB file

    Returns
    -------
    dict
        Metadata extracted from the source file
    """
    # Open source file
    with NWBHDF5IO(source_path, mode="r") as source_io:
        source_nwb = source_io.read()

        # Extract metadata
        session_id = source_nwb.session_id
        if session_id is None:
            raise ValueError(f"Session ID not found in {source_path}")

        session_start_time = source_nwb.session_start_time
        session_id_parts = session_id.split("_")
        type_of_data = session_id_parts[-3]
        pipeline_version = session_id_parts[-1]
        subject = session_id_parts[0]
        project_name = session_id_parts[1]

        # Open destination file
        with NWBHDF5IO(destination_path, mode="a") as dest_io:
            dest_nwb = dest_io.read()

            # Add PSTH data
            file_psth = source_nwb.scratch["psth_pipeline_format"]
            if type_of_data == "session":
                name = f"psth_session_data_{session_start_time.replace(tzinfo=None)}"
            else:
                name = f"psth_normalizers_{session_start_time.replace(tzinfo=None)}"

            dest_nwb.add_scratch(file_psth.data[:], name=name, description=file_psth.description)

            # Add units table
            add_units_table(source_nwb, dest_nwb, dest_nwb.processing["session_spike_times"], type_of_data)

            dest_io.write(dest_nwb)

    return {
        "subject": subject,
        "project_name": project_name,
        "pipeline_version": pipeline_version,
    }


def concatenate_nwb_files(
    file_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    pipeline_version: str = "DiLorean",
) -> Path:
    """Concatenate multiple NWB files into a single file.

    This function processes multiple NWB files and combines their data into a single file
    while maintaining memory efficiency. It performs the following steps:
    1. Creates a new NWB file with metadata from the first file
    2. Processes each source file individually, appending its data to the output file
    3. Validates metadata consistency across all files
    4. Creates and appends a concatenated PSTH combining all session data

    Parameters
    ----------
    file_paths : list of str or Path
        List of paths to the NWB files that should be concatenated. The files should follow
        the naming convention: {subject}_{project_name}_{session_date}_{session_time}_{type_of_data}_data_{pipeline_version}
    output_dir : str or Path
        Directory where the output file will be saved. The output filename will follow the format:
        {subject}_{project_name}_{pipeline_version}.nwb
    pipeline_version : str, optional
        Version identifier for the pipeline, by default "DiLorean"

    Returns
    -------
    Path
        Path to the created output file

    Notes
    -----
    The function preserves memory efficiency by:
    - Processing one source file at a time
    - Writing data to disk immediately after processing
    - Only loading session PSTHs when needed for concatenation

    The output file will contain:
    - Individual PSTHs from each input file
    - Concatenated PSTH combining all session data (axis=2)
    - Units tables from each input file
    - Probe information
    - Session information

    The function expects all input files to have consistent metadata:
    - Same subject name
    - Same project name
    - Same pipeline version

    Raises
    ------
    ValueError
        - If session IDs are missing from input files
        - If metadata is inconsistent across files
        - If files are not in the expected pipeline format
    """
    file_paths = [Path(p) for p in file_paths]
    output_dir = Path(output_dir)

    # Lists to store metadata
    metadata_list = []

    print(f"Found {len(file_paths)} NWB files to process")

    # Get metadata from first file to set up destination file
    with NWBHDF5IO(file_paths[0], mode="r") as source_io:
        source_nwb = source_io.read()
        session_id_parts = source_nwb.session_id.split("_")
        subject = session_id_parts[0]
        project_name = session_id_parts[1]
        pipeline_version = session_id_parts[-1]
        output_filename = f"{subject}_{project_name}_{pipeline_version}.nwb"
        final_session_id = f"{subject}_{project_name}_{pipeline_version}"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    # Create destination file with proper session ID
    create_destination_nwbfile(output_path, final_session_id, pipeline_version)

    # Process each file
    from tqdm.auto import tqdm

    for file_path in tqdm(file_paths, desc="Processing files", unit="file"):
        tqdm.write(f"Processing: {file_path.name}")
        metadata = process_single_file(file_path, output_path)
        metadata_list.append(metadata)

    # Extract metadata lists for validation
    subject_name_list = [m["subject"] for m in metadata_list]
    project_name_list = [m["project_name"] for m in metadata_list]
    pipeline_version_list = [m["pipeline_version"] for m in metadata_list]

    # Validate metadata consistency
    check_that_metadata_matches_across_files(
        file_paths=file_paths,
        subject_name_list=subject_name_list,
        project_name_list=project_name_list,
        pipeline_version_list=pipeline_version_list,
    )

    # Final pass: Read all session PSTHs and add concatenated data
    tqdm.write("Concatenating session PSTHs...")
    with NWBHDF5IO(output_path, mode="a") as io:
        nwbfile = io.read()

        # Collect all session PSTHs
        session_psths = []
        for key in nwbfile.scratch.keys():
            if key.startswith("psth_session_data_") and not key.endswith("concatenated"):
                session_psths.append(nwbfile.scratch[key].data[:])

        # Create and add concatenated PSTH
        concatenated_psth = np.concatenate(session_psths, axis=2)
        nwbfile.add_scratch(
            concatenated_psth,
            name="psth_session_data_concatenated",
            description="Concatenated PSTH from multiple files",
        )

        io.write(nwbfile)

    return output_path


if __name__ == "__main__":
    # Example file paths
    data_folder = Path("/media/heberto/One Touch/DiCarlo-CN-data-share")
    output_dir = data_folder / "nwb_files"

    # Get list of NWB files
    nwb_files = [path for path in output_dir.iterdir() if path.suffix == ".nwb"]

    # Concatenate files
    output_path = concatenate_nwb_files(file_paths=nwb_files, output_dir=output_dir)

    # Verify the output
    concatenated_file = load_nwb_file(output_path)
    print(f"Successfully created concatenated file at: {output_path}")
    print(f"Concatenated PSTH shape: {concatenated_file.scratch['psth_session_data_concatenated'].data.shape}")
