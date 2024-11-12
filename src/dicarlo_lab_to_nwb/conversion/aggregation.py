import datetime
from pathlib import Path
from typing import List, Optional, Union

import ndx_binned_spikes
import numpy as np
import pynwb
from pynwb import NWBHDF5IO, NWBFile
from tqdm.auto import tqdm

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
    dest_nwbfile: NWBFile,
    is_normalizer: bool,
):
    """Add units table from source file to the new file.

    Parameters
    ----------
    source_nwbfile : NWBFile
        Source NWB file containing the units table to copy
    dest_nwbfile : NWBFile
        Target NWB file where the units table will be added
    is_normalizer : str
        Whether the source file is a normalizer

    Returns
    -------
    pynwb.misc.Units
        The newly created units table
    """
    units_table = source_nwbfile.units
    session_start_time = source_nwbfile.session_start_time.replace(tzinfo=None)

    type_of_data = "normalizers" if is_normalizer else "session_data"
    name_in_aggregated_table = f"spike_times_{type_of_data}_{session_start_time}"
    new_units_table = pynwb.misc.Units(name=name_in_aggregated_table, description=units_table.description)

    # Add to processing module
    session_spikes_times_module = dest_nwbfile.processing["session_spike_times"]
    session_spikes_times_module.add(new_units_table)

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


def add_trials_table(
    source_nwbfile: NWBFile,
    dest_nwbfile: NWBFile,
    is_normalizer: bool,
):
    """Add trials table from source file to the new file.

    Parameters
    ----------
    source_nwbfile : NWBFile
        Source NWB file containing the trials table to copy
    dest_nwbfile : NWBFile
        Target NWB file where the trials table will be added
    is_normalizer: bool,
        Whether the source file is a normalizer
    Returns
    -------
    pynwb.file.TimeIntervals
        The newly created trials table
    """
    trials_table = source_nwbfile.trials
    if trials_table is None:
        return None

    session_start_time = source_nwbfile.session_start_time.replace(tzinfo=None)

    type_of_data = "normalizers" if is_normalizer else "session_data"
    name_in_aggregated_table = f"trials_table_{type_of_data}_{session_start_time}"

    # Create new trials table
    trials_df = trials_table.to_dataframe()

    # Create the trials table in destination file
    new_trials_table = dest_nwbfile.create_time_intervals(
        name=name_in_aggregated_table, description=trials_table.description or "Trial data from source file"
    )

    # Add all columns from source trials table
    for column in trials_df.columns:
        if column not in ["start_time", "stop_time", "tags"]:  # These are built-in columns
            new_trials_table.add_column(
                name=column,
                description="",
            )

    # Add units
    for row in trials_df.iterrows():
        row_dict = row[1].to_dict()
        new_trials_table.add_row(**row_dict)

    return new_trials_table


def propagate_session_data_to_aggregate_nwbfile(source_path: Path, destination_path: Path) -> dict:
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
        data_collection = source_nwb.data_collection
        if session_id is None:
            raise ValueError(f"Session ID not found in {source_path}")

        session_start_time = source_nwb.session_start_time.replace(tzinfo=None)

        # session_id = f"{project_name_camel_case}_{subject}_{stimulus_name_camel_case}_{session_date}_{session_time}_{pipeline_version}_thresholded"

        session_id_parts = session_id.split("_")

        pipeline_version = session_id_parts[-1]
        subject = session_id_parts[0]
        project_name = session_id_parts[1]

        is_normalizer = data_collection == "normalizer"
        # Open destination file
        with NWBHDF5IO(destination_path, mode="a") as dest_io:
            dest_nwb = dest_io.read()

            # Add PSTH data
            file_psth = source_nwb.scratch["psth_pipeline_format"]
            if is_normalizer:
                name = f"psth_normalizers_{session_start_time}"
            else:
                name = f"psth_session_data_{session_start_time}"

            dest_nwb.add_scratch(file_psth.data[:], name=name, description=file_psth.description)

            # Add units table
            add_units_table(source_nwb, dest_nwb, is_normalizer)

            # Add trials table
            add_trials_table(source_nwb, dest_nwb, is_normalizer)

            dest_io.write(dest_nwb)

    return {
        "subject": subject,
        "project_name": project_name,
        "pipeline_version": pipeline_version,
    }


def create_destination_nwbfile(output_path: Path, session_id: str) -> None:
    """Create the initial NWB file with basic metadata.

    Parameters
    ----------
    output_path : Path
        Where to save the initial NWB file
    session_id : str
        Session ID for the file
    """
    datetime_now = datetime.datetime.now()

    pipeline_version = session_id.split("_")[-1]

    nwbfile = NWBFile(
        session_start_time=datetime_now,
        session_description=pipeline_version,
        session_id=session_id,
        identifier=session_id,
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


def aggregate_nwbfiles(
    file_paths: List[Union[str, Path]],
    output_folder_path: Union[str, Path],
    pipeline_version: str = "DiLorean",
    is_unit_valid: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Path:
    """
    Aggregate multiple NWB files into a single file.

    This function processes the session and normalizer data from single sessions and aggregates them into a single NWB file.
    that contains the following:

    - Individual PSTHs from each input file
    - Concatenated PSTH combining all session data across stimuli presentation
    - Units tables (spike or thresholded data) from each input file
    - Trial tables from each input file

    Parameters
    ----------
    file_paths : list of str or Path
        List of paths to the NWB files that should be concatenated. The files should follow
        the naming convention: {subject}_{project_name}_{session_date}_{session_time}_{type_of_data}_data_{pipeline_version}
    output_folder_path : str or Path
        Directory where the output file will be saved. The output filename will follow the format:
        {subject}_{project_name}_{pipeline_version}.nwb
    pipeline_version : str, optional
        Version identifier for the pipeline, by default "DiLorean"
    is_unit_valid : numpy.ndarray, optional
        Boolean array indicating which units to include in the concatenated PSTH
    verbose : bool, optional
        Whether to print progress messages, by default True

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
    - Trial tables from each input file
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
    output_folder_path = Path(output_folder_path)

    # Lists to store metadata
    metadata_list = []

    if verbose:
        print(f"Found {len(file_paths)} NWB files to process")

    # Get metadata from first file to set up destination file
    with NWBHDF5IO(file_paths[0], mode="r") as source_io:
        source_nwb = source_io.read()
        session_id_parts = source_nwb.session_id.split("_")
        project_name = session_id_parts[0]
        subject = session_id_parts[1]
        pipeline_version = session_id_parts[-1]
        final_session_id = f"{subject}_{project_name}_{pipeline_version}"

    # Create output directory if it doesn't exist
    output_folder_path.mkdir(parents=True, exist_ok=True)
    output_filename = f"{final_session_id}.nwb"

    aggregated_nwbfile_path = output_folder_path / output_filename

    # Create destination file with proper session ID
    create_destination_nwbfile(aggregated_nwbfile_path, final_session_id)

    # Process each file

    for file_path in tqdm(file_paths, desc="Processing files", unit="file", disable=not verbose):
        if verbose:
            tqdm.write(f"Processing: {file_path.name}")
        metadata = propagate_session_data_to_aggregate_nwbfile(file_path, aggregated_nwbfile_path)
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
    if verbose:
        tqdm.write("Concatenating session PSTHs...")
    with NWBHDF5IO(aggregated_nwbfile_path, mode="a") as io:
        nwbfile = io.read()

        # Collect all session PSTHs
        session_psths = []
        for key in nwbfile.scratch.keys():
            if key.startswith("psth_session_data_") and not key.endswith("concatenated"):
                session_psths.append(nwbfile.scratch[key].data[:])

        # Create and add concatenated PSTH
        concatenated_psth = np.concatenate(session_psths, axis=2)

        # Filter units using is_unit_valid array
        if is_unit_valid is not None:
            number_of_units = concatenated_psth.shape[0]
            if number_of_units != is_unit_valid.shape[0]:
                raise ValueError(
                    f"Dimension mismatch in unit validation arrays:\n"
                    f"Number of units in PSTH: {number_of_units}\n"
                    f"Number of units in validity array: {is_unit_valid.shape[0]}\n"
                    f"These dimensions must match for proper unit-wise validation.\n"
                    "Please ensure both arrays contain the same number of units."
                )
            concatenated_psth = concatenated_psth[is_unit_valid, ...]

        nwbfile.add_scratch(
            concatenated_psth,
            name="psth_session_data_concatenated",
            description="Concatenated PSTH from multiple files",
        )

        io.write(nwbfile)

    return aggregated_nwbfile_path


def split_aggregated_nwbfile_train_test(
    aggregated_nwbfile_path: Union[str, Path],
    is_stimulus_test: np.ndarray,
) -> dict:
    """Split an aggregated NWB file into training and testing files.

    Parameters
    ----------
    aggregated_nwbfile_path : str or Path
        Path to the aggregated NWB file
    is_stimulus_test : np.ndarray
        Boolean array indicating which stimuli are in the testing set
    Returns
    -------
    dict
        Dictionary containing the paths to the training and testing files
    """
    aggregated_nwbfile_path = Path(aggregated_nwbfile_path)

    with NWBHDF5IO(aggregated_nwbfile_path, mode="r") as io:
        nwbfile = io.read()
        concatenated_psth = nwbfile.scratch["psth_session_data_concatenated"].data[:]
        session_start_time = nwbfile.session_start_time
        session_id = nwbfile.session_id
        session_description = nwbfile.session_description

    psth_data_test = concatenated_psth[:, is_stimulus_test, ...]
    psth_data_train = concatenated_psth[:, ~is_stimulus_test, ...]

    nwbfile_train = NWBFile(
        session_start_time=session_start_time,
        session_description=session_description,
        session_id=session_id,
        identifier=f"{session_id}_train",
    )

    nwbfile_test = NWBFile(
        session_start_time=session_start_time,
        session_description=session_description,
        session_id=session_id,
        identifier=f"{session_id}_test",
    )

    nwbfile_train.add_scratch(
        psth_data_train,
        name="psth_session_data_concatenated",
        description="Concatenated PSTH from multiple files (training data)",
    )

    nwbfile_test.add_scratch(
        psth_data_test,
        name="psth_session_data_concatenated",
        description="Concatenated PSTH from multiple files (testing data)",
    )

    # Save training and testing files
    output_folder_path = aggregated_nwbfile_path.parent
    nwbfile_train_path = output_folder_path / f"{aggregated_nwbfile_path.stem}_train.nwb"

    with NWBHDF5IO(nwbfile_train_path, mode="w") as io:
        io.write(nwbfile_train)

    nwbfile_test_path = output_folder_path / f"{aggregated_nwbfile_path.stem}_test.nwb"
    with NWBHDF5IO(nwbfile_test_path, mode="w") as io:
        io.write(nwbfile_test)

    return {
        "train": nwbfile_train_path,
        "test": nwbfile_test_path,
    }
