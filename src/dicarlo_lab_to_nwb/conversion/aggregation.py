import datetime
from pathlib import Path
from typing import List, Optional, Union

import ndx_binned_spikes
import numpy as np
import pandas as pd
import pynwb
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.data_utils import AbstractDataChunkIterator, DataChunk
from pynwb import NWBHDF5IO, NWBFile
from pynwb.core import ScratchData
from tqdm.auto import tqdm

from dicarlo_lab_to_nwb.conversion.probe import UtahArrayProbeInterface


def choose_psth_chunk_shape(shape: tuple, target_bytes: int = 4 * 2**20, itemsize: int = 8) -> tuple:
    """
    Pick the HDF5 chunk shape for a PSTH of `shape` (units, stimuli, repetitions, bins).

    Repetitions and bins are small, so a chunk holds them whole and spans a block of units and stimuli. The
    stimulus block is chosen to divide the axis almost evenly: HDF5 stores the ragged last chunk of every row
    in full, and the default that hdmf derives from the write blocks wasted 20 percent of the file that way
    (a block of 630 over an axis of 2521 leaves a chunk holding one column).

    Returns
    -------
    tuple
        The chunk shape.
    """
    number_of_units, number_of_stimuli, repetitions, number_of_bins = shape
    bytes_per_unit_stimulus = repetitions * number_of_bins * itemsize

    best_chunk, best_score = None, None
    for unit_block in (1, 2, 4, 8, 16, 24, 32, 48):
        if number_of_units % unit_block:
            continue
        for divisions in range(1, 41):
            stimulus_block = int(np.ceil(number_of_stimuli / divisions))
            chunk_bytes = unit_block * stimulus_block * bytes_per_unit_stimulus
            if not (2**20 <= chunk_bytes <= 8 * 2**20):
                continue
            padding = (divisions * stimulus_block - number_of_stimuli) / (divisions * stimulus_block)
            score = (padding, abs(chunk_bytes - target_bytes))
            if best_score is None or score < best_score:
                best_chunk, best_score = (unit_block, stimulus_block, repetitions, number_of_bins), score

    # Nothing in range for a small PSTH: one chunk holds all of it
    return best_chunk or shape


class SessionPSTHBlocks(AbstractDataChunkIterator):
    """
    One session's PSTH, copied into the aggregated file one block at a time.

    Reading it whole with `data[:]` holds it in memory until the aggregated file is written, which is 3 GiB
    for a session of the videos project and was the largest allocation of the aggregation.

    Parameters
    ----------
    session_psth_dataset
        The session's PSTH dataset, shaped (units, stimuli, repetitions, bins).
    bytes_per_block : int
        Rough size of each block read and written.
    """

    def __init__(self, session_psth_dataset, bytes_per_block: int = 256 * 2**20):
        self._dataset = session_psth_dataset
        self._shape = tuple(session_psth_dataset.shape)
        self._dtype = np.dtype(session_psth_dataset.dtype)

        number_of_units, number_of_stimuli, repetitions, number_of_bins = self._shape
        bytes_per_stimulus = number_of_units * repetitions * number_of_bins * self._dtype.itemsize
        self._stimuli_per_block = max(1, min(number_of_stimuli, bytes_per_block // max(1, bytes_per_stimulus)))
        self._starts = list(range(0, number_of_stimuli, self._stimuli_per_block))
        self._block_index = 0

    def __iter__(self):
        return self

    def __next__(self) -> DataChunk:
        if self._block_index >= len(self._starts):
            raise StopIteration
        start = self._starts[self._block_index]
        stop = min(start + self._stimuli_per_block, self._shape[1])
        self._block_index += 1
        return DataChunk(
            data=self._dataset[:, start:stop, :, :],
            selection=(slice(None), slice(start, stop), slice(None), slice(None)),
        )

    def recommended_chunk_shape(self) -> None:
        return None

    def recommended_data_shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def maxshape(self) -> tuple:
        return self._shape


class ConcatenatedSessionPSTH(AbstractDataChunkIterator):
    """
    The session PSTHs joined along the repetition axis, written one block at a time.

    Each session's PSTH is read from the aggregated file only as its block is written, so the joined array is
    never held in memory. Reading them all and calling `np.concatenate` peaked at 6.17 GB for a single day,
    where one session's PSTH is already 3 GiB, and it grows with every session added to a project.

    Parameters
    ----------
    session_psth_datasets : list
        The per-session PSTH datasets, shaped (units, stimuli, repetitions, bins). They must agree on every
        axis except the repetitions, which is the one they are joined on.
    unit_indices : np.ndarray, optional
        The units to keep, as indices. All of them when this is None.
    bytes_per_block : int
        Rough size of each block read and written.
    """

    def __init__(
        self,
        session_psth_datasets: List,
        unit_indices: Optional[np.ndarray] = None,
        bytes_per_block: int = 256 * 2**20,
    ):
        number_of_units, number_of_stimuli, _, number_of_bins = session_psth_datasets[0].shape
        for dataset in session_psth_datasets[1:]:
            if (
                dataset.shape[0] != number_of_units
                or dataset.shape[1] != number_of_stimuli
                or dataset.shape[3] != number_of_bins
            ):
                raise ValueError(
                    "The session PSTHs must agree on units, stimuli and bins to be joined on the repetition "
                    f"axis, and these do not: {[dataset.shape for dataset in session_psth_datasets]}."
                )

        self._datasets = session_psth_datasets
        self._unit_selection = slice(None) if unit_indices is None else list(unit_indices)
        self._number_of_units = number_of_units if unit_indices is None else len(unit_indices)
        self._number_of_stimuli = number_of_stimuli
        self._number_of_bins = number_of_bins
        self._dtype = np.dtype(session_psth_datasets[0].dtype)

        repetitions_per_session = [dataset.shape[2] for dataset in session_psth_datasets]
        self._repetition_offsets = np.cumsum([0] + repetitions_per_session)

        bytes_per_stimulus = (
            self._number_of_units * max(repetitions_per_session) * number_of_bins * self._dtype.itemsize
        )
        stimuli_per_block = max(1, min(number_of_stimuli, bytes_per_block // max(1, bytes_per_stimulus)))
        self._blocks = [
            (session_index, start, min(start + stimuli_per_block, number_of_stimuli))
            for session_index in range(len(session_psth_datasets))
            for start in range(0, number_of_stimuli, stimuli_per_block)
        ]
        self._block_index = 0

    def __iter__(self):
        return self

    def __next__(self) -> DataChunk:
        if self._block_index >= len(self._blocks):
            raise StopIteration
        session_index, start, stop = self._blocks[self._block_index]
        self._block_index += 1

        block = self._datasets[session_index][self._unit_selection, start:stop, :, :]
        selection = (
            slice(None),
            slice(start, stop),
            slice(int(self._repetition_offsets[session_index]), int(self._repetition_offsets[session_index + 1])),
            slice(None),
        )
        return DataChunk(data=block, selection=selection)

    def recommended_chunk_shape(self) -> None:
        return None

    def recommended_data_shape(self) -> tuple:
        return self.maxshape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def maxshape(self) -> tuple:
        return (
            self._number_of_units,
            self._number_of_stimuli,
            int(self._repetition_offsets[-1]),
            self._number_of_bins,
        )


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
    session_start_time = source_nwbfile.session_start_time.strftime("%Y-%m-%dT%H-%M-%S")

    type_of_data = "normalizers" if is_normalizer else "session_data"
    name_in_aggregated_table = f"spike_times_{type_of_data}_{session_start_time}"
    new_units_table = pynwb.misc.Units(name=name_in_aggregated_table, description=units_table.description)

    # Add to processing module
    session_spikes_times_module = dest_nwbfile.processing["session_spike_times"]
    session_spikes_times_module.add(new_units_table)

    # Transfer the data column by column. Adding the units row by row with add_unit stores every spike time
    # as a separate Python object, which takes about ten times the memory of the spike times themselves
    number_of_units = len(units_table)
    new_units_table.id.extend(list(range(number_of_units)))

    # Add non-canonical columns
    canonical_unit_columns = ["spike_times", "electrodes"]
    for column in units_table.colnames:
        if column not in canonical_unit_columns:
            new_units_table.add_column(name=column, description="", data=units_table[column][:])

    spike_times_index = units_table["spike_times"]
    new_units_table.add_column(
        name="spike_times",
        description=spike_times_index.target.description,
        data=spike_times_index[:],
        index=True,
    )

    electrodes_index = units_table["electrodes"]
    electrodes_end_indices = electrodes_index.data[:]
    electrode_indices = np.split(electrodes_index.target.data[:], electrodes_end_indices[:-1])
    new_units_table.add_column(
        name="electrodes",
        description=electrodes_index.target.description,
        data=electrode_indices,
        index=True,
        table=dest_nwbfile.electrodes,
    )


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

    session_start_time = source_nwbfile.session_start_time.strftime("%Y-%m-%dT%H-%M-%S")

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

        session_start_time = source_nwb.session_start_time.strftime("%Y-%m-%dT%H-%M-%S")

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

            # Streamed from the source file rather than read whole: one session's PSTH is 3 GiB
            session_psth_blocks = SessionPSTHBlocks(session_psth_dataset=file_psth.data)
            dest_nwb.add_scratch(
                ScratchData(
                    name=name,
                    data=H5DataIO(
                        data=session_psth_blocks,
                        chunks=choose_psth_chunk_shape(session_psth_blocks.maxshape),
                        compression="gzip",
                        compression_opts=4,
                    ),
                    description=file_psth.description,
                )
            )

            # Add PSTH time bins
            psth_bin_width_s = (1 / 1000.0) * source_nwb.processing["ecephys"][
                "BinnedAlignedSpikesToStimulus"
            ].bin_width_in_ms
            psth_start_s = (1 / 1000.0) * source_nwb.processing["ecephys"][
                "BinnedAlignedSpikesToStimulus"
            ].event_to_bin_offset_in_ms
            psth_num_bins = source_nwb.processing["ecephys"]["BinnedAlignedSpikesToStimulus"].number_of_bins
            psth_timebins_s = np.round(
                np.arange(psth_start_s, psth_start_s + psth_num_bins * psth_bin_width_s, psth_bin_width_s), 3
            )
            dest_nwb.add_scratch(
                psth_timebins_s, name=f"timebins_{name}", description="Time bins in units of seconds for PSTH data"
            )

            # Add units table
            add_units_table(source_nwb, dest_nwb, is_normalizer)

            # Add trials table
            add_trials_table(source_nwb, dest_nwb, is_normalizer)

            # Add unique stimulus meta info (mworks ID and filename; hash info used in mwk_rsvp.py, but in here as video hash is not supported in MWorks 0.13)
            if not is_normalizer and "stimulus_meta" not in dest_nwb.stimulus.keys():
                session_stim_events_df = source_nwb.trials.to_dataframe()
                stimuli_presentation_id = session_stim_events_df["stimulus_presented"]
                stimuli_filename = session_stim_events_df["stimulus_filename"]
                # check if the dataframe contains column name for stimulus hash
                if "image_hash" in session_stim_events_df.columns:
                    image_hash = session_stim_events_df["image_hash"]
                    unique_pairs_set = set(zip(stimuli_presentation_id, stimuli_filename, image_hash))
                    unique_pairs_list = list(unique_pairs_set)
                    unique_pairs_sorted = sorted(unique_pairs_list, key=lambda x: x[0])
                    stimuli_id_sorted = [pair[0] for pair in unique_pairs_sorted]
                    stimuli_filename_sorted = [pair[1] for pair in unique_pairs_sorted]
                    stimuli_hash_sorted = [pair[2] for pair in unique_pairs_sorted]

                    # make dataframe for sorted features
                    stimuli_df = pd.DataFrame(
                        {
                            "stimuli_presentation_id": stimuli_id_sorted,
                            "stimuli_filename": stimuli_filename_sorted,
                            "stimuli_hash": stimuli_hash_sorted,
                        }
                    )
                else:
                    unique_pairs_set = set(zip(stimuli_presentation_id, stimuli_filename))
                    unique_pairs_list = list(unique_pairs_set)
                    unique_pairs_sorted = sorted(unique_pairs_list, key=lambda x: x[0])
                    stimuli_id_sorted = [pair[0] for pair in unique_pairs_sorted]
                    stimuli_filename_sorted = [pair[1] for pair in unique_pairs_sorted]

                    # make dataframe for sorted features
                    stimuli_df = pd.DataFrame(
                        {
                            "stimuli_presentation_id": stimuli_id_sorted,
                            "stimuli_filename": stimuli_filename_sorted,
                        }
                    )
                # make dynamic table
                stim_meta_table = pynwb.misc.DynamicTable(
                    name="stimulus_meta", description="Stimulus labels and metadata"
                )
                for column in stimuli_df.columns:
                    # new_table.add_column(name=column, description=column, data=stimuli_df[column].values)
                    stim_meta_table.add_column(name=column, description="")

                for row in stimuli_df.iterrows():
                    row_dict = row[1].to_dict()
                    stim_meta_table.add_row(**row_dict)

                dest_nwb.add_stimulus(stim_meta_table)

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
    pipeline_version: str = "EVPP_0.9",
    is_unit_valid: Optional[np.ndarray] = None,
    qc_dataframes: List[pd.DataFrame] = None,
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

    # Final pass: concatenate the session PSTHs, reading them back one block at a time
    if verbose:
        tqdm.write("Concatenating session PSTHs...")
    with NWBHDF5IO(aggregated_nwbfile_path, mode="a") as io:
        nwbfile = io.read()

        # The datasets, not their contents: the iterator reads each block as it is written
        session_psth_datasets = [
            nwbfile.scratch[key].data
            for key in sorted(nwbfile.scratch.keys())
            if key.startswith("psth_session_data_") and not key.endswith("concatenated")
        ]

        # Add merged PSTH time bins info from first session PSTH
        session_key = next(key for key in nwbfile.scratch.keys() if key.startswith("timebins_psth_session_data_"))
        timebins_s = nwbfile.scratch[session_key].data[:]
        nwbfile.add_scratch(
            timebins_s,
            name="timebins_psth_session_data_concatenated",
            description="Time bins in units of seconds for concatenated PSTH",
        )

        # Filter units using is_unit_valid array
        unit_indices = None
        if is_unit_valid is not None:
            number_of_units = session_psth_datasets[0].shape[0]
            if number_of_units != is_unit_valid.shape[0]:
                raise ValueError(
                    f"Dimension mismatch in unit validation arrays:\n"
                    f"Number of units in PSTH: {number_of_units}\n"
                    f"Number of units in validity array: {is_unit_valid.shape[0]}\n"
                    f"These dimensions must match for proper unit-wise validation.\n"
                    "Please ensure both arrays contain the same number of units."
                )
            unit_indices = np.nonzero(np.asarray(is_unit_valid))[0]
            nwbfile.add_scratch(
                is_unit_valid.tolist(),
                name="sites_psth_session_data_concatenated",
                description="Boolean array indicating which sites are valid (visually-driven) from raw PSTH",
            )

        concatenated_psth = ConcatenatedSessionPSTH(
            session_psth_datasets=session_psth_datasets, unit_indices=unit_indices
        )
        nwbfile.add_scratch(
            ScratchData(
                name="psth_session_data_concatenated",
                data=H5DataIO(
                    data=concatenated_psth,
                    chunks=choose_psth_chunk_shape(concatenated_psth.maxshape),
                    compression="gzip",
                    compression_opts=4,
                ),
                description="Concatenated PSTH from multiple files",
            )
        )

        # add QC dataframes (add_analysis)
        if qc_dataframes is not None:
            for i, qc_df in enumerate(qc_dataframes):
                # make dynamic table for each qc dataframe
                qc_table = pynwb.misc.DynamicTable(
                    name=f"normalizer_analysis_{i}",
                    description="quality control (EVPP 0.9) analysis from one normalizer recording",
                )
                for column in qc_df.columns:
                    # new_table.add_column(name=column, description=column, data=stimuli_df[column].values)
                    qc_table.add_column(name=column, description="")

                for row in qc_df.iterrows():
                    row_dict = row[1].to_dict()
                    qc_table.add_row(**row_dict)

                nwbfile.add_analysis(qc_table)

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
