from pathlib import Path
from typing import Union


def locate_intan_file_path(
    data_folder: Union[Path, str],
    image_set_name: str,
    subject: str,
    session_date: str,
    session_time: str,
) -> Path:
    """Locates the path to the Intan RHD file based on a specific data structure.

    This function assumes the following directory structure for storing Intan data:

    data_folder
    ├── exp_{image_set_name}
    │   ├── exp_{image_set_name}.sub_{subject}
    │   │   └── raw_files
    │   │       └── intanraw
    │   │           └── {subject}_{image_set_name}_{session_date[2:]}_{session_time}
    │   │               └── info.rhd  <-- Target file

    If the data structure changes, this function will need to be modified accordingly.

    Parameters
    ----------
    data_folder : Path
        The root directory where the experimental data is stored.
    image_set_name : str
        The name of the image set used in the experiment.
    subject : str
        The identifier for the subject in the experiment.
    session_date : str
        The date of the recording session (e.g., '2024-06-24').
    session_time : str
        The time of the recording session (e.g., '15-30-00').

    Returns
    -------
    Path
        The path to the Intan RHD file (`info.rhd`).

    Raises
    ------
    AssertionError
        If any of the expected directories or files are not found.
    """

    assert data_folder.is_dir(), f"Data directory not found: {data_folder}"

    experiment_folder = data_folder / f"exp_{image_set_name}"
    assert experiment_folder.is_dir(), f"Experiment folder not found: {experiment_folder}"

    subject_folder = experiment_folder / f"exp_{image_set_name}.sub_{subject}"
    assert subject_folder.is_dir(), f"Subject folder not found: {subject_folder}"

    raw_data_folder = subject_folder / "raw_files"
    assert raw_data_folder.is_dir(), f"Raw files folder not found: {raw_data_folder}"

    intan_session_folder = (
        raw_data_folder / "intanraw" / f"{subject}_{image_set_name}_{session_date[2:]}_{session_time}"
    )
    assert intan_session_folder.is_dir(), f"Intan session folder not found: {intan_session_folder}"

    intan_file_path = intan_session_folder / "info.rhd"
    assert intan_file_path.is_file(), f"Intan file not found: {intan_file_path}"

    return intan_file_path


def locate_mworks_processed_file_path(
    data_folder: Union[Path, str],
    image_set_name: str,
    subject: str,
    session_date: str,
    session_time: str,
) -> Path:
    """Locates the path to the Mworks processed CSV file based on a specific data structure.

    This function assumes the following directory structure for storing processed Mworks data:

    data_folder
    ├── exp_{image_set_name}
    │   ├── exp_{image_set_name}.sub_{subject}
    │   │   └── raw_files
    │   │       └── mworksproc
    │   │           └── {subject}_{image_set_name}_{session_date[2:]}_{session_time}_mwk.csv  <-- Target file

    If the data structure changes, this function will need to be modified accordingly.

    Parameters
    ----------
    data_folder : Path
        The root directory where the experimental data is stored.
    image_set_name : str
        The name of the image set used in the experiment.
    subject : str
        The identifier for the subject in the experiment.
    session_date : str
        The date of the recording session (e.g., '2024-06-24').
    session_time : str
        The time of the recording session (e.g., '15-30-00').

    Returns
    -------
    Path
        The path to the processed Mworks CSV file.

    Raises
    ------
    AssertionError
        If any of the expected directories or files are not found.
    """

    assert data_folder.is_dir(), f"Data directory not found: {data_folder}"

    experiment_folder = data_folder / f"exp_{image_set_name}"
    assert experiment_folder.is_dir(), f"Experiment folder not found: {experiment_folder}"

    subject_folder = experiment_folder / f"exp_{image_set_name}.sub_{subject}"
    assert subject_folder.is_dir(), f"Subject folder not found: {subject_folder}"

    raw_data_folder = subject_folder / "raw_files"
    assert raw_data_folder.is_dir(), f"Raw files folder not found: {raw_data_folder}"

    mworks_processed_folder = raw_data_folder / "mworksproc"
    assert mworks_processed_folder.is_dir(), f"mworksproc folder not found: {mworks_processed_folder}"

    session_id = f"{subject}_{image_set_name}_{session_date[2:]}_{session_time}"
    mworks_processed_file_path = mworks_processed_folder / f"{session_id}_mwk.csv"
    assert mworks_processed_file_path.is_file(), f"Mworks file not found: {mworks_processed_file_path}"

    return mworks_processed_file_path
