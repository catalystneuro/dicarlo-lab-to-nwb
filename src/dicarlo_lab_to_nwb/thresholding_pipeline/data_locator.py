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
        The date of the recording session (e.g., '20240624').
    session_time : str
        The time of the recording session (e.g., '153000').

    Returns
    -------
    Path
        The path to the processed Mworks CSV file.

    Raises
    ------
    AssertionError
        If any of the expected directories or files are not found.
    """

    has_short_date = {"domain-transfer-2023": True, "Co3D": False}

    assert data_folder.is_dir(), f"Data directory not found: {data_folder}"

    experiment_folder = data_folder / f"exp_{image_set_name}"
    assert experiment_folder.is_dir(), f"Experiment folder not found: {experiment_folder}"

    subject_folder = experiment_folder / f"exp_{image_set_name}.sub_{subject}"
    assert subject_folder.is_dir(), f"Subject folder not found: {subject_folder}"

    raw_data_folder = subject_folder / "raw_files"
    assert raw_data_folder.is_dir(), f"Raw files folder not found: {raw_data_folder}"

    mworks_processed_folder = raw_data_folder / "mworksproc"
    assert mworks_processed_folder.is_dir(), f"mworksproc folder not found: {mworks_processed_folder}"

    if has_short_date[image_set_name]:
        session_date_mworks = session_date[2:]
    else:
        session_date_mworks = session_date
    session_id = f"{subject}_{image_set_name}_{session_date_mworks}_{session_time}"
    mworks_processed_file_path = mworks_processed_folder / f"{session_id}_mwk.csv"
    assert mworks_processed_file_path.is_file(), f"Mworks file not found: {mworks_processed_file_path}"

    return mworks_processed_file_path


def locate_session_paths(
    data_folder: Union[Path, str],
    subject: str,
    project_name: str,
    session_date: str,
) -> list[Path]:
    """Locates the path to the Intan RHD file based on a specific data structure.
    This function assumes the following directory structure for storing Intan data:
    data_folder
    ├── {subject}
    │   ├── {project_name}
    │   │   └── {session_date}
    │   │       ├── {stimulus_set_0}_{session_date}_{session_time}
    │   │       |   └── info.rhd    <-- Target file for recording 0
    │   │       |   └── *.mwk2      <-- Stimulus event file for recording 0
    │   │       ├── {stimulus_set_1}_{session_date}_{session_time}
    │   │       |   └── info.rhd  <-- Target file
    │   │       |   └── *.mwk2      <-- Stimulus event file for recording 1
    │   │       └── {stimulus_set_2}_{session_date}_{session_time}
    │   │           └── info.rhd  <-- Target file
    │   │           └── *.mwk2      <-- Stimulus event file for recording 2
    If the data structure changes, this function will need to be modified accordingly.
    Parameters
    ----------
    data_folder : Path
        The root directory where ALL experimental data is stored.
    subject : str
        The identifier for the subject in the experiment (e.g., pico).
    project_name : str
        The name of the project, which typically includes 'normalizers' and the test stimuli (images or videos, e.g., CO3D or MURI1320).
    stimulus_set_name : str
        The name of the stimulus set used in each daily session.
        This method will iterate subfolders (stimulus sets) as there could be many (e.g., Normalizers, CO3D).
    session_date : str
        The date of recording a stimulus_set (within a recording session) in YYYYMMDD (e.g., '20240624').

    Returns
    -------
    List of Paths
        List of Paths according to each subfolder in the session_date folder.
    Raises
    ------
    AssertionError
        If any of the expected directories or files are not found.
    """
    data_folder = Path(data_folder)
    assert data_folder.is_dir(), f"Data directory not found: {data_folder}"

    subject_folder = data_folder / f"{subject}"
    assert subject_folder.is_dir(), f"Subject folder not found: {subject_folder}"

    project_folder = subject_folder / f"{project_name}"
    assert project_folder.is_dir(), f"Project folder not found: {project_folder}"

    todays_folder = project_folder / f"{session_date}"
    assert todays_folder.is_dir(), f"Today's folder not found: {todays_folder}"

    # empty list of folders
    session_folders = []
    for folder_i in todays_folder.iterdir():
        if folder_i.is_dir() and not folder_i.name.startswith(".") and not folder_i.name.startswith("processed"):
            intan_file_path = folder_i / "info.rhd"
            assert intan_file_path.is_file(), f"Intan file not found in: {folder_i}"
            print(f"Detected {folder_i.name} in {todays_folder}")
            session_folders.append(folder_i)

    return session_folders
