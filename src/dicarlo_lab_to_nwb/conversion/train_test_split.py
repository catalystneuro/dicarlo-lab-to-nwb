import pandas as pd

# Columns of the MWorks table that identify the stimulus shown in a presentation
STIMULUS_IDENTITY_COLUMNS = ["stimulus_filename", "image_hash", "video_hash"]


def redact_test_stimuli(mworks_df: pd.DataFrame, train_test_split_df: pd.DataFrame) -> pd.DataFrame:
    """
    Hide the identity of the test stimuli in a table of MWorks stimulus presentations.

    Presentations are matched to the train-test split by the name of the file that MWorks displayed
    (the `stimulus_filename` column of the parsed MWorks table against the `filename` column of the split),
    so the numbering convention of the stimulus set does not matter. For every test presentation,
    `stimulus_presented` is set to -1 and the columns that name the stimulus (`stimulus_filename`,
    `image_hash`, `video_hash`) are blanked. The presentation itself, including its time, is kept.

    Parameters
    ----------
    mworks_df : pd.DataFrame
        Parsed MWorks table, one row per stimulus presentation, with a `stimulus_filename` column.
    train_test_split_df : pd.DataFrame
        Train-test split of the stimulus set with a `filename` and an `is_train` column.

    Returns
    -------
    pd.DataFrame
        A copy of `mworks_df` with the test presentations redacted.
    """
    if "stimulus_filename" not in mworks_df.columns:
        raise ValueError("The MWorks table has no `stimulus_filename` column to match against the train-test split.")
    missing_split_columns = {"filename", "is_train"} - set(train_test_split_df.columns)
    if missing_split_columns:
        raise ValueError(f"The train-test split is missing the columns {sorted(missing_split_columns)}.")

    is_train_by_filename = dict(zip(train_test_split_df["filename"], train_test_split_df["is_train"].astype(bool)))
    presented_filenames = mworks_df["stimulus_filename"]
    unmatched_filenames = sorted(set(presented_filenames) - set(is_train_by_filename))
    if unmatched_filenames:
        raise ValueError(
            f"{len(unmatched_filenames)} presented stimuli are not in the train-test split, "
            f"for example {unmatched_filenames[:5]}."
        )
    is_test = ~presented_filenames.map(is_train_by_filename).to_numpy(dtype=bool)

    redacted_df = mworks_df.copy()
    redacted_df["stimulus_presented"] = redacted_df["stimulus_presented"].astype("int64")
    redacted_df.loc[is_test, "stimulus_presented"] = -1
    for column in STIMULUS_IDENTITY_COLUMNS:
        # A column MWorks left empty for this stimulus type (e.g. `image_hash` for videos) names nothing
        if column in redacted_df.columns and not redacted_df[column].isna().all():
            redacted_df.loc[is_test, column] = ""

    return redacted_df
