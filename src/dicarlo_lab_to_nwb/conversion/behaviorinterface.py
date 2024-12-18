"""Primary class for converting experiment-specific behavior."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.utils import DeepDict
from pynwb.file import NWBFile


class BehavioralTrialsInterface(BaseDataInterface):
    """Behavior interface for conversion conversion"""

    keywords = ["behavior"]

    def __init__(
        self,
        file_path: str | Path,
        train_test_split_data_file_path: Optional[str | Path] = None,
    ):
        self.file_path = Path(file_path)
        self.train_test_split_data_file_path = (
            Path(train_test_split_data_file_path) if train_test_split_data_file_path else None
        )

    def get_metadata(self) -> DeepDict:
        # Automatically retrieve as much metadata as possible from the source files available
        metadata = super().get_metadata()

        return metadata

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict | None = None,
        stub_test: bool = False,
        ground_truth_time_column: str = "samp_on_us",
        is_stimuli_one_indexed: bool = False,
    ):
        # In this experiment setup the presentation of stimuli is batched as groups of at most 8 images.
        # Every presentation starts with a stim_on_time, then up to 8 images are presented
        # for stim_on time, then there is a stim_off_time before the next presentation.
        # The extract time of the presentation is samp_on_us or photo_diode_on_us.
        # We will make every image presentation a trial.

        # Note
        # In mworks the stimuli_indices are zero indexed.
        # On filenames and on the train-test-split csv the indices are 1-indexed for images and
        # 0 indexed for videos.

        dtype = {"stimulus_presented": np.uint32, "fixation_correct": bool}
        mworks_df = pd.read_csv(self.file_path, dtype=dtype)

        if self.train_test_split_data_file_path is not None:
            train_test_split_df = pd.read_csv(self.train_test_split_data_file_path)
            offset = 1 if is_stimuli_one_indexed else 0
            stimulus_index_for_test_data = -1
            offuscate = lambda row: row["stim_id"] - offset if row["is_train"] else stimulus_index_for_test_data
            offuscate_map = {row["stim_id"]: offuscate(row) for _, row in train_test_split_df.iterrows()}
            mworks_df["stimuli_index"] = mworks_df["stimulus_presented"].map(offuscate_map)

        if stub_test:
            mworks_df = mworks_df.iloc[:100]

        mworks_df["start_time"] = mworks_df[ground_truth_time_column] / 1e6
        mworks_df["stimuli_presentation_time_ms"] = mworks_df["stim_on_time_ms"]
        mworks_df["inter_stimuli_interval_ms"] = mworks_df["stim_off_time_ms"]
        mworks_df["stop_time"] = mworks_df["start_time"] + mworks_df["stimuli_presentation_time_ms"] / 1e3

        mworks_df["trial_index"] = (
            mworks_df["stimulus_order_in_trial"]
            .diff()  # Differences (5 - 1)
            .lt(0)  # Gets the point where it goes back to 1
            .cumsum()
        )

        descriptions = {
            "stimuli_presentation_time_ms": "Duration of the stimulus presentation in milliseconds",
            "inter_stimuli_interval_ms": "Inter stimulus interval in milliseconds",
            "stimulus_presented": "The stimulus ID presented",
            "fixation_correct": "Whether the fixation was correct during this stimulus presentation",
            "trial_index": "The index of the trial of stimuli presented",
        }

        # Add information of the following columns if they are present in the dataframe

        if "stimulus_order_in_trial":
            descriptions["stimulus_order_in_trial"] = "The order of the stimulus in the trial"

        if "stimulus_size_degrees" in mworks_df.columns:
            descriptions["stimulus_size_degrees"] = "The size of the stimulus in degrees"

        if "fixation_window_size_degrees" in mworks_df.columns:
            descriptions["fixation_window_size_degrees"] = "The size of the fixation window in degrees"

        if "fixation_point_size_degrees" in mworks_df.columns:
            descriptions["fixation_point_size_degrees"] = "The size of the fixation point in degrees"

        if "image_hash" in mworks_df.columns:
            descriptions["image_hash"] = "The hash of the image presented"

        if "video_hash" in mworks_df.columns:
            descriptions["video_hash"] = "The hash of the video presented"

        if "stimulus_filename" in mworks_df.columns:
            descriptions["stimulus_filename"] = "The name of the stimulus file"

        for column_name, description in descriptions.items():
            nwbfile.add_trial_column(name=column_name, description=description)

        columns_to_write = ["start_time", "stop_time"] + list(descriptions.keys())

        # Extract a pandas dictionary with each row of the columns_to_write
        for _, row in mworks_df[columns_to_write].iterrows():
            nwbfile.add_trial(**row.to_dict())
