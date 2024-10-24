"""Primary class for converting experiment-specific behavior."""

from pathlib import Path

import numpy as np
import pandas as pd
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.utils import DeepDict
from pynwb.file import NWBFile


class BehavioralTrialsInterface(BaseDataInterface):
    """Behavior interface for conversion conversion"""

    keywords = ["behavior"]

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def get_metadata(self) -> DeepDict:
        # Automatically retrieve as much metadata as possible from the source files available
        metadata = super().get_metadata()

        return metadata

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):
        # In this experiment setup the presentation of stimuli is batched as groups of at most 8 images.
        # Every presentation starts with a stim_on_time, then up to 8 images are presented
        # for stim_on time, then there is a stim_off_time before the next presentation.
        # The extract time of the presentation is samp_on_us or photo_diode_on_us.
        # We will make every image presentation a trial.

        dtype = {"stimulus_presented": np.uint32, "fixation_correct": bool}
        mwkorks_df = pd.read_csv(self.file_path, dtype=dtype)

        ground_truth_time_column = "samp_on_us"
        mwkorks_df["start_time"] = mwkorks_df[ground_truth_time_column] / 1e6
        mwkorks_df["stimuli_presentation_time_ms"] = mwkorks_df["stim_on_time_ms"]
        mwkorks_df["inter_stimuli_interval_ms"] = mwkorks_df["stim_off_time_ms"]
        mwkorks_df["stop_time"] = mwkorks_df["start_time"] + mwkorks_df["stimuli_presentation_time_ms"] / 1e3

        mwkorks_df["stimuli_block_index"] = (
            mwkorks_df["stimulus_order_in_trial"]
            .diff()  # Differences (5 - 1)
            .lt(0)  # Gets the point where it goes back to 1
            .cumsum()
        )

        descriptions = {
            "stimuli_presentation_time_ms": "Duration of the stimulus presentation in milliseconds",
            "inter_stimuli_interval_ms": "Inter stimulus interval in milliseconds",
            "stimulus_presented": "The stimulus ID presented",
            "fixation_correct": "Whether the fixation was correct during this stimulus presentation",
            "stimuli_block_index": "The index of the block of stimuli presented",
        }

        # Add information of the following columns if they are present in the dataframe
        if "stimulus_size_degrees" in mwkorks_df.columns:
            descriptions["stimulus_size_degrees"] = "The size of the stimulus in degrees"

        if "fixation_window_size_degrees" in mwkorks_df.columns:
            descriptions["fixation_window_size_degrees"] = "The size of the fixation window in degrees"

        if "fixation_point_size_degrees" in mwkorks_df.columns:
            descriptions["fixation_point_size_degrees"] = "The size of the fixation point in degrees"

        for column_name, description in descriptions.items():
            nwbfile.add_trial_column(name=column_name, description=description)

        columns_to_write = ["start_time", "stop_time"] + list(descriptions.keys())

        # Extract a pandas dictionary with each row of the columns_to_write
        for _, row in mwkorks_df[columns_to_write].iterrows():
            nwbfile.add_trial(**row.to_dict())
