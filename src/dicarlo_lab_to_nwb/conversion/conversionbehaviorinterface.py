"""Primary class for converting experiment-specific behavior."""

from pathlib import Path

import pandas as pd
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.utils import DeepDict
from pynwb.file import NWBFile


class ConversionBehaviorInterface(BaseDataInterface):
    """Behavior interface for conversion conversion"""

    keywords = ["behavior"]

    def __init__(self, file_path: str):
        # This should load the data lazily and prepare variables you need
        self.file_path = Path(file_path)

    def get_metadata(self) -> DeepDict:
        # Automatically retrieve as much metadata as possible from the source files available
        metadata = super().get_metadata()

        return metadata

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):
        # All the custom code to add the data the nwbfile

        dtype = {"stimulus_presented": int, "fixation_correct": int}
        mwkorks_df = pd.read_csv(self.file_path, dtype=dtype)

        mwkorks_df["stim_on_time"] = mwkorks_df["stim_on_time_ms"] / 1000.0
        mwkorks_df["stim_off_time"] = mwkorks_df["stim_off_time_ms"] / 1000.0

        ground_truth_time_column = "samp_on_us"

        mwkorks_df["start_time"] = mwkorks_df[ground_truth_time_column] / 1e6
        mwkorks_df["stop_time"] = mwkorks_df["start_time"] + mwkorks_df["stim_on_time"] + mwkorks_df["stim_off_time"]

        descriptions = {
            "stim_on_time": "Onset duration",
            "stim_off_time": "Inter stimulus interval",
            "stimulus_presented": "The stimulus ID presented",
            "fixation_correct": "Whether the fixation was correct during this stimulus presentation",
            "stimulus_size_degrees": "The size of the stimulus in degrees",
            "fixation_window_size_degrees": "The size of the fixation window in degrees",
            "fixation_point_size_degrees": "The size of the fixation point in degrees",
        }

        for column_name, description in descriptions.items():
            nwbfile.add_trial_column(name=column_name, description=description)

        columns_to_write = ["start_time", "stop_time"] + list(descriptions.keys())

        # Extract a pandas dictionary with each row of the columns_to_write
        for i, row in mwkorks_df[columns_to_write].iterrows():
            nwbfile.add_trial(**row.to_dict())
