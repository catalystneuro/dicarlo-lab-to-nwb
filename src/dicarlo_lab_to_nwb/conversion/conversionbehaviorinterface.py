"""Primary class for converting experiment-specific behavior."""

from pathlib import Path

import numpy as np
import pandas as pd
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.utils import DeepDict
from pynwb.base import ImageReferences
from pynwb.file import NWBFile
from pynwb.image import GrayscaleImage, Images, IndexSeries, RGBImage


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
        # In this experiment the presentation of stimuli is batched as groups of at most 8 images.
        # Every presentation starts with a stim_on_time, then up to 8 images are presented
        # for stim_on time, then there is a stim_off_time before the next presentation.
        # The extract time of the presentation is samp_on_us or photo_diode_on_us.
        # We will make every image presentation a trial.

        dtype = {"stimulus_presented": np.uint32, "fixation_correct": bool}
        mwkorks_df = pd.read_csv(self.file_path, dtype=dtype)

        ground_truth_time_column = "samp_on_us"
        image_presentation_time = mwkorks_df[ground_truth_time_column] / 1e6
        stimulus_id = mwkorks_df["stimulus_presented"]

        unique_stimulus_ids = stimulus_id.unique()

        x = 1
