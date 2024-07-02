from pathlib import Path

import numpy as np
import pandas as pd
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.utils import DeepDict
from PIL import Image
from pynwb.base import ImageReferences
from pynwb.file import NWBFile
from pynwb.image import GrayscaleImage, Images, IndexSeries, RGBAImage, RGBImage

map_image_set_to_folder_name = {"domain-transfer-2023": "RSVP-domain_transfer"}


class StimuliImagesInterface(BaseDataInterface):
    """Stimuli interface for DiCarlo Lab data."""

    keywords = [""]

    def __init__(self, file_path: str | Path, folder_path: str | Path, image_set_name: str):
        # This should load the data lazily and prepare variables you need
        self.file_path = Path(file_path)
        experiment_folder = map_image_set_to_folder_name[image_set_name]
        self.image_set_name = image_set_name

        self.stimuli_folder = folder_path / experiment_folder
        assert self.stimuli_folder.is_dir(), f"Experiment stimuli folder not found: {self.stimuli_folder}"

        if self.image_set_name == "domain-transfer-2023":
            self.stimuli_folder = self.stimuli_folder / "images"
            assert self.stimuli_folder.is_dir(), f"Experiment stimuli folder not found: {self.stimuli_folder}"

    def get_metadata(self) -> DeepDict:
        # Automatically retrieve as much metadata as possible from the source files available
        metadata = super().get_metadata()

        return metadata

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):
        dtype = {"stimulus_presented": np.uint32, "fixation_correct": bool}
        mwkorks_df = pd.read_csv(self.file_path, dtype=dtype)

        ground_truth_time_column = "samp_on_us"
        image_presentation_time = mwkorks_df[ground_truth_time_column] / 1e6
        stimulus_ids = mwkorks_df["stimulus_presented"]

        unique_stimulus_ids = stimulus_ids.unique()
        unique_stimulus_ids.sort()

        image_list = []
        for stimulus_id in unique_stimulus_ids:
            image_name = f"im{stimulus_id}"
            image_file_path = self.stimuli_folder / f"{image_name}.png"
            assert image_file_path.is_file(), f"Stimulus image not found: {image_file_path}"
            image = Image.open(image_file_path)
            image_array = np.array(image)
            # image_array = np.rot90(image_array, k=3)
            image_kwargs = dict(name=image_name, data=image_array, description="stimuli_image")
            if image_array.ndim == 2:
                image = GrayscaleImage(**image_kwargs)
            elif image_array.ndim == 3:
                if image_array.shape[2] == 3:
                    image = RGBImage(**image_kwargs)
                elif image_array.shape[2] == 4:
                    image = RGBAImage(**image_kwargs)
                else:
                    raise ValueError(f"Image array has unexpected number of channels: {image_array.shape[2]}")
            else:
                raise ValueError(f"Image array has unexpected dimensions: {image_array.ndim}")

            image_list.append(image)

        images_container = Images(
            name="stimuli",
            images=image_list,
            description=f"{self.image_set_name}",
            order_of_images=ImageReferences("order_of_images", image_list),
        )

        stimulus_id_to_index = {stimulus_id: index for index, stimulus_id in enumerate(unique_stimulus_ids)}
        data = np.array([stimulus_id_to_index[stimulus_id] for stimulus_id in stimulus_ids])
        index_series = IndexSeries(
            name="stimulus_presentation",
            data=data,
            indexed_images=images_container,
            unit="N/A",
            timestamps=image_presentation_time.values,
            description="Stimulus presentation index",
        )

        nwbfile.add_stimulus(images_container)
        nwbfile.add_stimulus(index_series)
