import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.datainterfaces.behavior.video.video_utils import VideoCaptureContext
from neuroconv.utils import DeepDict
from PIL import Image
from pynwb.base import ImageReferences
from pynwb.file import NWBFile
from pynwb.image import (
    GrayscaleImage,
    Images,
    ImageSeries,
    IndexSeries,
    RGBAImage,
    RGBImage,
)
from tqdm.auto import tqdm


class StimuliImagesInterface(BaseDataInterface):
    """Stimuli interface for DiCarlo Lab data."""

    keywords = [""]

    def __init__(self, mworks_file_path: str | Path, folder_path: str | Path, image_set_name: str, verbose: bool = False):
        # This should load the data lazily and prepare variables you need
        self.file_path = Path(mworks_file_path)
        self.image_set_name = image_set_name
        self.stimuli_folder = Path(folder_path)
        assert self.stimuli_folder.is_dir(), f"Experiment stimuli folder not found: {self.stimuli_folder}"

        self.verbose = verbose

    def get_metadata(self) -> DeepDict:
        # Automatically retrieve as much metadata as possible from the source files available
        metadata = super().get_metadata()

        return metadata

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):
        dtype = {"stimulus_presented": np.uint32, "fixation_correct": bool}
        mwkorks_df = pd.read_csv(self.file_path, dtype=dtype)
        ground_truth_time_column = "samp_on_us"
        image_presentation_time_seconds = mwkorks_df[ground_truth_time_column] / 1e6
        stimulus_indices = mwkorks_df["stimulus_presented"].to_list()
        stimulus_filenames = mwkorks_df["stimulus_filename"].to_list()
        stimulus_hashes = mwkorks_df["image_hash"].to_list()

        # stimulus size and durations are constant for all images
        stimulus_on_s = np.unique(mwkorks_df["stim_on_time_ms"]) / 1e3
        stimulus_off_s = np.unique(mwkorks_df["stim_off_time_ms"]) / 1e3
        stimulus_size_deg = np.unique(mwkorks_df["stimulus_size_degrees"])
        # Generate unique lists from the above example while preserving the order of the unique values from the first list (stimulus_indices)
        seen = set() # Initialize an empty set to track seen values from list1

        # Initialize empty lists to store the unique values
        uniq_stim_indices = []
        uniq_stim_filenames = []
        uniq_stim_hashes = []

        # Iterate through the lists and preserve unique values based on list1
        for l1, l2, l3 in zip(stimulus_indices, stimulus_filenames, stimulus_hashes):
            if l1 not in seen:
                uniq_stim_indices.append(l1)
                uniq_stim_filenames.append(l2)
                uniq_stim_hashes.append(l3)
                seen.add(l1)

        # Sort the paired lists based on the first list (stimulus_indices)
        paired_uniq_lists = list(zip(uniq_stim_indices, uniq_stim_filenames, uniq_stim_hashes))
        paired_uniq_lists.sort()
        uniq_stim_indices, uniq_stim_filenames, uniq_stim_hashes = zip(*paired_uniq_lists)

        # labels_dtype = {"filename": str, "hash": str}
        # stimulus_df = pd.read_csv(self.image_labels_file_path, dtype=labels_dtype)
        # uniq_stim_filenames = stimulus_df["filename"]
        # uniq_stim_hashes = stimulus_df["hash"]

        assert len(uniq_stim_indices) == len(uniq_stim_filenames), "Stimulus indices and filenames do not match"

        image_list = []

        for idx, stim_filename in enumerate(tqdm(
            uniq_stim_filenames, desc="Processing images", unit=" images", disable=not self.verbose
        )):
            image_filename = stim_filename
            image_hash = uniq_stim_hashes[idx]
            image_file_path = self.stimuli_folder / f"{image_filename}"
            assert image_file_path.is_file(), f"Stimulus image not found: {image_file_path}"
            image = Image.open(image_file_path)
            image_array = np.array(image)
            # in case of 2 channels: grayscale + alpha
            if image_array.ndim == 3 and image_array.shape[2] < 3:
                image_array = image_array[..., 0]
            # image_array = np.rot90(image_array, k=3)
            image_kwargs = dict(name=image_filename, data=image_array, description=image_hash)
            
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

        # stimulus index from the mworks file
        stimulus_id_to_index = {stimulus_id: index for index, stimulus_id in enumerate(uniq_stim_indices)}
        data = np.asarray([stimulus_id_to_index[stimulus_id] for stimulus_id in stimulus_indices], dtype=np.uint32)
        index_series = IndexSeries(
            name="stimulus_presentation_index",
            data=data,
            indexed_images=images_container,
            unit="N/A",
            timestamps=image_presentation_time_seconds.values,
            description="Stimulus presentation index",
        )

        nwbfile.add_stimulus(images_container)
        nwbfile.add_stimulus(index_series)


class StimuliVideoInterface(BaseDataInterface):

    def __init__(
        self,
        file_path: str | Path,
        folder_path: str | Path,
        image_set_name: str,
        video_copy_path: str | Path = None,
        verbose: bool = False,
    ):
        # This should load the data lazily and prepare variables you need
        self.file_path = Path(file_path)
        self.stimuli_folder = Path(folder_path)
        self.video_copy_path = Path(video_copy_path) if video_copy_path is not None else None
        if self.video_copy_path:
            self.video_copy_path.mkdir(parents=True, exist_ok=True)

        assert self.stimuli_folder.is_dir(), f"Experiment stimuli folder not found: {self.stimuli_folder}"
        self.image_set_name = image_set_name

        self.verbose = verbose

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):
        dtype = {"stimulus_presented": np.uint32, "fixation_correct": bool}
        mwkorks_df = pd.read_csv(self.file_path, dtype=dtype)

        ground_truth_time_column = "samp_on_us"
        image_presentation_time_seconds = mwkorks_df[ground_truth_time_column].values / 1e6

        stimuli_presented = mwkorks_df.stimulus_presented.values

        file_path_list = [self.stimuli_folder / f"{stimuli_number + 1}.mp4" for stimuli_number in stimuli_presented]
        missing_file_path = [file_path for file_path in file_path_list if not file_path.is_file()]
        assert len(missing_file_path) == 0, f"Missing files: {missing_file_path}"

        timestamps_per_video = []
        number_of_frames = []
        sampling_rates = []

        for file_path in tqdm(file_path_list, desc="Processing videos", unit="videos_processed"):
            with VideoCaptureContext(file_path) as video_context:
                timestamps_per_video.append(video_context.get_video_timestamps(display_progress=False))
                number_of_frames.append(video_context.get_video_frame_count())
                sampling_rates.append(video_context.get_video_fps())

        # Shift the video timestamps by their presentation time
        corrected_timestamps = np.concatenate(
            [ts + image_presentation_time_seconds[i] for i, ts in enumerate(timestamps_per_video)]
        )
        starting_frame = np.zeros_like(number_of_frames)
        starting_frame[1:] = np.cumsum(number_of_frames)[:-1]

        image_series = ImageSeries(
            name="stimuli",
            description=f"{self.image_set_name}",
            unit="n.a.",
            external_file=file_path_list,
            timestamps=corrected_timestamps,
            starting_frame=starting_frame,
        )

        nwbfile.add_stimulus(image_series)

        if self.video_copy_path:
            # Copy all the videos in file_path_list to the video_copy_path
            for file_path in file_path_list:
                video_output_file_path = self.video_copy_path / file_path.name
                shutil.copy(file_path, video_output_file_path)
