import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hdmf.data_utils import AbstractDataChunkIterator, DataChunk
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


class SingleImageIterator(AbstractDataChunkIterator):
    """Simple iterator to return a single image. This avoids loading the entire image into memory at initializing
    and instead loads it at writing time one by one"""

    def __init__(self, filename):
        self._filename = Path(filename)

        with Image.open(self._filename) as img:
            self.image_mode = img.mode
            self._image_shape = img.size[::-1]  # PIL uses (width, height) instead of (height, width)
            self._max_shape = (None, None)

            self.number_of_bands = len(img.getbands())
            if self.number_of_bands > 1:
                self._image_shape += (self.number_of_bands,)
                self._max_shape += (self.number_of_bands,)

        self._images_returned = 0  # Number of images returned in __next__

    def __iter__(self):
        """Return the iterator object"""
        return self

    def __next__(self):
        """
        Return the DataChunk with the single full image
        """
        if self._images_returned == 0:

            data = np.asarray(Image.open(self._filename))
            selection = (slice(None),) * data.ndim
            self._images_returned += 1
            return DataChunk(data=data, selection=selection)
        else:
            raise StopIteration

    def recommended_chunk_shape(self):
        """
        Recommend the chunk shape for the data array.
        """
        return self._image_shape

    def recommended_data_shape(self):
        """
        Recommend the initial shape for the data array.
        """
        return self._image_shape

    @property
    def dtype(self):
        """
        Define the data type of the array
        """
        return np.dtype(float)

    @property
    def maxshape(self):
        """
        Property describing the maximum shape of the data array that is being iterated over
        """
        return self._max_shape


class StimuliImagesInterface(BaseDataInterface):
    """Stimuli interface for DiCarlo Lab data."""

    keywords = [""]

    def __init__(
        self,
        file_path: str | Path,
        folder_path: str | Path,
        image_set_name: str,
        verbose: bool = False,
        train_test_split_data_file_path: Optional[str | Path] = None,
    ):
        # parsed mworks events to make sure unique images matches the stimuli set in folder_path
        self.file_path = Path(file_path)
        # This should load the data lazily and prepare variables you need
        self.image_set_name = image_set_name

        self.stimuli_folder = Path(folder_path)
        assert self.stimuli_folder.is_dir(), f"Experiment stimuli folder not found: {self.stimuli_folder}"
        self.train_test_split_data_file_path = (
            Path(train_test_split_data_file_path) if train_test_split_data_file_path else None
        )

        self.verbose = verbose

    def get_metadata(self) -> DeepDict:
        # Automatically retrieve as much metadata as possible from the source files available
        metadata = super().get_metadata()

        return metadata

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict,
        stub_test: bool = False,
    ):

        image_list = list(self.stimuli_folder.iterdir())
        if stub_test:
            image_list = image_list[:10]
        image_mode_to_nwb_class = {"LA": GrayscaleImage, "L": GrayscaleImage, "RGB": RGBImage, "RGBA": RGBAImage}

        for image_file_path in tqdm(image_list, desc="Processing images", unit=" images", disable=not self.verbose):

            image_filename = image_file_path.name
            iter_data = SingleImageIterator(filename=image_file_path)
            image_class = image_mode_to_nwb_class[iter_data.image_mode]

            image_kwargs = dict(name=image_filename, data=iter_data, description=f"")
            image_object = image_class(**image_kwargs)

            image_list.append(image_object)

        images_container = Images(
            name="stimuli",
            images=image_list,
            description=f"{self.image_set_name}",
            # order_of_images=ImageReferences("order_of_images", image_list), Not being able to add this is a bug
        )

        nwbfile.add_stimulus_template(images_container)


class SessionStimuliImagesInterface(StimuliImagesInterface):
    """Stimuli interface for DiCarlo Lab data."""

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict,
        stub_test: bool = False,
        ground_truth_time_column: str = "samp_on_us",
    ):
        dtype = {"stimulus_presented": np.uint32, "fixation_correct": bool}
        mworks_df = pd.read_csv(self.file_path, dtype=dtype)

        if self.train_test_split_data_file_path is not None:
            train_test_split_df = pd.read_csv(self.train_test_split_data_file_path)
            # Create separate mappings for train status and filenames
            stimulus_id_to_is_train = dict(zip(train_test_split_df["stim_id"] - 1, train_test_split_df["is_train"]))
            mworks_df["is_train"] = mworks_df["stimulus_presented"].map(stimulus_id_to_is_train)

            if "filename" in train_test_split_df.columns:
                stimulus_id_to_filename = dict(zip(train_test_split_df["stim_id"] - 1, train_test_split_df["filename"]))
                mworks_df["stimulus_filename"] = mworks_df["stimulus_presented"].map(stimulus_id_to_filename)
            else:
                mworks_df["stimulus_filename"] = [f"{stim_id}.png" for stim_id in mworks_df["stimulus_presented"]]

            # Filter to training data only
            mworks_df = mworks_df.query("is_train == True")

        if stub_test:
            mworks_df = mworks_df.iloc[:10]

        columns = mworks_df.columns
        assert ground_truth_time_column in columns, f"Column {ground_truth_time_column} not found in {columns}"
        image_presentation_time_seconds = mworks_df[ground_truth_time_column] / 1e6
        stimulus_indices = mworks_df["stimulus_presented"].to_list()

        # If the mworks does not contain stimulus_filename add it with an heurisitc
        if "stimulus_filenames" in mworks_df.columns:
            stimulus_filenames = mworks_df["stimulus_filenames"].to_list()
        else:
            stimulus_filenames = [f"{stimulus_id}.png" for stimulus_id in stimulus_indices]

        stimulus_filenames = mworks_df["stimulus_filename"].to_list()

        # Generate unique lists while preserving order
        seen = set()
        uniq_stim_indices = []
        uniq_stim_filenames = []

        for stim_idx, stim_filename in zip(stimulus_indices, stimulus_filenames):
            if stim_idx not in seen:
                uniq_stim_indices.append(stim_idx)
                uniq_stim_filenames.append(stim_filename)
                seen.add(stim_idx)

        # Sort the paired lists based on stimulus indices
        paired_uniq_lists = sorted(zip(uniq_stim_indices, uniq_stim_filenames))
        uniq_stim_indices, uniq_stim_filenames = zip(*paired_uniq_lists)
        assert len(uniq_stim_indices) == len(uniq_stim_filenames), "Stimulus indices and filenames do not match"

        image_list = []
        image_mode_to_nwb_class = {"L": GrayscaleImage, "RGB": RGBImage, "RGBA": RGBAImage}

        for index, stim_filename in enumerate(
            tqdm(uniq_stim_filenames, desc="Processing images", unit=" images", disable=not self.verbose)
        ):
            image_filename = stim_filename
            image_file_path = self.stimuli_folder / f"{image_filename}"
            assert image_file_path.is_file(), f"Stimulus image not found: {image_file_path}"
            image = Image.open(image_file_path)
            image_array = np.array(image)
            # in case of 2 channels: grayscale + alpha
            if image_array.ndim == 3 and image_array.shape[2] < 3:
                image_array = image_array[..., 0]
            image_kwargs = dict(name=image_filename, data=image_array, description="")

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

        nwbfile.add_stimulus(images_container)

        indexed_images = nwbfile.stimulus["stimuli"]

        # Add the stimulus presentation index
        # Note that every time has to be pointed to the exact index
        stimulus_id_to_index = {stimulus_id: index for index, stimulus_id in enumerate(uniq_stim_indices)}
        data = np.asarray([stimulus_id_to_index[stimulus_id] for stimulus_id in stimulus_indices], dtype=np.uint32)
        index_series = IndexSeries(
            name="stimulus_presentation",
            data=data,
            indexed_images=indexed_images,
            unit="N/A",
            timestamps=image_presentation_time_seconds.values,
            description="Stimulus presentation index",
        )

        nwbfile.add_stimulus(index_series)


class StimuliVideoInterface(BaseDataInterface):

    def __init__(
        self,
        folder_path: str | Path,
        image_set_name: str,
        train_test_split_data_file_path: Optional[str | Path] = None,
        video_copy_path: str | Path = None,
        verbose: bool = False,
    ):
        # This should load the data lazily and prepare variables you need
        self.stimuli_folder = Path(folder_path)
        self.video_copy_path = Path(video_copy_path) if video_copy_path is not None else None
        if self.video_copy_path:
            self.video_copy_path.mkdir(parents=True, exist_ok=True)

        assert self.stimuli_folder.is_dir(), f"Experiment stimuli folder not found: {self.stimuli_folder}"
        self.image_set_name = image_set_name

        self.train_test_split_data_file_path = (
            Path(train_test_split_data_file_path) if train_test_split_data_file_path else None
        )

        self.verbose = verbose

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict | None = None,
        stub_test: bool = False,
        ground_truth_time_column: str = "samp_on_us",
    ):

        video_file_paths = list(self.stimuli_folder.iterdir())
        for video_file_path in tqdm(
            video_file_paths, desc="Processing videos", unit="videos", disable=not self.verbose
        ):

            image_series = ImageSeries(
                name="stimuli",
                description=f"{self.image_set_name}",
                unit="n.a.",
                external_file=[video_file_path],
                rate=1.0,  # Fake sampling rate
            )

            nwbfile.add_stimulus(image_series)

        if self.video_copy_path:
            # Copy all the videos in file_path_list to the video_copy_path
            for file_path in video_file_paths:
                video_output_file_path = self.video_copy_path / file_path.name
                shutil.copy(file_path, video_output_file_path)


class SessionStimuliVideoInterface(BaseDataInterface):

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

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict | None = None,
        stub_test: bool = False,
        ground_truth_time_column: str = "samp_on_us",
    ):

        dtype = {"stimulus_presented": np.uint32, "fixation_correct": bool}
        mworks_df = pd.read_csv(self.file_path, dtype=dtype)

        if self.train_test_split_data_file_path is not None:
            train_test_split_df = pd.read_csv(self.train_test_split_data_file_path)
            # Create separate mappings for train status and filenames
            stimulus_id_to_is_train = dict(zip(train_test_split_df["stim_id"], train_test_split_df["is_train"]))
            mworks_df["is_train"] = mworks_df["stimulus_presented"].map(stimulus_id_to_is_train)

            # If the filename is on the train test split csv use that one
            if "filename" in train_test_split_df.columns:
                stimulus_id_to_filename = dict(zip(train_test_split_df["stim_id"], train_test_split_df["filename"]))
                file_path_list = mworks_df["stimulus_presented"].map(stimulus_id_to_filename)
            else:
                file_path_list = [
                    self.stimuli_folder / f"{stimuli_number}.mp4" for stimuli_number in mworks_df["stimulus_presented"]
                ]

            # Filter to training data only
            mworks_df = mworks_df.query("is_train == True")

        if stub_test:
            mworks_df = mworks_df.iloc[:10]

        columns = mworks_df.columns
        assert ground_truth_time_column in columns, f"Column {ground_truth_time_column} not found in {columns}"
        image_presentation_time_seconds = mworks_df[ground_truth_time_column] / 1e6

        # If the mworks does not contain stimulus_filename add it with an heurisitc
        if "stimulus_filenames" in mworks_df.columns:
            stimulus_filenames = mworks_df["stimulus_filenames"].to_list()
        else:
            stimulus_filenames = [
                self.stimuli_folder / f"{stimuli_number}.mp4" for stimuli_number in mworks_df["stimulus_presented"]
            ]

        file_path_list = mworks_df["stimulus_filename"].to_list()

        missing_file_path = [file_path for file_path in stimulus_filenames if not file_path.is_file()]
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
