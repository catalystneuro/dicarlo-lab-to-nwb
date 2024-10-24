from pathlib import Path
from typing import Optional

import pandas as pd
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools.spikeinterface import add_electrodes_to_nwbfile
from probeinterface import Probe, ProbeGroup
from pynwb.file import NWBFile
from spikeinterface.core import generate_recording
from spikeinterface.extractors import IntanRecordingExtractor


class UtahArrayProbeInterface(BaseDataInterface):

    def __init__(self, file_path: Optional[str | Path] = None):

        probe_info_path = file_path
        if probe_info_path is None:
            probe_info_path = _fetch_default_probe_info_path()
        self.probe_info_path = probe_info_path

        super().__init__()

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: Optional[dict] = None):

        probe_info_df = pd.read_csv(filepath_or_buffer=self.probe_info_path)
        channel_ids = probe_info_df["Intan"].values
        channel_ids.sort()
        channel_ids = channel_ids.tolist()
        number_of_channels = len(channel_ids)
        dummy_recording = generate_recording(durations=[0], num_channels=number_of_channels)
        dummy_recording = dummy_recording.rename_channels(new_channel_ids=channel_ids)

        attach_probe_to_recording(recording=dummy_recording, probe_info_path=self.probe_info_path)

        contact_vector = dummy_recording.get_property("contact_vector")
        brain_area = contact_vector["brain_area"].astype("str")
        dummy_recording.set_property(key="brain_area", values=brain_area)
        add_electrodes_to_nwbfile(nwbfile=nwbfile, recording=dummy_recording, metadata=metadata)


def add_geometry_to_probe_data_frame(probe_info_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds geometric coordinates to a DataFrame containing probe information.

    The function calculates relative x and y positions for probes based on their column, row, and connector
    information. The geometric arrangement is determined by the specifications of the Utah array and the
    configuration used during the surgery.

    Parameters
    ----------
    probe_info_df : pd.DataFrame
        DataFrame containing probe information with columns 'col', 'row', and 'Connector'.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'rel_x' and 'rel_y' columns representing the relative x and y positions of the probes.

    Notes
    -----
    - The geometry and the 400 micrometers (um) distance between each probe in a column is based on the Utah array specifications.
    - The surgery configuration assumed a distance of 2000 um between each probe.
    - All distance units are in micrometers (um).
    - The probes are arranged in a pattern such that the first probe connected to probes A, B and C are at the bottom,
    followed by subsequent probes at calculated y-offsets.

    Example
    -------
    >>> data = {
    >>>     'col': [0, 1, 2, 0, 1, 2, 0, 1],
    >>>     'row': [0, 0, 0, 1, 1, 1, 2, 2],
    >>>     'Connector': ['A', 'A', 'A', 'D', 'D', 'D', 'G', 'G']
    >>> }
    >>> probe_info_df = pd.DataFrame(data)
    >>> add_geometry_to_probe_data_frame(probe_info_df)
       col  row Connector  rel_x   rel_y
    0    0    0         A    0.0     0.0
    1    1    0         A  400.0     0.0
    2    2    0         A  800.0     0.0
    3    0    1         D    0.0  2400.0
    4    1    1         D  400.0  2400.0
    5    2    1         D  800.0  2400.0
    6    0    2         G    0.0  4800.0
    7    1    2         G  400.0  4800.0
    """
    # This is from the datasheet of the Utah array
    distance_um: float = 400.0

    # Approximation. The surgery was performed such that probes are 2000 um apart
    number_of_columns = 10
    probe_span_um = distance_um * number_of_columns
    distance_between_probes_um = 2000.0

    offset_between_probes_um = distance_between_probes_um + probe_span_um

    # This considers that the first prob is the one in A,B,C at the bottom, then D,E,F and finally G,H at the top
    port_to_y_offset = {
        "A": 0,
        "B": 0,
        "C": 0,
        "D": offset_between_probes_um,
        "E": offset_between_probes_um,
        "F": offset_between_probes_um,
        "G": offset_between_probes_um * 2,
        "H": offset_between_probes_um * 2,
    }

    # We add the relative x and y positions
    probe_info_df["rel_x"] = probe_info_df["col"] * distance_um
    probe_info_df["rel_y"] = [
        row * distance_um + port_to_y_offset[connector]
        for row, connector in zip(probe_info_df["row"], probe_info_df["Connector"])
    ]

    return probe_info_df


def add_intan_wiring_to_probe_data_frame(
    recording: IntanRecordingExtractor,
    probe_info_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds digital channel indices from an Intan recording to a DataFrame containing probe information.

    This function matches the digital channels listed in the probe information DataFrame with their corresponding
    channel ids in the Intan recording to find its indices and then adds them to the DataFrame.

    These indices are used to make the mapping between the digital channels in the acquisition system
    and the electrodes in the probe

    Parameters
    ----------
    recording : IntanRecordingExtractor
        An instance of IntanRecordingExtractor containing the recording data and channel information.
    probe_info_df : pd.DataFrame
        DataFrame containing probe information with a column 'Intan' listing the digital channels.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added 'digital_channel_index' column representing the index of each digital channel in the recording.
    """
    digital_channel_in_probe = probe_info_df["Intan"].values.tolist()
    channel_ids = recording.get_channel_ids().tolist()

    channel_indices = []
    for channel_in_probe in digital_channel_in_probe:
        channel_index = channel_ids.index(channel_in_probe)
        channel_indices.append(channel_index)

    probe_info_df["digital_channel_index"] = channel_indices

    return probe_info_df


def build_probe_group(recording: IntanRecordingExtractor, probe_info_path: str | Path | None = None) -> ProbeGroup:
    """
    Builds a ProbeGroup object from an Intan recording by processing probe information and geometry.

    These function follows the following workflow:
    1) Read the probe information from a CSV file.
    2) Filter the probe information to match the ports available in the recording.
    3) Add geometric coordinates to the probe information taking into account the geometry of the  Utah array.
    4) Builds a mapping between the digital channels in the acquisition system and the electrodes in the probe.

    With that information for each probe (which is specified in the )

    This function reads the probe information from a CSV file, filters it to match the ports available in the recording,
    adds geometric coordinates and digital channel indices, and constructs a ProbeGroup with the processed information.

    Parameters
    ----------
    recording : IntanRecordingExtractor
        An instance of IntanRecordingExtractor containing the recording data and channel information.
    probe_info_path : str | Path, optional
        Path to the CSV file containing the probe information. If not provided, the default path is used.
        See the _fetch_default_probe_info_path() function for more information.

    Returns
    -------
    ProbeGroup
        A ProbeGroup object containing the probes with their respective geometric and digital channel information.
    """

    if probe_info_path is None:
        probe_info_path = _fetch_default_probe_info_path()

    probe_info_df = pd.read_csv(filepath_or_buffer=probe_info_path)

    # Filter to the ports available in the recording
    channel_ids = recording.get_channel_ids()
    channel_port = [channel_id.split("-")[0] for channel_id in channel_ids]
    available_ports = set(channel_port)
    probe_info_df = probe_info_df[probe_info_df["Connector"].isin(available_ports)]

    probe_info_df = add_geometry_to_probe_data_frame(probe_info_df=probe_info_df)
    probe_info_df = add_intan_wiring_to_probe_data_frame(recording=recording, probe_info_df=probe_info_df)

    probe_group = ProbeGroup()
    probe_group_names = probe_info_df.probe_group.unique()
    for probe_name in probe_group_names:
        probe_group_df = probe_info_df[probe_info_df.probe_group == probe_name]

        probe = Probe(ndim=2, si_units="um")
        contact_ids = [f"{id.replace('elec', '')}" for id in probe_group_df["label"].values]

        positions = probe_group_df[["rel_x", "rel_y"]].values
        size_of_tip_on_micrometer = 50  # Estimated size of the tip of the probe in micrometers
        probe.set_contacts(
            positions=positions,
            contact_ids=contact_ids,
            shape_params={"radius": size_of_tip_on_micrometer},
        )

        # We leave a margin of 400 um between probes and the border
        margin_um = 400
        probe.create_auto_shape(probe_type="rect", margin=margin_um)
        probe.set_device_channel_indices(probe_group_df["digital_channel_index"].values)

        brain_area = probe_group_df["brain_area"].values
        probe.annotate_contacts(brain_area=brain_area)

        probe_group.add_probe(probe)

    return probe_group


def attach_probe_to_recording(recording: IntanRecordingExtractor, probe_info_path: str | Path | None = None):
    """
    Builds a ProbeGroup and then attaches to an Intan recording and sets properties for group and probe names.

    This function builds a ProbeGroup from the recording, sets the ProbeGroup in the recording, and assigns
    group and probe names based on the digital channels and ports in the recording.

    Parameters
    ----------
    recording : IntanRecordingExtractor
        An instance of IntanRecordingExtractor containing the recording data and channel information.
    probe_info_path : str | Path, optional
        Path to the CSV file containing the probe information. If not provided, the default path is used.
        See the _fetch_default_probe_info_path() function for more information.
    """

    if probe_info_path is None:
        probe_info_path = _fetch_default_probe_info_path()

    probe_group = build_probe_group(recording=recording, probe_info_path=probe_info_path)
    recording.set_probegroup(probe_group, group_mode="by_probe", in_place=True)

    group_numbers = recording.get_property("group")
    number_to_group_name = {0: "ABC", 1: "DEF", 2: "GH"}  # We name the probe groups by the ports in the Intan system
    group_names = [number_to_group_name[int(group_number)] for group_number in group_numbers]
    recording.set_property("group_name", group_names)

    number_to_probe_name = {0: "ProbeABC", 1: "ProbeDEF", 2: "ProbeGH"}  #
    probe_names = [number_to_probe_name[int(group_name)] for group_name in group_numbers]
    recording.set_property("probe", probe_names)

    # This is redundant but adding here to shield this pipeline from changes in the spikeinterface and neo API
    available_properties = recording.get_property_keys()
    if "channel_names" not in available_properties:
        channel_names = recording.get_channel_ids()
        recording.set_property("channel_names", channel_names)


def _fetch_default_probe_info_path() -> Path:
    """
    Fetches the default path for the probe information CSV file.

    This function constructs the default path to the probe information CSV file.
    This data is specially formatted for the series of experiments conducted in the DiCarlo lab that use
    the two or three Utah arrays connected to the Intan acquisition system.

    The format of the file is as follows:

    | Intan | Connector |  | Array | Bank | elec | label | col | row | probe_group | brain_area |
    |-------|-----------|--|-------|------|------|-------|-----|-----|-------------|------------|
    | C-008 | C         |31| 1     | C    | 31   | elec1 | 9   | 1   | ABC         | V1         |
    | A-007 | A         |32| 2     | A    | 32   | elec2 | 9   | 2   | ABC         | V1         |
    | A-006 | A         |30| 3     | A    | 30   | elec3 | 9   | 3   | ABC         | V1         |
    | A-005 | A         |28| 4     | A    | 28   | elec4 | 9   | 4   | ABC         | V1         |
    | A-004 | A         |26| 5     | A    | 26   | elec5 | 9   | 5   | ABC         | V1         |
    | A-003 | A         |24| 6     | A    | 24   | elec6 | 9   | 6   | ABC         | V1         |
    | A-002 | A         |22| 7     | A    | 22   | elec7 | 9   | 7   | ABC         | V1         |
    | B-008 | B         |18| 9     | B    | 18   | elec9 | 8   | 1   | ABC         | V1         |
    | B-007 | B         |16| 10    | B    | 16   | elec10| 8   | 2   | ABC         | V1         |
    | B-001 | B         |4 | 16    | B    | 4    | elec16| 8   | 8   | ABC         | V1         |
    | C-007 | C         |2 | 17    | C    | 2    | elec17| 7   | 1   | ABC         | V1         |
    | C-006 | C         |0 | 18    | C    | 0    | elec18| 7   | 2   | ABC         | V1         |
    | C-005 | C         |31| 19    | C    | 31   | elec19| 7   | 3   | ABC         | V1         |

    By default the data is packed together with the dicarlo-lab-to-nwb repository and should
    not require any additional setup.

    Returns
    -------
    Path
        The default path to the probe information CSV file.
    """

    probe_info_path = Path(__file__).parent / "probe_info.csv"
    assert probe_info_path.is_file(), f"Probe info CSV not found: {probe_info_path}"

    return probe_info_path
