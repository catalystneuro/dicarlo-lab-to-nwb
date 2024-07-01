from pathlib import Path

import pandas as pd
from probeinterface import Probe, ProbeGroup
from spikeinterface.extractors import IntanRecordingExtractor


def add_geometry_to_probe_data_frame(probe_info_df: pd.DataFrame) -> pd.DataFrame:

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

    digital_channel_in_probe = probe_info_df["Intan"].values.tolist()
    channel_ids = recording.get_channel_ids().tolist()

    channel_indices = []
    for channel_in_probe in digital_channel_in_probe:
        channel_index = channel_ids.index(channel_in_probe)
        channel_indices.append(channel_index)

    probe_info_df["digital_channel_index"] = channel_indices

    return probe_info_df


def build_probe_group(
    recording: IntanRecordingExtractor,
) -> ProbeGroup:

    # Get the probe info stored in the repo
    probe_info_path = Path(__file__).parent / "probe_info.csv"
    assert probe_info_path.is_file(), f"Probe info CSV not found: {probe_info_path}"
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

        probe_group.add_probe(probe)

    return probe_group
