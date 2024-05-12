"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter
from neuroconv.datainterfaces import (
    IntanRecordingInterface
)

from dicarlo_lab_to_nwb.conversion import ConversionBehaviorInterface


class ConversionNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        Recording=IntanRecordingInterface,
    )
