"""Primary NWBConverter class for this dataset."""

from neuroconv import NWBConverter
from neuroconv.datainterfaces import IntanRecordingInterface

from .conversionbehaviorinterface import ConversionBehaviorInterface
from .stimuli_interface import StimuliInterface


class ConversionNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        Recording=IntanRecordingInterface,
        Behavior=ConversionBehaviorInterface,
        Stimuli=StimuliInterface,
    )
