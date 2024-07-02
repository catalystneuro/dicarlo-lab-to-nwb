"""Primary NWBConverter class for this dataset."""

from neuroconv import NWBConverter
from neuroconv.datainterfaces import IntanRecordingInterface

from .behaviorinterface import BehavioralTrialsInterface
from .stimuli_interface import StimuliImagesInterface


class ConversionNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        Recording=IntanRecordingInterface,
        Behavior=BehavioralTrialsInterface,
        Stimuli=StimuliImagesInterface,
    )
