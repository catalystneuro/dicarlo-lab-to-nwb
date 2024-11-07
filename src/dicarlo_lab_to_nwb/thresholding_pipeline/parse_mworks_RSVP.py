from pathlib import Path
import os
import sys
from dicarlo_lab_to_nwb.mworks_library import mwk_rsvp, mwk_rsvp_new, mwk_bars
from spikeinterface.extractors import IntanRecordingExtractor
import time


def parse_mworks_file(mworks_folder: str | Path, raw_data_folder: str | Path, output_folder: str | Path):
    mworks_folder = Path(mworks_folder)
    assert mworks_folder.is_dir(), f"Folder {mworks_folder} does not exist"
    mworks_filepath = list(mworks_folder.glob('*.mwk2'))[0]

    print(f"... processing .mwk2 file at: {mworks_filepath}...\n")

    time_start = time.time()

    intan_file_path = Path(raw_data_folder) / "info.rhd"
    assert intan_file_path.is_file()

    recording = IntanRecordingExtractor(
        file_path=intan_file_path,
        stream_name="RHD2000 amplifier channel",
    )
    sampling_freq = recording.get_sampling_frequency()
    photodiode_file = raw_data_folder / 'board-ANALOG-IN-1.dat'
    digi_event_file = raw_data_folder / 'board-DIGITAL-IN-02.dat'
        
    # run parser
    if "normalizer" in mworks_filepath.name: 
       output_filepath = mwk_rsvp.dump_events_rsvp(sampling_freq, mworks_filepath, photodiode_file, digi_event_file, output_folder)
    elif "bar_mapping" in mworks_filepath.name: 
        print(f"parsing bar mapping from: {mworks_filepath.name}")
        output_filepath = mwk_bars.dump_events_bars(sampling_freq, mworks_filepath, photodiode_file, digi_event_file, output_folder)
    else:
        output_filepath = mwk_rsvp_new.dump_events_rsvp(sampling_freq, mworks_filepath, photodiode_file, digi_event_file, output_folder)

    time_stop = time.time()
    time_taken = time_stop - time_start
    print(f"... extracted MWorks behavioral data in {time_taken} seconds")

    return output_filepath

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: python parse_mworks.py <mworks_filepath> <raw_data_folder> <output_folder>")
        for arg in sys.argv:
            print(arg)
        sys.exit(1)

    mworks_filepath = sys.argv[1]
    raw_data_folder = sys.argv[2]
    output_folder = sys.argv[3]

    parse_mworks_file(mworks_filepath, raw_data_folder, output_folder)
