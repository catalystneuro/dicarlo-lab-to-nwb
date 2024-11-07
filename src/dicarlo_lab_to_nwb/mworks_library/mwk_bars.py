import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from dicarlo_lab_to_nwb.mworks_library.mwk2reader import MWKFile
from pathlib import Path



def closest_value(input_list, input_value):
    arr = np.asarray(input_list)
    i = (np.abs(arr - input_value)).argmin()
    return arr[i]


def equal_for_all_trials(events):
    return all(e.data == events[0].data for e in events)


def listify_events(events):
    return list(e.data for e in events)


def get_events(event_file, name):
    data = {
        'code': [],
        'name': [],
        'time': [],
        'data': [],
    }
    for event in event_file.get_events_iter(codes=name):
        data['code'].append(event.code)
        data['name'].append(event_file.codec[event.code])
        data['time'].append(event.time)
        data['data'].append(event.data)
    data = pd.DataFrame(data)
    data = data.sort_values(by='time').reset_index(drop=True)
    return data


def dump_events_bars(SAMPLING_FREQUENCY_HZ, filename, photodiode_filepath, digi_event_filepath, output_dir: str = './'):
    print(f"Sampling rate: {SAMPLING_FREQUENCY_HZ}, {filename}")
    # # Expt start time
    # fid = open(expt_event_file, 'r')
    # filesize = os.path.getsize(filename)  # in bytes
    # num_samples = filesize // 2  # uint16 = 2 bytes
    # digital_in = np.fromfile(fid, 'uint16', num_samples)
    # fid.close()
    #
    # samp_on, = np.nonzero(digital_in[:-1] < digital_in[1:])  # Look for 0->1 transitions
    # samp_on = samp_on + 1  # Previous line returns indexes of 0s seen before spikes, but we want indexes of first spikes
    # pre_expt_start_ms = samp_on / SAMPLING_FREQUENCY_HZ * 1000

    # mworks
    event_file = MWKFile(filename)
    event_file.open()

    # Variables we'd like to fetch data for
    names = ['experiment_state_line',
             'trial_start_line',
             'locations_shown',
             'bars_per_trial',
             'stim_on_time',
             'stim_off_time',
             'stim_on_delay',
             'fixation_window_size',
             'location_x_list',
             'location_y_list',
             'stimulus_presented',
            ]
    data = get_events(event_file=event_file, name=names)
    event_file.close()

    # experiment start time
    expt_states = data[data.name == 'experiment_state_line']
    expt_start_time_ms = expt_states[expt_states.data == 1]['time'].values[0]/1000
    expt_stop_time_ms = expt_states[expt_states.data == 1]['time'].values[-1] / 1000

    # successful trial timestamps (stim on for 1200 ms before this timestamp)
    successes = data[data.name == 'locations_shown']
    trials_time_ms = successes[successes.data > 0]['time'].values / 1000

    # locations
    locations_x = data[data.name == 'location_x_list']['data'].values[-1]
    locations_y = data[data.name == 'location_y_list']['data'].values[-1]

    locations = list(zip(locations_x, locations_y))
    uniq_loc = sorted(set(locations))

    id_list = []
    for i, xy_tuple in enumerate(locations):
        id_list.append(uniq_loc.index(xy_tuple))

    ###########################################################################
    # Extract stimulus presentation order and fixation information
    ###########################################################################
    # stimulus_presented = data[data.name == 'stimulus_presented'].reset_index(drop=True)
    correct_fixation_df = data[data.name == 'correct_fixation'].reset_index(drop=True)
    # if len(stimulus_presented) > len(correct_fixation_df):
    #     locations_shown = stimulus_presented[:-1]
    # assert len(stimulus_presented) == len(correct_fixation_df)
    # Drop `empty` data (i.e. -1) before the experiment actually began and after it had already ended
    # correct_fixation_df = correct_fixation_df[stimulus_presented.data != -1].reset_index(drop=True)
    # stimulus_presented = stimulus_presented[stimulus_presented.data != -1].reset_index(drop=True)
    # Add `first_in_trial` info to other data frame too
    # correct_fixation_df['first_in_trial'] = stimulus_presented['first_in_trial']

    ###########################################################################
    # Read sample on file
    ###########################################################################
    fid = open(digi_event_filepath, 'r')
    filesize = os.path.getsize(filename)  # in bytes
    num_samples = filesize // 2  # uint16 = 2 bytes
    digital_in = np.fromfile(fid, 'uint16', num_samples)
    fid.close()

    samp_on, = np.nonzero(digital_in[:-1] < digital_in[1:])  # Look for 0->1 transitions
    samp_on = samp_on + 1  # Previous line returns indexes of 0s seen before spikes, but we want indexes of first spikes

    ###########################################################################
    # Read photodiode file
    ###########################################################################
    fid = open(photodiode_filepath, 'r')
    filesize = os.path.getsize(photodiode_filepath)  # in bytes
    num_samples = filesize // 2  # uint16 = 2 bytes
    v = np.fromfile(fid, 'uint16', num_samples)
    fid.close()

    # Convert to volts (use this if the data file was generated by Recording Controller)
    # v = (v - 32768) * 0.0003125
    v = v * 0.195
    
    upper_quantile = np.quantile(v, 0.75)
    lower_quantile = np.quantile(v, 0.25)
    v_range = upper_quantile - lower_quantile
    
    thresh = v_range * 0.5 + lower_quantile


    # Detect rises in the oscillating photodiode signal
    peaks, _ = find_peaks(v, height=0)  # Find all peaks
    # peaks = np.asarray([p for p in peaks if v[p] > THRESHOLD])  # Apply threshold
    peaks = np.asarray([p for p in peaks if v[p] > thresh])  # Apply threshold
    photodiode_on = np.asarray([min(peaks[(peaks >= s) & (peaks < (s + 100_000))]) for s in samp_on])

    assert len(photodiode_on) == len(samp_on)

    # Convert both times to microseconds to match MWorks
    photodiode_on = photodiode_on * 1_000 / SAMPLING_FREQUENCY_HZ  # in ms
    samp_on = samp_on * 1_000 / SAMPLING_FREQUENCY_HZ  # in ms
    print(f'Delay recorded on photodiode is {np.mean(photodiode_on - samp_on):.2f} ms on average')

    trial_start_ms = []
    for success_time in trials_time_ms:
        trial_start_ms.append(closest_value(photodiode_on, success_time - expt_start_time_ms - 1000))

    ###########################################################################
    # Create a dict to store output information
    ###########################################################################
    output = dict(stim_on_time_ms=data[data.name == 'stim_on_time']['data'].values[-1] / 1000.,
                  stim_off_time_ms=data[data.name == 'stim_off_time']['data'].values[-1] / 1000.,
                  stim_on_delay_ms=data[data.name == 'stim_on_delay']['data'].values[-1] / 1000.,
                  expt_start_time=expt_start_time_ms,
                  expt_stop_time=expt_stop_time_ms,
                  bars_per_trial=data[data.name == 'bars_per_trial']['data'].values[0])

    ###########################################################################
    # Save output
    ###########################################################################
    output['locations_x'] = locations_x
    output['locations_y'] = locations_y
    output['uniq_cond_id'] = id_list
    output['trial_start_ms'] = trial_start_ms

    # match standard outputs
    output['stimulus_presented'] = id_list
    output['fixation_correct'] = np.ones(len(trial_start_ms))
    output['samp_on_us'] = trial_start_ms
    output['stim_on_time_ms'] = np.ones(len(trial_start_ms)) * 1200
    
    # output['pupil_time_ms'] = pupil_time
    
    # save to processed folder
    output = pd.DataFrame(output)
    output_filepath = os.path.join(output_dir, str(filename).split('/')[-1][:-5] + '_mwk.csv')  # -5 in filename to delete the .mwk2 extension
    # output.to_csv(filename.split('/')[-1][:-5] + '_mwk.csv', index=False)  # -5 in filename to delete the .mwk2 extension
    output.to_csv(output_filepath, index=False) 

    # # save another to raw folder (for buildign psths)
    # data_folder = Path(photodiode_filepath).parent
    # output_filepath_raw = os.path.join(data_folder, str(filename).split('/')[-1][:-5] + '_mwk.csv')  # -5 in filename to delete the .mwk2 extension
    # output.to_csv(output_filepath_raw, index=False)

    # ###########################################################################
    # # Repetitions
    # ###########################################################################
    # selected_indexes = correct_fixation_df[correct_fixation_df.data == 1]['data'].index.tolist()
    # correct_trials = np.asarray(stimulus_presented_df.data.values.tolist())[selected_indexes]
    # num_repetitions = np.asarray([len(correct_trials[correct_trials == stimulus]) for stimulus in np.unique(stimulus_presented_df.data.values.tolist())])
    # print(f'{min(num_repetitions)} repeats, range is {np.unique(num_repetitions)}')

    return output_filepath

# if __name__ == '__main__':
#     dump_events(sys.argv[1], sys.argv[2], sys.argv[3])
