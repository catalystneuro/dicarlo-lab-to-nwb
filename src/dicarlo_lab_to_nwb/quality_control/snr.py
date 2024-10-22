from ephys.data_structure.convert_h5_to_nwb import load_psth_from_nwb
from pathlib import Path
import numpy as np
import scipy.stats as stats



# Stimulus signal-to-noise ratio (SNR)
def compute_snr(rates):
    n_units, n_stimuli, n_reps = rates.shape
    
    signal_var_pop = np.var(np.nanmean(rates, axis=-1), axis=1)
    noise_var_pop = np.mean(np.nanvar(rates, axis=-1), axis=1)
    snr = signal_var_pop / noise_var_pop
    return snr

# Stimulus signal-to-noise ratio (SNR) w/ resampling
def compute_snr_resampled(rates, n_resamples=200) -> np.ndarray:
    n_units, n_stimuli, n_reps = rates.shape
    
    snr_samples = np.full((n_units, n_resamples), np.nan)
    for i in range(n_resamples):
        # random resampling of rates with replacement in reps
        rates_resampled = rates[:, :, np.random.choice(n_reps, n_reps, replace=True)]
        snr_samples[:,i] = compute_snr(rates_resampled)
    return snr_samples

# NULL distribution of stimulus signal-to-noise ratio (SNR)
def compute_null_snr_resampled(rates, n_resamples=200) -> np.ndarray:
    n_units, n_stimuli, n_reps = rates.shape
    
    snr_null_samples = np.full((n_units, n_resamples), np.nan)
    for i in range(n_resamples):
        # random resampling of rates with replacement in reps
        rates_u_resampled = rates[:, np.random.choice(n_reps, n_reps, replace=True)]
        flattened_data = rates_u_resampled.flatten()
        permuted_data = np.random.permutation(flattened_data)
        # print(f"flattened_data shape: {flattened_data.shape}")
        rates_null = permuted_data.reshape(rates_u_resampled.shape)
        
        snr_null_samples[:,i] = compute_snr(rates_null)
    return snr_null_samples


if __name__ == "__main__":

    home_folder = Path("/Users/yoon/Dropbox (MIT)/dorsal_ventral/SFM_project")
    datafolder_root = home_folder / 'conversion_nwb'

    # filepath to NWB file
    nwb_path = datafolder_root / 'oleo_SFM_foveal_images_LIP.nwb'

    # load psth from nwb files
    psth, psth_t_ms = load_psth_from_nwb(nwb_path)

    # get stimulus duration
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(nwb_path, mode="r") as io:
        nwbfile = io.read()
        meta = nwbfile.scratch['psth meta'][:]

    # integrate spike counts over stimulus presentation duration + response latency
    stim_duration_ms = meta[-1] # 200.0
    stim_onset_ms = meta[-2] # 0.0
    response_latency_ms = 90.0
    intg_window = [stim_onset_ms + response_latency_ms, stim_onset_ms + response_latency_ms + stim_duration_ms]
    intg_window_size_ms = np.diff(intg_window)[0]
    intg_window_idx = [np.argmin(np.abs(psth_t_ms - t)) for t in intg_window]
    print(f"integration window index: {intg_window_idx}")
    print(f"integration window time (ms): {intg_window_size_ms}")
    rates = np.sum(psth[:, :, :, intg_window_idx[0]:intg_window_idx[1]], axis=3) / (intg_window_size_ms/1000)
    
    # SNR
    snr_resampled = compute_snr_resampled(rates)
    print(f"SNR resampled: {snr_resampled.shape}")
    print(f"SNR mean: {np.mean(snr_resampled, axis=1)}")