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
