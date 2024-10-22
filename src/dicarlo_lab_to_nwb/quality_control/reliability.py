from pynwb import NWBHDF5IO, NWBFile
from pathlib import Path
import numpy as np
import scipy.stats as stats


def get_NRR(rates, n_resamples=100, n_samples: int=2, seed: float=0) -> np.ndarray:
    """
    calculate N-repeated reliability, in which the Pearson correlation between 
    two halves of the data is calculated from the mean of randomly drawing 
    n_samples of repetitions from the data. For example, if n_samples=2, 
    we get the so-called single-repeat reliability (SRR) that is not affected 
    by the number of repetitions. If n_samples=n_reps, we get the half-split 
    reliability (HSR) that is sensitive to the number of repetitions.

    Parameters
    ----------
    rates : num_units x num_stimuli x num_repeats 
    num_permutations : (int) resampling, optional
    n_samples : (int) number of samples to draw from repetitions. Used to get the two halves of the data
    
    Returns
    -------
    list of reliability values for each unit
    """

    n_units, n_stimuli, n_reps = rates.shape
    rho_resampled = np.full((n_units, n_resamples), np.nan)
    rand_state = np.random.RandomState(seed)
    for u in range(n_units):
        rates_u = rates[u]
        for i in range(n_resamples):
            
            # n-repeats
            repeats_i = rand_state.choice(n_reps, n_samples, replace=False)
            rep_0 = np.nanmean(rates_u[:, repeats_i[:repeats_i.size//2]], axis=1)
            rep_1 = np.nanmean(rates_u[:, repeats_i[repeats_i.size//2:]], axis=1)

            # filter out NaN from both reps
            mask = np.logical_and(~np.isnan(rep_0), ~np.isnan(rep_1))
            if np.sum(mask) > 1: # needs at least 2 data points
                # rho_resampled[u, i] = stats.pearsonr(rep_0[mask], rep_1[mask])[0]
                rho_i = stats.pearsonr(rep_0[mask], rep_1[mask])[0]

                 # Spearman-Brown correction for splitting into halves
                n_splits = 2
                rho_i_corrected = (n_splits * rho_i) / (1 + (n_splits-1)*rho_i)
                rho_resampled[u, i] = rho_i_corrected
            
    return rho_resampled



def get_p_values(rates, num_permutations: int=100) -> np.ndarray:
    """
    calculate P-values according to stimulus (signal) driven variances vs null distribution of rates

    Parameters
    ----------
    rates : num_units x num_stimuli x num_repeats 
    num_permutations : (int) resampling, optional

    
    Returns
    -------
    list of p-values for each unit
    """
    p_list = []
    n_units, n_stimuli, n_reps = rates.shape
    for u in range(n_units):
        rates_u = rates[u]

        # null vs test distributions of signal variances
        sv_null_dist = []
        sv_test_dist = []

        for _ in range(num_permutations):
            # random resampling of rates with replacement in reps
            rates_u_resampled = rates_u[:, np.random.choice(n_reps, n_reps, replace=True)]
            flattened_data = rates_u.flatten()
            permuted_data = np.random.permutation(flattened_data)
            rates_u_null = permuted_data.reshape(rates_u.shape)

            # adjusted variance of the mean (subtracting sampliing error)
            # Sampling error of the null distribution accounts for n_stimuli & n_reps as we shuffled both reps and stimuli
            signal_var_null = np.var(np.nanmean(rates_u_null, axis=-1), ddof=1) - np.nanvar(rates_u_null)/(n_reps*n_stimuli)
            sv_null_dist.append(signal_var_null)

            # adjusted variance of the mean (subtracting sampliing error)
            signal_var_test = np.var(np.nanmean(rates_u_resampled, axis=-1), ddof=1) - np.nanvar(rates_u_resampled)/(n_reps)
            sv_test_dist.append(signal_var_test)

        sv_null_dist = np.array(sv_null_dist)
        sv_test_dist = np.array(sv_test_dist)

        # Calculate the p-value
        p_value = np.mean(sv_test_dist <= sv_null_dist)
        p_list.append(p_value)
        
    return np.array(p_list)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from tqdm import tqdm

    # Path to the NWB file
    # nwb_folder_path = Path("/Users/yoon/raw_data/Apollo/Monkeyvalence/processed/")
    nwb_folder_path = Path("/Users/yoon/raw_data/Apollo/test/processed/")

    # List of all NWB files in the folder
    nwb_filepaths = [file for file in nwb_folder_path.iterdir() if file.suffix == '.nwb']

    mode = 'a' # append to existing NWB file    
    with NWBHDF5IO(nwb_filepaths[0], mode=mode) as io:
        nwbfile = io.read()
        
        psth = nwbfile.acquisition["PSTH"].data[:]
        n_units, n_stimuli, n_reps, n_timebins = psth.shape
        latencies_s = nwbfile.scratch['latency'][:]

        psth_timebins_s = nwbfile.acquisition["PSTH"].timestamps[:]
        assert n_timebins == len(psth_timebins_s), "Time bins do not match"

        trial_df = nwbfile.intervals["trials"].to_dataframe()
        stim_duration_s = trial_df["stimuli_presentation_time_ms"].unique()[0] / 1e3
        stim_size_deg = trial_df["stimulus_size_degrees"].unique()[0]

        # integrate spike counts over stimulus presentation duration + response latency
        psth_stim_onset_s = 0
        response_latencies_s = latencies_s
        rates = np.full((n_units, n_stimuli, n_reps), np.nan)
        for i in range(n_units):
            intg_window_s_0 = psth_stim_onset_s + 0.080 #response_latencies_s[i]
            intg_window_s_1 = intg_window_s_0 + stim_duration_s
            intg_window = [intg_window_s_0, intg_window_s_1]
            intg_window_size_s = np.diff(intg_window)[0]
            intg_window_idx = [np.argmin(np.abs(psth_timebins_s - t)) for t in intg_window]

            psth_unit = psth[i]
            rates[i,:,:] = np.sum(psth_unit[..., intg_window_idx[0]:intg_window_idx[1]], axis=-1) / intg_window_size_s
        
        print(f"rates shape: {rates.shape}")
        
        # p values
        p_values = get_p_values(rates)
        valid_units = p_values < 0.05
        print(f"with p < 0.05: N={np.sum(valid_units)} out of {len(p_values)} units")

        # reliabilities
        n_samples = 10
        rhos_samples = get_NRR(rates, n_resamples=200, n_samples=n_samples)
        rhos_mean = np.nanmean(rhos_samples, axis=1)
        print(f"half-split reliability (above 0.5) : {np.sum(rhos_mean>0.5)}")
        srr = get_NRR(rates, n_resamples=200, n_samples=2)
        print(f"single-repeat reliability: {srr}")