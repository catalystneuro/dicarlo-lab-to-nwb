from pynwb import NWBHDF5IO, NWBFile
from pathlib import Path
import numpy as np
import scipy.stats as stats
from dicarlo_lab_to_nwb.conversion.quality_control.latency import get_unit_latencies_from_reliabilities


def get_NRR(rates, n_resamples=100, n_reps: int=2, correction: bool=True, seed: float=0) -> np.ndarray:
    """
    calculate N-repeated reliability, in which the Pearson correlation between 
    two halves of the data is calculated from the mean of randomly drawing 
    n_reps of repetitions from the data. For example, if n_samples=2, 
    we get the so-called single-repeat reliability (SRR) that is not affected 
    by the number of repetitions. If n_reps=total repeats, we get the half-split 
    reliability, but this metric is sensitive to the number of repetitions.

    Parameters
    ----------
    rates : num_units x num_stimuli x num_repeats 
    num_permutations : (int) resampling, optional
    n_samples : (int) number of samples to draw from repetitions. Used to get the two halves of the data
    
    Returns
    -------
    list of reliability values for each unit
    """
    
    n_units, n_stimuli_0, n_reps = rates.shape
    rho_resampled = np.full((n_units, n_resamples), np.nan)
    rand_state = np.random.RandomState(seed)
    for u in range(n_units):
        rates_u = rates[u]
        # remove repetitions with NaN
        nan_reps = np.isnan(rates_u).any(axis=0)
        rates_u = rates_u[:,~nan_reps]
        n_stimuli, n_reps_u = rates_u.shape
        assert n_stimuli_0 == n_stimuli, "number of stimuli should be the same across units"
        if n_reps_u < n_reps:
            n_reps = n_reps_u
        for i in range(n_resamples):
            
            # n-repeats
            repeats_i = rand_state.choice(n_reps_u, n_reps, replace=False)
            rep_0 = np.nanmean(rates_u[:, repeats_i[:repeats_i.size//2]], axis=1)
            rep_1 = np.nanmean(rates_u[:, repeats_i[repeats_i.size//2:]], axis=1)

            # filter out NaN from both reps
            mask = np.logical_and(~np.isnan(rep_0), ~np.isnan(rep_1))
            if np.sum(mask) > 1: # needs at least 2 data points
                # rho_resampled[u, i] = stats.pearsonr(rep_0[mask], rep_1[mask])[0]
                rho_i = stats.pearsonr(rep_0[mask], rep_1[mask])[0]
                
                # Spearman-Brown correction for splitting into halves
                if correction:
                    n_splits = 2
                    rho_i_corrected = (n_splits * rho_i) / (1 + (n_splits-1)*rho_i)
                    rho_resampled[u, i] = rho_i_corrected
                else:
                    rho_resampled[u, i] = rho_i
            
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
    import pandas as pd


    # Path to the NWB file
    nwb_folder_path = Path("/Users/yoon/Downloads/nwb_files/20240924")

    # List of all NWB files in the folder
    nwb_filepaths = [file for file in nwb_folder_path.iterdir() if file.suffix == '.nwb']
    nwb_filepaths.sort()

    mode = 'r' # append to existing NWB file  
    for nwb_filepath in nwb_filepaths:  
        with NWBHDF5IO(nwb_filepath, mode=mode) as io:
            nwbfile = io.read()
            
            psth = nwbfile.scratch["psth_pipeline_format"].data[:]
            n_units, n_stimuli, n_reps, n_timebins = psth.shape
            df = nwbfile.electrodes.to_dataframe()
            channel_names = df["channel_name"].values

            binned_spikes = nwbfile.processing['ecephys']['BinnedAlignedSpikesToStimulus']
            psth_timebin_ms = binned_spikes.bin_width_in_milliseconds
            psth_0 = binned_spikes.milliseconds_from_event_to_first_bin 
            psth_1 = psth_0 + psth_timebin_ms * n_timebins - psth_timebin_ms
            psth_timebins_s = np.linspace(psth_0, psth_1, n_timebins) / 1e3
            latencies_s = get_unit_latencies_from_reliabilities(psth, psth_timebins_s)
            
            trial_df = nwbfile.intervals["trials"].to_dataframe()
            stim_duration_s = trial_df["stimuli_presentation_time_ms"].unique()[0] / 1e3
            stim_size_deg = trial_df["stimulus_size_degrees"].unique()[0]

            # integrate spike counts over stimulus presentation duration + response latency
            psth_stim_onset_s = 0
            rates = np.full((n_units, n_stimuli, n_reps), np.nan)
            mean_rates = np.full(n_units, np.nan)
            for i in range(n_units):
                intg_window_s_0 = psth_stim_onset_s + latencies_s[i]
                intg_window_s_1 = intg_window_s_0 + stim_duration_s
                intg_window = [intg_window_s_0, intg_window_s_1]
                intg_window_size_s = np.diff(intg_window)[0]
                intg_window_idx = [np.argmin(np.abs(psth_timebins_s - t)) for t in intg_window]

                psth_unit = psth[i]
                rates[i,:,:] = np.sum(psth_unit[..., intg_window_idx[0]:intg_window_idx[1]], axis=-1) / intg_window_size_s
                # average rates across all dimensions except the first one
                mean_rates[i] = np.nanmean(rates[i])
            
            # p values
            p_values = get_p_values(rates)
            p_thresh = 0.05
            valid_units = p_values < p_thresh
            print(f"with p < {p_thresh}: N={np.sum(valid_units)} out of {len(p_values)} units")
            print(f"mean latencies (valid units): {np.nanmean(latencies_s[valid_units])}")

            # reliabilities
            n_samples = n_reps // 2
            rhos_samples = get_NRR(rates, n_reps=n_samples)
            rhos_mean_values = np.nanmean(rhos_samples, axis=1)
            print(f"half-split reliability (above 0.5) : {np.sum(rhos_mean_values>0.5)}")
            srr_samples = get_NRR(rates, n_reps=2, correction=False)
            srr_mean_values = np.nanmean(srr_samples, axis=-1)
            print(f"SRR mean (valid units): {np.mean(srr_mean_values[valid_units])}")
            # save results to a dataframe
            df = pd.DataFrame({
                "channel_name": channel_names,
                "p_value": p_values,
                "valid_unit": valid_units,
                "mean_rate": mean_rates,
                "response_latency_ms": latencies_s,
                "half_split_reliability": rhos_mean_values,
                "single_repeat_reliability": srr_mean_values,
            })
            csv_filepath = nwb_folder_path / f"{nwbfile.session_id}_QC.csv"
            df.to_csv(csv_filepath, index=False)