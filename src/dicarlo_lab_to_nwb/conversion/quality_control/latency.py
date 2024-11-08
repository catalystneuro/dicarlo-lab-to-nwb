from pynwb import NWBHDF5IO
from pathlib import Path
import numpy as np
from tqdm import tqdm


# estimate response latency from reliability of responses
# def get_unit_latencies_from_reliabilities(nwb_filepath: str | Path) -> list[float]:
def get_unit_latencies_from_reliabilities(psth: np.array, psth_timebins_s: np.array, search_window_s: np.array = [0.010, 0.150]) -> list[float]:
    n_units, n_stimuli, n_reps, n_timebins = psth.shape
    search_window_idx_range = [np.argmin(np.abs(psth_timebins_s - t)) for t in search_window_s]
    search_window_idx = np.arange(search_window_idx_range[0], search_window_idx_range[1])
    n_search_window_bins = len(search_window_idx)
    n_reps_half = (n_reps - 1) // 2 # last rep is often filled with NaNs
        
    latencies_s = np.full(n_units, np.nan)
    n_runs = n_reps
    psth_windowed = psth[..., search_window_idx]
    for i in tqdm(range(n_units), total=n_units, desc="Calculating unit latencies"):
        psth_windowed_i = psth_windowed[i, ...]
        rho_samples = np.full((n_runs, n_search_window_bins), np.nan)
        for r in np.arange(n_runs):
            # randomly select n_reps_half indices from range(n_reps)
            first_half_indices = np.sort(np.random.choice(n_reps, n_reps_half, replace=False))
            # select the other half of indices
            second_half_indices = np.sort(np.setdiff1d(np.arange(n_reps), first_half_indices))
            
            mean_psth_0 = np.nanmean(psth_windowed_i[:, first_half_indices, :], axis=1)
            mean_psth_1 = np.nanmean(psth_windowed_i[:, second_half_indices, :], axis=1)
            # make sure fancy indexing is doesn't mess up the shape
            assert mean_psth_0.shape[0] == n_stimuli, "Half-split PSTH shape mismatch"
            
            for t in np.arange(n_search_window_bins):
                rho_t_r = np.corrcoef(mean_psth_0[:, t], mean_psth_1[:, t])[0, 1]
                rho_samples[r, t] = rho_t_r

        rho_means = np.nanmean(rho_samples, axis=0)
        
        if np.isnan(rho_means).all():
            latencies_s[i] = np.nan
        else:
            psth_timebins_s_idx = np.argmax(rho_means) + search_window_idx_range[0]
            latencies_s[i] = psth_timebins_s[psth_timebins_s_idx]
                

    return latencies_s

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from tqdm import tqdm

    # Path to the NWB file
    nwb_folder_path = Path("/Users/yoon/raw_data/Apollo/Monkeyvalence/processed/")
    # List of all NWB files in the folder
    nwb_filepaths = [file for file in nwb_folder_path.iterdir() if file.suffix == '.nwb']

    # Get the latencies for all the NWB files
    latencies = get_unit_latencies_from_reliabilities(nwb_filepaths[0])
    
        