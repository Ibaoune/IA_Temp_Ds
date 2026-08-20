import xarray as xr
import numpy as np

def scaling_delta_mapping(gcm_full, gcm_hist, obs_hist):
    """
    Signal-Preserving Scaling Delta Mapping (SDM), inspired by
    Baño-Medina et al. (2022).

    Parameters
    ----------
    gcm_full : xarray.DataArray
        Full period to correct.
        Example historical: LMDZ 1980-1985.
        Example future: LMDZ 2041-2050.

    gcm_hist : xarray.DataArray
        Historical/reference period of the same GCM/RCM.
        Example: LMDZ 1980-1984.

    obs_hist : xarray.DataArray
        Observational/reanalysis reference period.
        Example: ERA5 1980-1984.

    Logic
    -----
    1. Signal = monthly_mean(gcm_full) - monthly_mean(gcm_hist)
    2. Detrended = gcm_full - Signal
    3. Bias correction using mean/std of gcm_hist and obs_hist
    4. Re-add Signal

    This function is the same for historical and future periods.
    The difference comes only from the config.
    """
    
    # Group by month for seasonal cycle
    gcm_hist_grouped = gcm_hist.groupby("time.month")
    obs_hist_grouped = obs_hist.groupby("time.month")
    gcm_full_grouped = gcm_full.groupby("time.month")
    
    # 1. Calculate Monthly stats
    # Use .compute() to evaluate them and avoid repeatedly scanning the 22GB datasets!
    mean_gcm_hist = gcm_hist_grouped.mean(dim="time").compute()
    std_gcm_hist = gcm_hist_grouped.std(dim="time").compute()
    
    mean_obs_hist = obs_hist_grouped.mean(dim="time").compute()
    std_obs_hist = obs_hist_grouped.std(dim="time").compute()
    
    mean_gcm_full = gcm_full_grouped.mean(dim="time").compute()
    
    # 2. Apply Signal-Preserving Correction
    def _apply_sdm_signal_preserved(group):
        month = group.time.dt.month[0].values
        
        # Monthly parameters
        mu_gh = mean_gcm_hist.sel(month=month)
        sig_gh = std_gcm_hist.sel(month=month)
        mu_oh = mean_obs_hist.sel(month=month)
        sig_oh = std_obs_hist.sel(month=month)
        mu_gf = mean_gcm_full.sel(month=month)
        
        # Calculate Signal (Trend)
        signal = mu_gf - mu_gh
        
        # Detrend anomalies
        detrended = group - signal
        
        # Bias Correct anomalies to Observation level
        corrected_anomalies = (detrended - mu_gh) / (sig_gh + 1e-8) * sig_oh + mu_oh
        
        # Add Signal back
        final = corrected_anomalies + signal
        return final

    gcm_corrected = gcm_full.groupby("time.month").map(_apply_sdm_signal_preserved)
    
    return gcm_corrected

def standardize_predictors(ds, mean_ref, std_ref):
    """
    Standardize predictors based on a reference mean and std.
    """
    return (ds - mean_ref) / (std_ref + 1e-8)
