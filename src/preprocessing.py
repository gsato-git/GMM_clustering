"""Data preprocessing utilities."""

import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler, RobustScaler


def remove_climatology(data, method='monthly'):
    """
    Remove climatology (seasonal cycle) from time series.
    
    Parameters
    ----------
    data : np.ndarray
        Input data with shape (time, space)
    method : {'monthly', 'annual'}, default='monthly'
        Climatology period
    
    Returns
    -------
    np.ndarray
        Anomalies (data - climatology)
    """
    if method == 'monthly':
        # For monthly data, compute 30-year climatology
        n_months = 12
        n_years = data.shape[0] // n_months
        
        # Reshape to (years, months, space)
        data_reshaped = data[:n_years*n_months].reshape(n_years, n_months, -1)
        
        # Compute climatology for each month
        climatology = np.nanmean(data_reshaped, axis=0)  # (months, space)
        
        # Compute anomalies
        anomalies = np.zeros_like(data[:n_years*n_months])
        for i in range(n_years):
            for j in range(n_months):
                anomalies[i*n_months + j] = data[i*n_months + j] - climatology[j]
        
        return anomalies
    
    elif method == 'annual':
        climatology = np.nanmean(data, axis=0, keepdims=True)
        return data - climatology
    
    else:
        raise ValueError(f"Unknown method: {method}")


def standardize(data, scaler_type='standard'):
    """
    Standardize data (zero mean, unit variance).
    
    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_samples, n_features)
    scaler_type : {'standard', 'robust'}, default='standard'
        Type of scaling
    
    Returns
    -------
    np.ndarray
        Standardized data
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    return scaler.fit_transform(data)


def detrend_data(data, detrend_type='linear'):
    """
    Remove trend from data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_samples, n_features)
    detrend_type : {'linear', 'polynomial'}, default='linear'
        Type of detrending
    
    Returns
    -------
    np.ndarray
        Detrended data
    """
    if detrend_type == 'linear':
        detrended = signal.detrend(data, axis=0, type='linear')
    elif detrend_type == 'polynomial':
        detrended = signal.detrend(data, axis=0, type='polynomial')
    else:
        raise ValueError(f"Unknown detrend type: {detrend_type}")
    
    return detrended


def handle_missing_values(data, method='forward_fill'):
    """
    Handle missing values in data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    method : {'forward_fill', 'interpolate', 'drop'}, default='forward_fill'
        Handling method
    
    Returns
    -------
    np.ndarray
        Data with missing values handled
    """
    if method == 'forward_fill':
        # Forward fill along time axis
        for i in range(1, data.shape[0]):
            mask = np.isnan(data[i])
            data[i][mask] = data[i-1][mask]
    
    elif method == 'interpolate':
        # Linear interpolation along time axis
        for j in range(data.shape[1]):
            data[:, j] = np.interp(
                np.arange(data.shape[0]),
                np.arange(data.shape[0])[~np.isnan(data[:, j])],
                data[~np.isnan(data[:, j]), j]
            )
    
    elif method == 'drop':
        # Remove rows with any NaN
        data = data[~np.any(np.isnan(data), axis=1)]
    
    return data


def remove_outliers(data, method='iqr', threshold=1.5):
    """
    Remove or cap outliers in data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    method : {'iqr', 'zscore'}, default='iqr'
        Outlier detection method
    threshold : float, default=1.5
        Threshold for outlier detection
    
    Returns
    -------
    np.ndarray
        Data with outliers removed or capped
    """
    data_clean = data.copy()
    
    if method == 'iqr':
        Q1 = np.nanpercentile(data, 25, axis=0)
        Q3 = np.nanpercentile(data, 75, axis=0)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Cap outliers
        data_clean = np.clip(data, lower_bound, upper_bound)
    
    elif method == 'zscore':
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        z_scores = np.abs((data - mean) / std)
        data_clean[z_scores > threshold] = np.nan
    
    return data_clean
