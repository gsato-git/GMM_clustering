"""Data loading utilities for climate datasets."""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path


def load_netcdf(filepath, variables=None):
    """
    Load data from NetCDF file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to NetCDF file
    variables : list, optional
        List of variable names to load. If None, loads all variables.
    
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Loaded data
    """
    ds = xr.open_dataset(filepath)
    if variables:
        return ds[variables]
    return ds


def load_sst_data(filepath, region=None, time_slice=None):
    """
    Load Sea Surface Temperature data.
    
    Parameters
    ----------
    filepath : str or Path
        Path to SST data file
    region : dict, optional
        Geographic region bounds {'lat_min', 'lat_max', 'lon_min', 'lon_max'}
    time_slice : dict, optional
        Time bounds {'start', 'end'}
    
    Returns
    -------
    xarray.DataArray
        SST data
    """
    ds = xr.open_dataset(filepath)
    
    # Select region
    if region:
        ds = ds.sel(
            latitude=slice(region['lat_min'], region['lat_max']),
            longitude=slice(region['lon_min'], region['lon_max'])
        )
    
    # Select time period
    if time_slice:
        ds = ds.sel(time=slice(time_slice['start'], time_slice['end']))
    
    # Get SST variable (handle various naming conventions)
    sst_names = ['sst', 'SST', 'sea_surface_temperature', 'analysed_sst']
    sst_var = None
    for name in sst_names:
        if name in ds.data_vars:
            sst_var = ds[name]
            break
    
    if sst_var is None:
        raise ValueError(f"SST variable not found. Available: {list(ds.data_vars)}")
    
    return sst_var


def load_precipitation_data(filepath, region=None, time_slice=None):
    """
    Load precipitation data.
    
    Parameters
    ----------
    filepath : str or Path
        Path to precipitation data file
    region : dict, optional
        Geographic region bounds
    time_slice : dict, optional
        Time bounds
    
    Returns
    -------
    xarray.DataArray
        Precipitation data
    """
    ds = xr.open_dataset(filepath)
    
    # Select region
    if region:
        ds = ds.sel(
            latitude=slice(region['lat_min'], region['lat_max']),
            longitude=slice(region['lon_min'], region['lon_max'])
        )
    
    # Select time period
    if time_slice:
        ds = ds.sel(time=slice(time_slice['start'], time_slice['end']))
    
    # Get precipitation variable
    precip_names = ['precip', 'precipitation', 'tp', 'pr', 'precips']
    precip_var = None
    for name in precip_names:
        if name in ds.data_vars:
            precip_var = ds[name]
            break
    
    if precip_var is None:
        raise ValueError(f"Precipitation variable not found. Available: {list(ds.data_vars)}")
    
    return precip_var


def reshape_for_analysis(data_array, reshape_to_2d=True):
    """
    Reshape spatiotemporal data for analysis.
    
    Parameters
    ----------
    data_array : xarray.DataArray
        Input data with dimensions (time, lat, lon)
    reshape_to_2d : bool, default=True
        If True, reshape to (time, space). If False, keep original shape.
    
    Returns
    -------
    np.ndarray
        Reshaped data
    """
    data = data_array.values
    
    if reshape_to_2d:
        # Reshape (time, lat, lon) to (time, space)
        time_dim = data.shape[0]
        space_dim = np.prod(data.shape[1:])
        data = data.reshape(time_dim, space_dim)
    
    return data


def save_processed_data(data, filepath, variable_name="data", coords=None):
    """
    Save processed data to file.
    
    Parameters
    ----------
    data : np.ndarray or xarray.DataArray
        Data to save
    filepath : str or Path
        Output file path
    variable_name : str, default="data"
        Variable name for NetCDF
    coords : dict, optional
        Coordinates for DataArray
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(data, np.ndarray):
        if filepath.suffix == '.nc':
            da = xr.DataArray(data, name=variable_name, coords=coords)
            da.to_netcdf(filepath)
        else:
            np.save(filepath, data)
    else:
        data.to_netcdf(filepath)
