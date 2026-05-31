# Data Directory

## Data Structure

```
data/
├── raw/                    # Original data files (do not modify)
│   ├── sst.nc             # Sea Surface Temperature data (NetCDF format)
│   └── precip.nc          # Precipitation data (NetCDF format)
├── processed/             # Processed data files
│   ├── sst_anomalies.nc
│   ├── precip_anomalies.nc
│   ├── sst_standardized.npy
│   └── precip_standardized.npy
└── README.md              # This file
```

## Data Requirements

### Sea Surface Temperature (SST) Data
- **Format**: NetCDF (.nc) or similar
- **Variables**: Sea surface temperature (monthly mean)
- **Dimensions**: Time × Latitude × Longitude
- **Units**: Celsius or Kelvin
- **Temporal Resolution**: Monthly
- **Recommended Sources**:
  - NOAA Optimum Interpolation SST (OISST)
  - HadISST
  - ERSST

### Precipitation Data
- **Format**: NetCDF (.nc) or similar
- **Variables**: Precipitation (monthly total or anomaly)
- **Dimensions**: Time × Latitude × Longitude
- **Units**: mm/day or mm/month
- **Temporal Resolution**: Monthly
- **Recommended Sources**:
  - GPCC (Global Precipitation Climatology Centre)
  - CMAP (CPC Merged Analysis of Precipitation)
  - CRU (Climate Research Unit)

## Data Download Instructions

### Option 1: Using Python (xarray + intake-esm)
```python
import xarray as xr
import intake

# Example: Download OISST data
cat = intake.open_catalog('https://raw.githubusercontent.com/intake/intake-esm-datastore/master/catalogs/glade-cesm2.json')
ds = cat['NOAA'].to_dask()
```

### Option 2: Manual Download
- NOAA OISST: https://www.ncei.noaa.gov/products/optimum-interpolation-sst
- GPCC: https://www.dwd.de/DE/leistungen/met_verfahren/mosmix/mosmix_stationskatalog.cfg
- HadISST: https://www.metoffice.gov.uk/hadobs/hadisst/

## Data Preprocessing

The pipeline includes automatic preprocessing:

1. **Climatology Removal**: Calculate 30-year climatology and remove to get anomalies
2. **Detrending**: Remove linear or polynomial trends
3. **Standardization**: Z-score normalization (mean=0, std=1)
4. **Quality Control**: Remove missing values and outliers
5. **Spatial Subsetting**: Select region of interest

Modify `config/config.yaml` to customize preprocessing parameters.

## Data Format Conversion

If your data is in a different format, use xarray:

```python
import xarray as xr

# Load from various formats
ds = xr.open_dataset('data.nc')      # NetCDF
ds = xr.open_dataarray('data.h5')    # HDF5
ds = xr.from_pandas(df)              # Pandas DataFrame

# Save to NetCDF
ds.to_netcdf('output.nc')
```

## Storage Recommendations

- Keep raw data in `data/raw/` without modifications
- Store processed data in `data/processed/`
- Use efficient formats (NetCDF with compression) for large files
- Document data sources and preprocessing steps

## Notes

- Large NetCDF files can be slow to load; consider using xarray with Dask for lazy loading
- Time series should be aligned between SST and precipitation data
- Consider temporal subsetting for computational efficiency during development
