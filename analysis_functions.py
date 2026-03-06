import xarray as xr
import numpy as np
import xesmf as xe

def fix_longitudes(da):
    """
    Detects the longitude coordinate name and shifts from [0, 360] to [-180, 180] if needed.

    Parameters:
        da (xr.DataArray or xr.Dataset): Input data with a longitude coordinate.

    Returns:
        xr.DataArray or xr.Dataset: Data with longitudes corrected if necessary.
    """
    # Try common longitude coordinate names
    lon_names = ['lon', 'lons', 'longitude']
    lon_name = next((name for name in lon_names if name in da.coords), None)

    if lon_name is None:
        print("No longitude coordinate found. Skipping longitude fix.")
        return da

    lon_vals = da[lon_name].values
    if lon_vals.min() >= 0 and lon_vals.max() > 180:
        da = da.assign_coords({lon_name: (((da[lon_name] + 180) % 360) - 180)})
        da = da.sortby(lon_name)
        print(f"Longitudes in '{lon_name}' shifted to [-180, 180] range.")
    else:
        print(f"ℹLongitudes in '{lon_name}' already in [-180, 180] range — no shift applied.")

    return da

def process_model_data(
    data: xr.DataArray | xr.Dataset,
    region_from: xr.DataArray | xr.Dataset
) -> xr.DataArray:
    """
    Preprocess model data for bias correction.

    Steps:
    - Remove 'height' coordinate if present
    - Convert from Kelvin to Celsius
    - Fix longitudes
    - Regrid model data to a standard 1° or 2° grid depending on resolution
    - Select Brazil region based on 'region_from' bounding box
    - Standardize variable name, calendar, and chunking

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Raw model temperature data (e.g., tas).
    region_from : xr.DataArray or xr.Dataset
        Dataset used to define the Brazil bounding box (usually obs).

    Returns
    -------
    model_brazil : xr.DataArray
        Processed model data over Brazil, ready for bias correction.
    """
    data = data.copy()

    if "height" in data.coords:
        del data["height"]

    data = data - 273.15
    data = fix_longitudes(data)

    lat_step = 2 if float(np.diff(data.lat)[0]) > 1.5 else 1
    lon_step = 2 if float(np.diff(data.lon)[0]) > 1.5 else 1

    lats = np.arange(-90, 90.5, lat_step)
    lons = np.arange(-180, 180.5, lon_step)
    target_grid = xr.Dataset(coords={"lat": lats, "lon": lons})
    target_grid["dummy"] = (("lat", "lon"), np.zeros((len(lats), len(lons))))

    data.data = np.ascontiguousarray(data.data)

    regridder = xe.Regridder(data, target_grid, method="bilinear", unmapped_to_nan=True, reuse_weights=False)
    model_rg = regridder(data)
    # model_rg = model_rg.chunk({'time': -1, 'lat': 60, 'lon': 60})

    lat_min = float(region_from.lat.min())
    lat_max = float(region_from.lat.max())
    lon_min = float(region_from.lon.min())
    lon_max = float(region_from.lon.max())

    lat_tol = lat_step / 2
    lon_tol = lon_step / 2

    lat_slice = slice(
        model_rg.lat.sel(lat=lat_min - lat_tol, method="nearest").values,
        model_rg.lat.sel(lat=lat_max + lat_tol, method="nearest").values
    )
    lon_slice = slice(
        model_rg.lon.sel(lon=lon_min - lon_tol, method="nearest").values,
        model_rg.lon.sel(lon=lon_max + lon_tol, method="nearest").values
    )

    model_brazil = model_rg.sel(lat=lat_slice, lon=lon_slice)
    model_brazil.name = "tas"
    model_brazil = model_brazil.transpose("lat", "lon", "time") #.chunk("auto")
    model_brazil = model_brazil.convert_calendar("proleptic_gregorian", use_cftime=True)
    model_brazil['time'] = model_brazil['time'].dt.floor('D')

    return model_brazil

def regrid_obs_to_model(
    obs: xr.DataArray | xr.Dataset,
    model_brazil: xr.DataArray
) -> xr.DataArray:
    """
    Regrid observation data to match the processed model grid over Brazil.

    Parameters
    ----------
    obs : xr.DataArray or xr.Dataset
        Raw global observation dataset.
    model_brazil : xr.DataArray
        Preprocessed model data defining the target grid over Brazil.

    Returns
    -------
    obs_rg : xr.DataArray
        Observation data regridded to match the model grid.
    """
    obs = obs.copy()

    regridder = xe.Regridder(obs, model_brazil, method="bilinear", unmapped_to_nan=True, reuse_weights=False)
    obs_rg = regridder(obs)

    obs_rg.name = "tas"
    obs_rg = obs_rg.transpose("lat", "lon", "time") #.chunk("auto")
    obs_rg = obs_rg.convert_calendar("proleptic_gregorian", use_cftime=True)

    return obs_rg

def compare_grids(ds_coarse, ds_fine, dim_name):

    """
    Checks if a fine grid is a uniform subdivision of a coarse grid along a given dimension.

    Parameters:
        ds_coarse (xr.Dataset or xr.DataArray): Coarse-resolution data.
        ds_fine (xr.Dataset or xr.DataArray): Fine-resolution data.
        dim_name (str): Name of the coordinate dimension to compare (e.g., 'lat', 'lon').

    Returns:
        float: Maximum absolute difference between expected and actual fine grid coordinates.
    """

    x = ds_coarse[dim_name].values
    y = ds_fine[dim_name].values

    # Downscaling factor (how many fine points per coarse point)
    f = len(y) // len(x)
    print(f"Downscaling factor for {dim_name}: {f}")

    dx = np.diff(x)
    # Average spacing s at cell edges
    s = 0.5 * (np.concatenate(([dx[0]], dx)) + np.concatenate((dx, [dx[-1]])))

    t = np.arange(1, f + 1) / f  # fractional positions within each coarse cell
    y_delta = np.repeat(s, f) * np.tile(t - 0.5 * t[0], len(x))
    y_expected = np.repeat(x - 0.5 * s, f) + y_delta

    # Print results and comparison
    # print(f"Actual {dim_name} values (fine grid):")
    # print(y)
    # print(f"Expected {dim_name} values (fine grid):")
    # print(y_expected)
    print(f"Do actual and expected match? {np.allclose(y, y_expected)}")
    print(f"Max absolute difference: {np.max(np.abs(y - y_expected))}")

    sign = np.sign(np.mean(y_expected - y))

    return sign * np.max(np.abs(y - y_expected))

def trim_model_and_obs_to_match(model_data, obs_data, target_step):

    """
    Trim model and observation datasets to compatible grids for downscaling.

    Parameters:
        model_data (xr.DataArray or xr.Dataset): Model data with lat/lon coords.
        obs_data (xr.DataArray or xr.Dataset): Observation data with lat/lon coords.
        target_step (float): Target grid resolution in degrees.

    Returns:
        tuple: (trimmed model data, trimmed observation data) with aligned grids.
    """

    # Extract coordinates
    lat_model = model_data.lat 
    lon_model = model_data.lon
    lat_obs = obs_data.lat
    lon_obs = obs_data.lon

    # Calculate model grid spacing
    lat_step = float(np.diff(lat_model)[0])
    lon_step = float(np.diff(lon_model)[0])

    # Calculate scale factors (model resolution ÷ target resolution)
    lat_scalef = lat_step / target_step
    lon_scalef = lon_step / target_step

    # Ensure scale factors are integers
    if not lat_scalef.is_integer() or not lon_scalef.is_integer():
        raise ValueError(f"Scale factors must be integers. Got: {lat_scalef} (lat), {lon_scalef} (lon)")

    lat_scalef = int(lat_scalef)
    lon_scalef = int(lon_scalef)

    # Determine original obs sizes
    obs_lat_len = len(lat_obs)
    obs_lon_len = len(lon_obs)

    # Adjust obs sizes down to nearest multiple of scale factor
    obs_lat_len_adj = (obs_lat_len // lat_scalef) * lat_scalef
    obs_lon_len_adj = (obs_lon_len // lon_scalef) * lon_scalef

    # Trim obs if needed
    lat_obs_trim = obs_lat_len - obs_lat_len_adj
    lon_obs_trim = obs_lon_len - obs_lon_len_adj

    lat_obs_start = lat_obs_trim // 2
    lat_obs_end = lat_obs_start + obs_lat_len_adj
    lon_obs_start = lon_obs_trim // 2
    lon_obs_end = lon_obs_start + obs_lon_len_adj

    obs_trimmed = obs_data.isel(lat=slice(lat_obs_start, lat_obs_end),
                                lon=slice(lon_obs_start, lon_obs_end))

    # Determine matching model size
    model_lat_len = obs_lat_len_adj // lat_scalef
    model_lon_len = obs_lon_len_adj // lon_scalef

    lat_model_trim = len(lat_model) - model_lat_len
    lon_model_trim = len(lon_model) - model_lon_len

    if lat_model_trim < 0 or lon_model_trim < 0:
        raise ValueError("Model grid is too small to match downscaled obs resolution.")

    lat_model_start = lat_model_trim // 2
    lat_model_end = lat_model_start + model_lat_len
    lon_model_start = lon_model_trim // 2
    lon_model_end = lon_model_start + model_lon_len

    model_trimmed = model_data.isel(lat=slice(lat_model_start, lat_model_end),
                                    lon=slice(lon_model_start, lon_model_end))

    return model_trimmed, obs_trimmed

def prepare_bias_corrected_and_obs(
    bias_corrected: xr.Dataset, 
    obs_05x05: xr.Dataset,
    obs_resolution: float = 0.5
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Aligns bias-corrected model output with observed data:
    - Trims lat and lon so that len(obs) / len(model) = integer (scale factor)
    - Ensures spatial grid centers align
    - Standardizes variable name and dimension order
    - Converts calendar to Proleptic Gregorian

    Parameters
    ----------
    bias_corrected : xr.Dataset
        Bias-corrected model output with dimensions (time, lat, lon).
    obs_05x05 : xr.Dataset
        Observational dataset to match with model.
    obs_resolution : float, optional
        Spatial resolution of observational data (default is 0.5).

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Tuple of (bias_corrected_trimmed, obs_05x05_trimmed).
    """

    # Trim model and obs to have compatible lat lon grids 
    bias_trim, obs_trim = trim_model_and_obs_to_match(
        bias_corrected, obs_05x05, obs_resolution
    )

    # Align grid centers
    lat_adjust = compare_grids(bias_trim, obs_trim, "lat")
    lon_adjust = compare_grids(bias_trim, obs_trim, "lon")
    bias_trim = bias_trim.assign_coords(
        lat=bias_trim.lat - lat_adjust,
        lon=bias_trim.lon - lon_adjust
    )

    # Rename and reorder dimensions
    obs_trim.name = "tas"
    obs_trim = obs_trim.transpose("lat", "lon", "time")
    obs_trim = obs_trim.convert_calendar("proleptic_gregorian", use_cftime=True)

    bias_trim = bias_trim.transpose("lat", "lon", "time")
    bias_trim = bias_trim.convert_calendar("proleptic_gregorian", use_cftime=True)

    return bias_trim, obs_trim