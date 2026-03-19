# import python packages
import xarray as xr
import pandas as pd
import gdown
import xarray as xr
import h5py
import numpy as np
import matplotlib.pyplot as plt
from exactextract import exact_extract
import geopandas as gpd
import rioxarray
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset



def process_data(historical, natural, observations, original, pop_density, pop_density_original_resolution, bias_not_downscaled=None):
    
    # crop the edge to match pop density format
    historical = historical.isel(lat=slice(1,83), lon=slice(1,83))
    natural = natural.isel(lat=slice(1,83), lon=slice(1,83))
    observations = observations.isel(lat=slice(1,83), lon=slice(1,83))

    # Align spatial reference systems
    historical.rio.write_crs("EPSG:4674", inplace=True)
    natural.rio.write_crs("EPSG:4674", inplace=True)
    observations.rio.write_crs("EPSG:4674", inplace=True)
    original.rio.write_crs("EPSG:4674", inplace=True)
    pop_density = pop_density.rio.write_crs("EPSG:4674", inplace=True)
    pop_density_original_resolution = pop_density_original_resolution.rio.write_crs("EPSG:4674", inplace=True)

    # convert kelvin to celsius where necessary
    original['tas'] = original['tas'] - 273.15
    
    # Convert original and pop_density longitudes from 0-360 to -180-180
    original = original.assign_coords(lon=(original.lon + 180) % 360 - 180).sortby('lon')
    pop_density_original_resolution = pop_density_original_resolution.assign_coords(
        lon=(pop_density_original_resolution.lon + 180) % 360 - 180
    ).sortby('lon')

    # Process bias_not_downscaled: same resolution as original so needs lon conversion
    bias_not_downscaled.rio.write_crs("EPSG:4674", inplace=True)
    bias_not_downscaled = bias_not_downscaled.assign_coords(
            lon=(bias_not_downscaled.lon + 180) % 360 - 180
     ).sortby('lon')
    # rassign data variable to tas
    bias_not_downscaled = bias_not_downscaled.rename({'__xarray_dataarray_variable__': 'tas'})
    
    return historical, natural, observations, original, pop_density, pop_density_original_resolution, bias_not_downscaled


def plot_timeseries(timeseries_historical, timeseries_natural, title='Daily Mean Temperature'):
    """
    Plot population-weighted temperature timeseries for historical and natural
    forcing runs, with an inset showing the first austral summer.

    Parameters
    ----------
    timeseries_historical : pd.DataFrame
        DataFrame with a 'population_weighted_mean' column and a DatetimeIndex.
    timeseries_natural : pd.DataFrame
        DataFrame with a 'population_weighted_mean' column and a DatetimeIndex.
    title : str, optional
        Title for the main plot.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(timeseries_historical.index, timeseries_historical['population_weighted_mean'],
            label='Historical (0.5°)', linewidth=0.8, color='orange')
    ax.plot(timeseries_natural.index, timeseries_natural['population_weighted_mean'],
            label='Natural (0.5°)', linewidth=0.8, color='green')

    ax.set_xlabel('Date')
    ax.set_ylabel('Population-weighted Temperature (°C)')
    ax.set_title(title)
    ax.legend(loc='lower right')

    # Inset: pick the first austral summer (Nov-Mar) within the timeseries
    index = timeseries_historical.index
    first_nov = index[((index.month == 11) | (index.month == 12) | (index.month <= 3))][0]
    summer_start = pd.Timestamp(year=first_nov.year, month=11, day=1)
    summer_end = pd.Timestamp(year=first_nov.year + 1, month=3, day=31)
    summer_start = max(summer_start, index[0])
    summer_end = min(summer_end, index[-1])

    axins = inset_axes(ax, width='30%', height='40%', loc='upper left', borderpad=2)

    mask_hist = (timeseries_historical.index >= summer_start) & (timeseries_historical.index <= summer_end)
    mask_nat  = (timeseries_natural.index >= summer_start) & (timeseries_natural.index <= summer_end)

    axins.plot(timeseries_historical.index[mask_hist], timeseries_historical['population_weighted_mean'][mask_hist],
               linewidth=1, color='orange')
    axins.plot(timeseries_natural.index[mask_nat], timeseries_natural['population_weighted_mean'][mask_nat],
               linewidth=1, alpha=0.8, color='green')
    axins.set_title(f'Summer {summer_start.year}/{str(summer_end.year)[2:]}', fontsize=9)
    axins.tick_params(labelsize=7)
    axins.xaxis.set_tick_params(rotation=30)

    mark_inset(ax, axins, loc1=1, loc2=2, fc='none', ec='gray', linewidth=0.8)

    plt.tight_layout()
    plt.show()
    
def plot_climatology(timeseries_historical, timeseries_natural, timeseries_obs, timeseries_original=None,
                     title='Climatology', smooth_window=15):
    """
    Calculate and plot smoothed day-of-year climatologies for historical,
    natural, observational (and optionally original-resolution) timeseries.
    X-axis is labelled by month (ticked at mid-month, ~day 15).
    The area between the historical and natural curves is hatched in orange.

    Parameters
    ----------
    timeseries_historical : pd.DataFrame
        DataFrame with a 'population_weighted_mean' column and a DatetimeIndex.
    timeseries_natural : pd.DataFrame
        DataFrame with a 'population_weighted_mean' column and a DatetimeIndex.
    timeseries_obs : pd.DataFrame
        DataFrame with a 'population_weighted_mean' column and a DatetimeIndex.
    timeseries_original : pd.DataFrame, optional
        Original-resolution DataFrame. Plotted as a dashed gray line if provided.
    title : str, optional
        Title for the plot.
    smooth_window : int, optional
        Window size for the rolling mean smoother. Default is 15.
    """
    def _smooth(ts):
        clim = ts.groupby(ts.index.dayofyear).mean()
        return clim['population_weighted_mean'].rolling(window=smooth_window, center=True).mean()

    # Mid-day of each month in a non-leap year
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_ticks  = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]

    fig, ax = plt.subplots(figsize=(14, 5))

    if timeseries_original is not None:
        s = _smooth(timeseries_original)
        ax.plot(s.index, s, label='Original (~2.8°)', linewidth=0.8, linestyle='--', color='gray')

    s_hist = _smooth(timeseries_historical)
    s_nat  = _smooth(timeseries_natural)
    s_obs  = _smooth(timeseries_obs)

    # align indices for fill_between (use common days)
    common = s_hist.index.intersection(s_nat.index)
    ax.fill_between(common, s_hist.reindex(common), s_nat.reindex(common),
                    facecolor='none', edgecolor='orange', hatch='///', linewidth=0.0,
                    label='Hist–Nat difference')

    ax.plot(s_hist.index, s_hist, label='Historical (0.5°)', linewidth=1, color='orange')
    ax.plot(s_nat.index,  s_nat,  label='Natural (0.5°)',    linewidth=1, color='green')
    ax.plot(s_obs.index,  s_obs,  label='Observations (0.5°)', linewidth=1.4, color='black')

    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels, fontsize=12)
    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('Population-weighted Temperature (°C)', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend()
    plt.tight_layout()
    plt.show()
    

def plot_region(region_name, shapefile, original, historical, bias_not_downscaled=None, vmin=24, vmax=30, n_levels=7):
    """
    Plot a comparison of original vs (optionally) bias-corrected vs downscaled temperature
    for a given RGI region, with a Brazil locator inset and discrete colorbar.

    Parameters
    ----------
    region_name : str
        Name of the region to highlight (matched against 'nome_rgi' column).
    shapefile : GeoDataFrame
        Brazilian municipality shapefile.
    original : xr.Dataset
        Original resolution temperature dataset.
    historical : xr.Dataset
        Downscaled historical temperature dataset.
    bias_not_downscaled : xr.Dataset, optional
        Bias-corrected but not downscaled dataset. If provided, a middle panel is added.
    vmin, vmax : float
        Colorbar range in °C.
    n_levels : int
        Number of discrete colour levels. Default is 7.
    """
    import matplotlib as mpl

    region = shapefile[shapefile['nome_rgi'] == region_name]
    bounds = region.total_bounds
    brazil_bounds = shapefile.total_bounds

    step = max(1, int(np.round((vmax - vmin) / n_levels)))
    levels = np.arange(int(vmin), int(vmax) + step, step)
    n_colors = len(levels) - 1
    cmap = plt.get_cmap('RdYlBu_r', n_colors)
    norm = mpl.colors.BoundaryNorm(levels, ncolors=n_colors)

    datasets = [
        (original.sel(time='2005-01-01').tas,  3.0, 'Original Resolution (~2.8°)'),
    ]
    if bias_not_downscaled is not None:
        datasets.append((bias_not_downscaled.sel(time='2005-01-01').tas, 3.0, 'Bias corrected (~1°)'))
    datasets.append((historical.sel(time='2005-01-01').tas, 1.0, 'Downscaled (0.5°)'))

    n_panels = len(datasets)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    if n_panels == 1:
        axes = [axes]

    for ax, (data, pad, title) in zip(axes, datasets):
        im = data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            add_colorbar=False,
            zorder=1,
        )
        shapefile.plot(ax=ax, color='lightgrey', edgecolor='grey', linewidth=0.5, transform=ccrs.PlateCarree(), zorder=2, alpha=0.4)
        region.boundary.plot(ax=ax, color='black', linewidth=1.2, transform=ccrs.PlateCarree(), zorder=3)
        ax.set_extent([bounds[0] - pad, bounds[2] + pad, bounds[1] - pad, bounds[3] + pad])
        ax.set_title(title, fontsize=18)

    # shared discrete colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.7, pad=0.02,
                        ticks=levels, spacing='uniform')
    cbar.set_label('Temperature (°C)')

    # locator map: right side of figure, no box
    ax_inset = fig.add_axes([0.75, 0.75, 0.2, 0.35], projection=ccrs.PlateCarree())
    shapefile.plot(ax=ax_inset, color='lightgrey', edgecolor='grey', linewidth=0.3, transform=ccrs.PlateCarree())
    region.plot(ax=ax_inset, color='red', edgecolor='red', linewidth=0.5, transform=ccrs.PlateCarree())
    ax_inset.set_extent([brazil_bounds[0], brazil_bounds[2], brazil_bounds[1], brazil_bounds[3]])
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.set_title('Location', fontsize=12, pad=2)
    for spine in ax_inset.spines.values():
        spine.set_visible(False)

    fig.suptitle('Bias correction and downscaling', fontsize=24, y=1.01)
    plt.show()

