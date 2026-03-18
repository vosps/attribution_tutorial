# (C) 2022 Potsdam Institute for Climate Impact Research (PIK)
# 
# This file is part of ISIMIP3BASD.
#
# ISIMIP3BASD is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ISIMIP3BASD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ISIMIP3BASD. If not, see <http://www.gnu.org/licenses/>.
#
# These functions were modified by R. Lieber in March 2026 to support array-based input for notebook usage. 



"""
Statistical downscaling
=======================

Provides functions for statistical downscaling of climate simulation data using
climate observation data with the same temporal and higher spatial resolution.

"""



import warnings
import numpy as np
import utility_functions as uf
from itertools import product



def weighted_sum_preserving_mbcn(
        x_obs, x_sim_coarse, x_sim,
        sum_weights, rotation_matrices=[], n_quantiles=50):
    """
    Applies the core of the modified MBCn algorithm for statistical downscaling
    as described in Lange (2019) <https://doi.org/10.5194/gmd-12-3055-2019>.

    Parameters
    ----------
    x_obs : (M,N) ndarray
        Array of N observed time series of M time steps each at fine spatial
        resolution.
    x_sim_coarse : (M,) array
        Array of simulated time series of M time steps at coarse spatial
        resolution.
    x_sim : (M,N) ndarray
        Array of N simulated time series of M time steps each at fine spatial
        resolution, derived from x_sim_coarse by bilinear interpolation.
    sum_weights : (N,) array
        Array of N grid cell-area weights.
    rotation_matrices : list of (N,N) ndarrays, optional
        List of orthogonal matrices defining a sequence of rotations in the  
        second dimension of x_obs and x_sim.
    n_quantiles : int, optional
        Number of quantile-quantile pairs used for non-parametric quantile
        mapping.

    Returns
    -------
    x_sim : (M,N) ndarray
        Result of application of the modified MBCn algorithm.

    """
    # initialize total rotation matrix
    n_variables = sum_weights.size
    o_total = np.diag(np.ones(n_variables))

    # p-values in percent for non-parametric quantile mapping
    p = np.linspace(0., 1., n_quantiles+1)

    # normalise the sum weights vector to length 1
    sum_weights = sum_weights / np.sqrt(np.sum(np.square(sum_weights)))

    # rescale x_sim_coarse for initial step of algorithm
    x_sim_coarse = x_sim_coarse * np.sum(sum_weights)

    # iterate
    n_loops = len(rotation_matrices) + 2
    for i in range(n_loops):
        if not i:  # rotate to the sum axis
            o = uf.generate_rotation_matrix_fixed_first_axis(sum_weights)
        elif i == n_loops - 1:  # rotate back to original axes for last qm
            o = o_total.T
        else:  # do random rotation
            o = rotation_matrices[i-1]

        # compute total rotation
        o_total = np.dot(o_total, o)

        # rotate data
        x_sim = np.dot(x_sim, o)
        x_obs = np.dot(x_obs, o)
        sum_weights = np.dot(sum_weights, o)

        if not i:
            # restore simulated values at coarse grid scale
            x_sim[:,0] = x_sim_coarse

            # quantile map observations to values at coarse grid scale
            q_sim = uf.percentile1d(x_sim_coarse, p)
            q_obs = uf.percentile1d(x_obs[:,0], p)
            x_obs[:,0] = \
                uf.map_quantiles_non_parametric_with_constant_extrapolation(
                x_obs[:,0], q_obs, q_sim)
        else:
            # do univariate non-parametric quantile mapping for every variable
            x_sim_previous = x_sim.copy()
            for j in range(n_variables):
                q_sim = uf.percentile1d(x_sim[:,j], p)
                q_obs = uf.percentile1d(x_obs[:,j], p)
                x_sim[:,j] = \
                    uf.map_quantiles_non_parametric_with_constant_extrapolation(
                    x_sim[:,j], q_sim, q_obs)

            # preserve weighted sum of original variables
            if i < n_loops - 1:
                x_sim -= np.outer(np.dot(
                   x_sim - x_sim_previous, sum_weights), sum_weights)

    return x_sim



def downscale_one_month(
        data, long_term_mean,
        lower_bound=None, lower_threshold=None,
        upper_bound=None, upper_threshold=None,
        randomization_seed=None, **kwargs):
    """
    1. Replaces invalid values in time series.
    2. Replaces values beyond thresholds by random numbers.
    3. Applies the modified MBCn algorithm for statistical downscaling.
    4. Replaces values beyond thresholds by the respective bound.

    Parameters
    ----------
    data : dict of str : masked ndarray
        Keys : 'obs_fine', 'sim_coarse', 'sim_coarse_remapbil'.
        Values : arrays of shape (M,N), (M,), (M,N).
    long_term_mean : dict of str : scalar or array
        Keys : 'obs_fine', 'sim_coarse', 'sim_coarse_remapbil'.
        Values : scalar (for key 'sim_coarse') or array respresenting the
        average of all valid values in the complete time series for one climate
        variable and one location.
    lower_bound : float, optional
        Lower bound of values in data.
    lower_threshold : float, optional
        Lower threshold of values in data. All values below this threshold are
        replaced by random numbers between lower_bound and lower_threshold
        before application of the modified MBCn algorithm.
    upper_bound : float, optional
        Upper bound of values in data.
    upper_threshold : float, optional
        Upper threshold of values in data. All values above this threshold are
        replaced by random numbers between upper_threshold and upper_bound
        before application of the modified MBCn algorithm.
    randomization_seed : int, optional
        Used to seed the random number generator before replacing values beyond
        the specified thresholds.

    Returns
    -------
    x_sim_fine : (M,N) ndarray
        Result of application of the modified MBCn algorithm.

    Other Parameters
    ----------------
    **kwargs : Passed on to weighted_sum_preserving_mbcn.
    
    """
    x = {}
    for key, d in data.items():
        # remove invalid values from masked array and store resulting data array
        x[key] = uf.sample_invalid_values(
            d, randomization_seed, long_term_mean[key])[0]

        # randomize censored values, use high powers to create many values close
        # to the bounds as this keeps weighted sums similar to original values
        x[key] = uf.randomize_censored_values(x[key], 
            lower_bound, lower_threshold, upper_bound, upper_threshold,
            False, False, randomization_seed, 10., 10.)

    # downscale
    x_sim_coarse_remapbil = x['sim_coarse_remapbil'].copy()
    x_sim_fine = weighted_sum_preserving_mbcn(
        x['obs_fine'], x['sim_coarse'], x['sim_coarse_remapbil'], **kwargs)

    # de-randomize censored values
    uf.randomize_censored_values(x_sim_fine, 
        lower_bound, lower_threshold, upper_bound, upper_threshold, True, True)

    # make sure there are no invalid values
    uf.assert_no_infs_or_nans(x_sim_coarse_remapbil, x_sim_fine)

    return x_sim_fine



def downscale_one_location_array(
        i_loc_coarse, variable,
        downscaling_factors, ascending, circular, sum_weights,
        months=[1,2,3,4,5,6,7,8,9,10,11,12],
        lower_bound=None, lower_threshold=None,
        upper_bound=None, upper_threshold=None,
        if_all_invalid_use=np.nan, **kwargs):
    """
    Applies the modified MBCn algorithm for statistical downscaling calendar
    month by calendar month to climate data within one coarse grid cell.
    This version works directly with in-memory arrays.

    Parameters
    ----------
    i_loc_coarse : tuple
        Coarse location index.
    variable : str
        Name of variable to be downscaled (used for compatibility).
    downscaling_factors : array of ints
        Downscaling factors for all grid dimensions.
    ascending : tuple of booleans
        Whether coordinates are monotonically increasing.
    circular : tuple of booleans
        Whether coordinates are circular.
    sum_weights : ndarray
        Array of fine grid cell area weights.
    months : list, optional
        List of ints from {1,...,12} representing calendar months for which 
        results of statistical downscaling are to be returned.
    lower_bound : float, optional
        Lower bound of values in data.
    lower_threshold : float, optional
        Lower threshold of values in data.
    upper_bound : float, optional
        Upper bound of values in data.
    upper_threshold : float, optional
        Upper threshold of values in data.
    if_all_invalid_use : float, optional
        Used to replace invalid values if there are no valid values.

    Returns
    -------
    None (updates global sim_fine_data in place).

    Other Parameters
    ----------------
    **kwargs : Passed on to downscale_one_month.

    """
    global obs_fine_data, sim_coarse_data, sim_fine_data, month_numbers, space_shapes, grids
    
    # get local input data
    i_loc_fine = tuple(slice(df * i_loc_coarse[i], df * (i_loc_coarse[i] + 1))
        for i, df in enumerate(downscaling_factors))
    j_loc_fine = tuple(np.arange(s.start, s.stop) for s in i_loc_fine)
    oshape = lambda key: (np.prod(downscaling_factors), month_numbers[key].size)
    data = {}
    
    # Extract observation data at fine resolution
    key = 'obs_fine'
    x = obs_fine_data[i_loc_fine]
    data[key] = x.reshape(oshape(key)).T
    
    # Extract simulation data at coarse resolution
    key = 'sim_coarse'
    x = sim_coarse_data[i_loc_coarse]
    data[key] = x
    
    # Use interpolated coarse simulation data at fine resolution
    key = 'sim_coarse_remapbil'
    x = sim_fine_data[i_loc_fine]
    data[key] = x.reshape(oshape(key)).T

    # abort here if there are only missing values in at least one time series
    # do not abort though if the if_all_invalid_use option has been specified
    if np.isnan(if_all_invalid_use):
        if uf.only_missing_values_in_at_least_one_time_series(data):
            print(i_loc_coarse, 'skipped due to missing data')
            return None

    # otherwise continue
    # print(i_loc_coarse) # Uncomment this to print location index for each coarse grid cell being processed. Note can output a lot of lines if there are many coarse grid cells.

    # compute mean value over all time steps for invalid value sampling
    long_term_mean = {}
    for key, d in data.items():
        long_term_mean[key] = uf.average_valid_values(d, if_all_invalid_use,
            lower_bound, lower_threshold, upper_bound, upper_threshold)

    # do statistical downscaling calendar month by calendar month
    result = data['sim_coarse_remapbil'].copy()
    sum_weights_loc = sum_weights[i_loc_fine].flatten()
    data_this_month = {}
    for month in months:
        # extract data
        for key, d in data.items():
            m = month_numbers[key] == month
            assert np.any(m), f'no data found for month {month} in {key}'
            data_this_month[key] = d[m]

        # do statistical downscaling
        result_this_month = downscale_one_month(data_this_month, long_term_mean,
            lower_bound, lower_threshold, upper_bound, upper_threshold,
            sum_weights=sum_weights_loc, **kwargs)
    
        # put downscaled data into result
        m = month_numbers['sim_coarse_remapbil'] == month
        result[m] = result_this_month

    # save local result of statistical downscaling
    for i, i_loc in enumerate(product(*j_loc_fine)):
        sim_fine_data[i_loc] = result[:,i]

    return None






def downscale_array_based(
        obs_fine_array, sim_coarse_array, sim_fine_remapbil,
        month_numbers_obs, month_numbers_sim,
        n_processes=1, **kwargs):
    """
    Applies the modified MBCn algorithm for statistical downscaling calendar
    month by calendar month and coarse grid cell by coarse grid cell.
    This version works directly with in-memory arrays.

    Parameters
    ----------
    obs_fine_array : ndarray
        Observation data at fine resolution. Shape: (lat_fine, lon_fine, time)
    sim_coarse_array : ndarray
        Simulation data at coarse resolution. Shape: (lat_coarse, lon_coarse, time)
    sim_fine_remapbil : ndarray
        Simulated data interpolated to fine resolution. Shape: (lat_fine, lon_fine, time)
    month_numbers_obs : ndarray
        Month numbers for observation data. Length: time
    month_numbers_sim : ndarray
        Month numbers for simulation data. Length: time
    n_processes : int, optional
        Number of processes used for parallel processing (currently only 1 is supported).

    Other Parameters
    ----------------
    **kwargs : Passed on to downscale_one_location.

    Returns
    -------
    result : ndarray
        Downscaled simulations at fine resolution. Shape: (lat_fine, lon_fine, time)
    """
    # Set up global arrays for direct access
    global obs_fine_data, sim_coarse_data, sim_fine_data
    global month_numbers, space_shapes, grids
    
    # Initialize global dictionaries if not already done
    if 'space_shapes' not in globals():
        space_shapes = {}
    if 'month_numbers' not in globals():
        month_numbers = {}
    if 'grids' not in globals():
        grids = {}
    
    obs_fine_data = obs_fine_array
    sim_coarse_data = sim_coarse_array
    sim_fine_data = sim_fine_remapbil.copy()
    
    # Update global variables with array shape information
    space_shapes['sim_coarse'] = sim_coarse_array.shape[:-1]
    space_shapes['obs_fine'] = obs_fine_array.shape[:-1]
    
    month_numbers['obs_fine'] = month_numbers_obs
    month_numbers['sim_coarse'] = month_numbers_sim
    month_numbers['sim_coarse_remapbil'] = month_numbers_sim
    
    # Process each coarse location
    i_locations_coarse = np.ndindex(space_shapes['sim_coarse'])
    
    # For now, only single-process mode supported
    result = np.empty_like(sim_fine_remapbil)
    for i_loc_coarse in i_locations_coarse:
        downscale_one_location_array(i_loc_coarse, **kwargs)
    
    return sim_fine_data


def main(obs_fine_arrays, sim_coarse_arrays, **kwargs):
    """
    Prepares and executes the application of the modified MBCn algorithm for
    statistical downscaling using in-memory arrays (suitable for notebook use).

    This function works with timeseries of any length, as long as all input
    arrays have consistent shapes and the metadata arrays match the time dimension.

    Parameters
    ----------
    obs_fine_arrays : list of arrays
        Observation data at fine resolution for each variable. Shape: (lat_fine, lon_fine, time)
    sim_coarse_arrays : list of arrays
        Simulation data at coarse resolution for each variable. Shape: (lat_coarse, lon_coarse, time)

    **kwargs : Additional parameters
        variable : list of strs
            Names of variables being downscaled.
        month_numbers : dict
            Month numbers for each dataset. Keys: 'obs_fine', 'sim_coarse'
            Values: arrays with length matching the time dimension
        years : dict, optional
            Years for each dataset (for compatibility with bias correction pattern)
        days : dict, optional
            Days of year for each dataset (for compatibility with bias correction pattern)
        downscaling_factors : tuple of ints, optional
            Downscaling factors for all grid dimensions (computed from array shapes if not provided)
        ascending : tuple of booleans, optional
            Whether coordinates are monotonically increasing (default: all True)
        circular : tuple of booleans, optional
            Whether coordinates are circular (default: all False)
        sum_weights : ndarray, optional
            Array of fine grid cell area weights (computed if not provided)
        months : list, optional
            List of ints from {1,...,12} representing calendar months (default: all)
        n_iterations : int, optional
            Number of iterations for statistical downscaling (default: 20)
        lower_bound : float, optional
            Lower bound of variable (default: None)
        upper_bound : float, optional
            Upper bound of variable (default: None)
        lower_threshold : float, optional
            Lower threshold of variable (default: None)
        upper_threshold : float, optional
            Upper threshold of variable (default: None)
        randomization_seed : int, optional
            Seed for random number generation
        n_quantiles : int, optional
            Number of quantiles for non-parametric quantile mapping (default: 50)
        if_all_invalid_use : float, optional
            Value to use if all values are invalid (default: np.nan)

    Returns
    -------
    downscaled_data : list of arrays
        Downscaled simulations at fine resolution for each variable.
        Each array has shape (lat_fine, lon_fine, time)

    Raises
    ------
    ValueError
        If input arrays have inconsistent shapes or metadata lengths don't match.
    """
    # Extract parameters from kwargs
    variable = kwargs.pop('variable')
    month_numbers_dict = kwargs.pop('month_numbers')
    
    # Set defaults for optional parameters
    months = kwargs.pop('months', [1,2,3,4,5,6,7,8,9,10,11,12])
    n_iterations = kwargs.pop('n_iterations', 20)
    lower_bound = kwargs.pop('lower_bound', None)
    lower_threshold = kwargs.pop('lower_threshold', None)
    upper_bound = kwargs.pop('upper_bound', None)
    upper_threshold = kwargs.pop('upper_threshold', None)
    randomization_seed = kwargs.pop('randomization_seed', None)
    n_quantiles = kwargs.pop('n_quantiles', 50)
    if_all_invalid_use = kwargs.pop('if_all_invalid_use', np.nan)
    
    # Extract geometric parameters with defaults
    downscaling_factors = kwargs.pop('downscaling_factors', None)
    ascending = kwargs.pop('ascending', None)
    circular = kwargs.pop('circular', None)
    sum_weights = kwargs.pop('sum_weights', None)
    
    # Map the keys to match downscaling expectations
    # The input uses 'obs_hist' and 'sim_hist' but downscaling expects 'obs_fine' and 'sim_coarse'
    month_numbers_obs = month_numbers_dict.get('obs_hist', month_numbers_dict.get('obs_fine'))
    month_numbers_sim = month_numbers_dict.get('sim_hist', month_numbers_dict.get('sim_coarse'))
    
    if month_numbers_obs is None:
        raise ValueError("month_numbers must contain 'obs_hist' or 'obs_fine' key")
    if month_numbers_sim is None:
        raise ValueError("month_numbers must contain 'sim_hist' or 'sim_coarse' key")
    
    # Validate input consistency
    n_variables = len(variable)
    if len(obs_fine_arrays) != n_variables or len(sim_coarse_arrays) != n_variables:
        raise ValueError('Number of variables does not match array list lengths')
    
    # Process first variable (downscaling is typically single-variable)
    obs_fine_array = obs_fine_arrays[0]
    sim_coarse_array = sim_coarse_arrays[0]
    
    if obs_fine_array.shape[-1] != month_numbers_obs.size:
        raise ValueError('obs_fine_array time dimension does not match month_numbers_obs length')
    if sim_coarse_array.shape[-1] != month_numbers_sim.size:
        raise ValueError('sim_coarse_array time dimension does not match month_numbers_sim length')
    
    # Compute downscaling factors from array shapes if not provided
    if downscaling_factors is None:
        spatial_dims_obs = obs_fine_array.shape[:-1]
        spatial_dims_sim = sim_coarse_array.shape[:-1]
        downscaling_factors = tuple(int(o / s) for o, s in zip(spatial_dims_obs, spatial_dims_sim))
    
    # Set defaults for grid properties
    if ascending is None:
        ascending = tuple(True for _ in downscaling_factors)
    if circular is None:
        circular = tuple(False for _ in downscaling_factors)
    
    # Compute sum_weights (grid cell weights at fine resolution) if not provided
    if sum_weights is None:
        # Create uniform weights by default
        sum_weights = np.ones(obs_fine_array.shape[:-1])
    
    # Create interpolated sim at fine resolution (simple bilinear interpolation)
    from scipy.interpolate import griddata
    spatial_dims_coarse = sim_coarse_array.shape[:-1]
    spatial_dims_fine = obs_fine_array.shape[:-1]
    
    sim_fine_remapbil = np.zeros((spatial_dims_fine + (sim_coarse_array.shape[-1],)))
    
    for t in range(sim_coarse_array.shape[-1]):
        # Create coordinate grids
        lat_coarse, lon_coarse = np.ogrid[0:spatial_dims_coarse[0], 0:spatial_dims_coarse[1]]
        lat_fine, lon_fine = np.ogrid[0:spatial_dims_fine[0], 0:spatial_dims_fine[1]]
        
        # Interpolate using nearest neighbor as simpler alternative
        for i in range(spatial_dims_fine[0]):
            for j in range(spatial_dims_fine[1]):
                i_coarse = int(i / downscaling_factors[0])
                j_coarse = int(j / downscaling_factors[1])
                i_coarse = min(i_coarse, spatial_dims_coarse[0] - 1)
                j_coarse = min(j_coarse, spatial_dims_coarse[1] - 1)
                sim_fine_remapbil[i, j, t] = sim_coarse_array[i_coarse, j_coarse, t]
    
    print('preparing downscaling ...')
    
    # Get list of rotation matrices to be used for all locations and months
    if randomization_seed is not None:
        np.random.seed(randomization_seed)
    rotation_matrices = [uf.generateCREmatrix(np.prod(downscaling_factors))
        for i in range(n_iterations)]

    # Do statistical downscaling
    spatial_dims_str = 'x'.join(str(d) for d in obs_fine_array.shape[:-1])
    print(f'downscaling with fine resolution shape ({spatial_dims_str}) ...')
    
    downscale_array_based(
        obs_fine_array, sim_coarse_array, sim_fine_remapbil,
        month_numbers_obs, month_numbers_sim,
        n_processes=1,
        variable=variable[0],
        downscaling_factors=downscaling_factors,
        ascending=ascending,
        circular=circular,
        sum_weights=sum_weights,
        rotation_matrices=rotation_matrices,
        months=months,
        lower_bound=lower_bound,
        lower_threshold=lower_threshold,
        upper_bound=upper_bound,
        upper_threshold=upper_threshold,
        n_quantiles=n_quantiles,
        if_all_invalid_use=if_all_invalid_use,
        randomization_seed=randomization_seed)
    
    # Return the downscaled data (stored in global variable)
    global sim_fine_data
    return [sim_fine_data]





