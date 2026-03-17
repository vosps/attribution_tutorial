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
Bias adjustment
===============

Provides functions for bias adjustment of climate simulation data using climate
observation data with the same spatial and temporal resolution.

"""



import warnings
import numpy as np
import scipy.stats as sps
import utility_functions as uf
from functools import partial


def map_quantiles_parametric_trend_preserving(
        x_obs_hist, x_sim_hist, x_sim_fut, 
        distribution=None, trend_preservation='additive',
        adjust_p_values=False,
        lower_bound=None, lower_threshold=None,
        upper_bound=None, upper_threshold=None,
        unconditional_ccs_transfer=False, trendless_bound_frequency=False,
        n_quantiles=50, p_value_eps=1e-10,
        max_change_factor=100., max_adjustment_factor=9.):
    """
    Adjusts biases using the trend-preserving parametric quantile mapping method
    described in Lange (2019) <https://doi.org/10.5194/gmd-12-3055-2019>.

    Parameters
    ----------
    x_obs_hist : array
        Time series of observed climate data representing the historical or
        training time period.
    x_sim_hist : array
        Time series of simulated climate data representing the historical or
        training time period.
    x_sim_fut : array
        Time series of simulated climate data representing the future or
        application time period.
    distribution : str, optional
        Kind of distribution used for parametric quantile mapping:
        [None, 'normal', 'weibull', 'gamma', 'beta', 'rice'].
    trend_preservation : str, optional
        Kind of trend preservation used for non-parametric quantile mapping:
        ['additive', 'multiplicative', 'mixed', 'bounded'].
    adjust_p_values : boolean, optional
        Adjust p-values for a perfect match in the reference period.
    lower_bound : float, optional
        Lower bound of values in x_obs_hist, x_sim_hist, and x_sim_fut.
    lower_threshold : float, optional
        Lower threshold of values in x_obs_hist, x_sim_hist, and x_sim_fut.
        All values below this threshold are replaced by lower_bound in the end.
    upper_bound : float, optional
        Upper bound of values in x_obs_hist, x_sim_hist, and x_sim_fut.
    upper_threshold : float, optional
        Upper threshold of values in x_obs_hist, x_sim_hist, and x_sim_fut.
        All values above this threshold are replaced by upper_bound in the end.
    unconditional_ccs_transfer : boolean, optional
        Transfer climate change signal using all values, not only those within
        thresholds.
    trendless_bound_frequency : boolean, optional
        Do not allow for trends in relative frequencies of values below lower
        threshold and above upper threshold.
    n_quantiles : int, optional
        Number of quantile-quantile pairs used for non-parametric quantile
        mapping.
    p_value_eps : float, optional
        In order to keep p-values with numerically stable limits, they are
        capped at p_value_eps (lower bound) and 1 - p_value_eps (upper bound).
    max_change_factor : float, optional
        Maximum change factor applied in non-parametric quantile mapping with
        multiplicative or mixed trend preservation.
    max_adjustment_factor : float, optional
        Maximum adjustment factor applied in non-parametric quantile mapping
        with mixed trend preservation.

    Returns
    -------
    x_sim_fut_ba : array
        Result of bias adjustment.

    """
    lower = lower_bound is not None and lower_threshold is not None
    upper = upper_bound is not None and upper_threshold is not None

    # use augmented quantile delta mapping to transfer the simulated
    # climate change signal to the historical observation
    i_obs_hist = np.ones(x_obs_hist.shape, dtype=bool)
    i_sim_hist = np.ones(x_sim_hist.shape, dtype=bool)
    i_sim_fut = np.ones(x_sim_fut.shape, dtype=bool)
    if lower:
        i_obs_hist = np.logical_and(i_obs_hist, x_obs_hist > lower_threshold)
        i_sim_hist = np.logical_and(i_sim_hist, x_sim_hist > lower_threshold)
        i_sim_fut = np.logical_and(i_sim_fut, x_sim_fut > lower_threshold)
    if upper:
        i_obs_hist = np.logical_and(i_obs_hist, x_obs_hist < upper_threshold)
        i_sim_hist = np.logical_and(i_sim_hist, x_sim_hist < upper_threshold)
        i_sim_fut = np.logical_and(i_sim_fut, x_sim_fut < upper_threshold)
    if unconditional_ccs_transfer:
        # use all values
        x_target = uf.map_quantiles_non_parametric_trend_preserving(
            x_obs_hist, x_sim_hist, x_sim_fut,
            trend_preservation, n_quantiles,
            max_change_factor, max_adjustment_factor,
            True, lower_bound, upper_bound)
    else:
        # use only values within thresholds
        x_target = x_obs_hist.copy()
        x_target[i_obs_hist] = uf.map_quantiles_non_parametric_trend_preserving(
            x_obs_hist[i_obs_hist], x_sim_hist[i_sim_hist],
            x_sim_fut[i_sim_fut], trend_preservation, n_quantiles,
            max_change_factor, max_adjustment_factor,
            True, lower_threshold, upper_threshold)

    # determine extreme value probabilities of future obs
    if lower:
        p_lower = lambda x : np.mean(x <= lower_threshold)
        p_lower_target = p_lower(x_obs_hist) \
            if trendless_bound_frequency else uf.ccs_transfer_sim2obs(
            p_lower(x_obs_hist), p_lower(x_sim_hist), p_lower(x_sim_fut))
    if upper:
        p_upper = lambda x : np.mean(x >= upper_threshold)
        p_upper_target = p_upper(x_obs_hist) \
            if trendless_bound_frequency else uf.ccs_transfer_sim2obs(
            p_upper(x_obs_hist), p_upper(x_sim_hist), p_upper(x_sim_fut))
    if lower and upper:
        p_lower_or_upper_target = p_lower_target + p_upper_target
        if p_lower_or_upper_target > 1 + 1e-10:
            msg = 'sum of p_lower_target and p_upper_target exceeds one'
            warnings.warn(msg)
            p_lower_target /= p_lower_or_upper_target
            p_upper_target /= p_lower_or_upper_target

    # do a parametric quantile mapping of the values within thresholds
    x_source = x_sim_fut
    y = x_source.copy()

    # determine indices of values to be mapped
    i_source = np.ones(x_source.shape, dtype=bool)
    i_target = np.ones(x_target.shape, dtype=bool)
    if lower:
        # make sure that lower_threshold_source < x_source 
        # because otherwise sps.beta.ppf does not work
        lower_threshold_source = \
            uf.percentile1d(x_source, np.array([p_lower_target]))[0] \
            if p_lower_target > 0 else lower_bound if not upper else \
            lower_bound - 1e-10 * (upper_bound - lower_bound)
        i_lower = x_source <= lower_threshold_source
        i_source = np.logical_and(i_source, np.logical_not(i_lower))
        i_target = np.logical_and(i_target, x_target > lower_threshold)
        y[i_lower] = lower_bound
    if upper:
        # make sure that x_source < upper_threshold_source
        # because otherwise sps.beta.ppf does not work
        upper_threshold_source = \
            uf.percentile1d(x_source, np.array([1.-p_upper_target]))[0] \
            if p_upper_target > 0 else upper_bound if not lower else \
            upper_bound + 1e-10 * (upper_bound - lower_bound)
        i_upper = x_source >= upper_threshold_source
        i_source = np.logical_and(i_source, np.logical_not(i_upper))
        i_target = np.logical_and(i_target, x_target < upper_threshold)
        y[i_upper] = upper_bound

    # map quantiles
    while np.any(i_source):
        # break here if target distributions cannot be determined
        if not np.any(i_target):
            msg = 'unable to do any quantile mapping' \
                + ': leaving %i value(s) unadjusted'%np.sum(i_source)
            warnings.warn(msg)
            break

        # use the within-threshold values of x_sim_fut for the source
        # distribution fitting
        x_source_fit = x_source[i_sim_fut]
        x_target_fit = x_target[i_target]

        # determine distribution parameters
        spsdotwhat = sps.norm if distribution == 'normal' else \
                     sps.weibull_min if distribution == 'weibull' else \
                     sps.gamma if distribution == 'gamma' else \
                     sps.beta if distribution == 'beta' else \
                     sps.rice if distribution == 'rice' else \
                     None
        if spsdotwhat is None:
            # prepare non-parametric quantile mapping
            x_source_map = x_source[i_source]
            shape_loc_scale_source = None
            shape_loc_scale_target = None
        else:
            # prepare parametric quantile mapping
            if lower or upper:
                # map the values in x_source to be quantile-mapped such that
                # their empirical distribution matches the empirical
                # distribution of the within-threshold values of x_sim_fut
                x_source_map = uf.map_quantiles_non_parametric_brute_force(
                    x_source[i_source], x_source_fit)
            else:
                x_source_map = x_source

            # fix location and scale parameters for fitting
            floc = lower_threshold if lower else None
            fscale = upper_threshold - lower_threshold \
                if lower and upper else None
    
            # because sps.rice.fit and sps.weibull_min.fit cannot handle
            # fscale=None
            if distribution in ['rice', 'weibull']:
                fwords = {'floc': floc}
            else:
                fwords = {'floc': floc, 'fscale': fscale}
    
            # fit distributions to x_source and x_target
            shape_loc_scale_source = uf.fit(spsdotwhat, x_source_fit, fwords)
            shape_loc_scale_target = uf.fit(spsdotwhat, x_target_fit, fwords)

        # do non-parametric quantile mapping if fitting failed
        if shape_loc_scale_source is None or shape_loc_scale_target is None:
            msg = 'unable to do parametric quantile mapping' \
                + ': doing non-parametric quantile mapping instead'
            if spsdotwhat is not None: warnings.warn(msg)
            p_zeroone = np.linspace(0., 1., n_quantiles + 1)
            q_source_fit = uf.percentile1d(x_source_map, p_zeroone)
            q_target_fit = uf.percentile1d(x_target_fit, p_zeroone)
            y[i_source] = \
                uf.map_quantiles_non_parametric_with_constant_extrapolation(
                x_source_map, q_source_fit, q_target_fit)
            break

        # compute source p-values
        limit_p_values = lambda p : np.maximum(p_value_eps,
                                    np.minimum(1-p_value_eps, p))
        p_source = limit_p_values(spsdotwhat.cdf(
                   x_source_map, *shape_loc_scale_source))

        # compute target p-values
        if adjust_p_values:
            x_obs_hist_fit = x_obs_hist[i_obs_hist]
            x_sim_hist_fit = x_sim_hist[i_sim_hist]
            shape_loc_scale_obs_hist = uf.fit(spsdotwhat,
                                       x_obs_hist_fit, fwords)
            shape_loc_scale_sim_hist = uf.fit(spsdotwhat,
                                       x_sim_hist_fit, fwords)
            if shape_loc_scale_obs_hist is None \
            or shape_loc_scale_sim_hist is None:
                msg = 'unable to adjust p-values: leaving them unadjusted'
                warnings.warn(msg)
                p_target = p_source
            else:
                p_obs_hist = limit_p_values(spsdotwhat.cdf(
                             x_obs_hist_fit, *shape_loc_scale_obs_hist))
                p_sim_hist = limit_p_values(spsdotwhat.cdf(
                             x_sim_hist_fit, *shape_loc_scale_sim_hist))
                p_target = limit_p_values(uf.transfer_odds_ratio(
                           p_obs_hist, p_sim_hist, p_source))
        else:
            p_target = p_source

        # map quantiles
        y[i_source] = spsdotwhat.ppf(p_target, *shape_loc_scale_target)
        break

    return y



def adjust_bias_one_month(
        data, years, long_term_mean,
        lower_bound=[None], lower_threshold=[None],
        upper_bound=[None], upper_threshold=[None],
        unconditional_ccs_transfer=[False], trendless_bound_frequency=[False],
        randomization_seed=None, detrend=[False], rotation_matrices=[],
        n_quantiles=50, distribution=[None],
        trend_preservation=['additive'], adjust_p_values=[False],
        invalid_value_warnings=False, **kwargs):
    """
    1. Replaces invalid values in time series.
    2. Detrends time series if desired.
    3. Replaces values beyond thresholds by random numbers.
    4. Adjusts inter-variable copula.
    5. Adjusts marginal distributions for every variable.
    6. Replaces values beyond thresholds by the respective bound.
    7. Restores trends.

    Parameters
    ----------
    data : dict of str : list of arrays
        Keys : 'obs_hist', 'sim_hist', 'sim_fut'.
        Values : time series for all climate variables.
    years : dict of str : array
        Keys : 'obs_hist', 'sim_hist', 'sim_fut'.
        Values : years of time steps of time series, used for detrending.
    long_term_mean : dict of str: list of floats
        Keys : 'obs_hist', 'sim_hist', 'sim_fut'.
        Values : average of valid values in complete time series.
    lower_bound : list of floats, optional
        Lower bounds of values in data.
    lower_threshold : list of floats, optional
        Lower thresholds of values in data.
        All values below this threshold are replaced by random numbers between
        lower_bound and lower_threshold before bias adjustment.
    upper_bound : list of floats, optional
        Upper bounds of values in data.
    upper_threshold : list of floats, optional
        Upper thresholds of values in data.
        All values above this threshold are replaced by random numbers between
        upper_threshold and upper_bound before bias adjustment.
    unconditional_ccs_transfer : boolean, optional
        Transfer climate change signal using all values, not only those within
        thresholds.
    trendless_bound_frequency : boolean, optional
        Do not allow for trends in relative frequencies of values below lower
        threshold and above upper threshold.
    randomization_seed : int, optional
        Used to seed the random number generator before replacing invalid
        values and values beyond the specified thresholds.
    detrend : list of booleans, optional
        Detrend time series before bias adjustment and put trend back in
        afterwards.
    rotation_matrices : list of (n,n) ndarrays, optional
        List of orthogonal matrices defining a sequence of rotations in variable
        space, where n is the number of variables.
    n_quantiles : int, optional
        Number of quantile-quantile pairs used for non-parametric quantile
        mapping.
    distribution : list of strs, optional
        Kind of distribution used for parametric quantile mapping:
        [None, 'normal', 'weibull', 'gamma', 'beta', 'rice'].
    trend_preservation : list of strs, optional
        Kind of trend preservation used for non-parametric quantile mapping:
        ['additive', 'multiplicative', 'mixed', 'bounded'].
    adjust_p_values : list of booleans, optional
        Adjust p-values for a perfect match in the reference period.
    invalid_value_warnings : boolean, optional
        Raise user warnings when invalid values are replaced bafore bias
        adjustment.

    Returns
    -------
    x_sim_fut_ba : list of arrays
        Result of bias adjustment.

    Other Parameters
    ----------------
    **kwargs : Passed on to map_quantiles_parametric_trend_preserving.
    
    """
    # remove invalid values from masked arrays and store resulting numpy arrays
    x = {}
    for key, data_list in data.items():
        x[key] = [uf.sample_invalid_values(d, randomization_seed,
            long_term_mean[key][i], invalid_value_warnings)[0]
            for i, d in enumerate(data_list)]

    n_variables = len(detrend)
    trend_sim_fut = [None] * n_variables
    for key, y in years.items():
        for i in range(n_variables):
            # subtract trend
            if detrend[i]:
                x[key][i], t = uf.subtract_or_add_trend(x[key][i], y)
                if key == 'sim_fut': trend_sim_fut[i] = t
            else:
                x[key][i] = x[key][i].copy()
        
            # randomize censored values
            # use low powers to ensure successful transformations of values
            # beyond thresholds to values within thresholds during quantile
            # mapping
            uf.randomize_censored_values(x[key][i], 
                lower_bound[i], lower_threshold[i],
                upper_bound[i], upper_threshold[i],
                True, False, randomization_seed, 1., 1.)

    # use MBCn to adjust copula
    if n_variables > 1 and len(rotation_matrices):
        x['sim_fut'] = uf.adjust_copula_mbcn(x, rotation_matrices, n_quantiles)

    x_sim_fut_ba = []
    for i in range(n_variables):
        # adjust distribution and de-randomize censored values
        y = map_quantiles_parametric_trend_preserving(
            x['obs_hist'][i], x['sim_hist'][i], x['sim_fut'][i],
            distribution[i], trend_preservation[i],
            adjust_p_values[i],
            lower_bound[i], lower_threshold[i],
            upper_bound[i], upper_threshold[i],
            unconditional_ccs_transfer[i], trendless_bound_frequency[i],
            n_quantiles, **kwargs)
    
        # add trend
        if detrend[i]:
            y = uf.subtract_or_add_trend(y, years['sim_fut'], trend_sim_fut[i])
    
        # make sure there are no invalid values
        uf.assert_no_infs_or_nans(x['sim_fut'][i], y)
        x_sim_fut_ba.append(y)

    return x_sim_fut_ba



def adjust_bias_one_location(
        i_loc, variable, data_obs_hist, data_sim_hist, data_sim_fut,
        step_size=0, window_centers=None,
        months=[1,2,3,4,5,6,7,8,9,10,11,12],
        halfwin_upper_bound_climatology=[0],
        lower_bound=[None], lower_threshold=[None],
        upper_bound=[None], upper_threshold=[None],
        if_all_invalid_use=[np.nan], **kwargs):
    """
    Adjusts biases in climate data representing one grid cell calendar month by
    calendar month and returns result as arrays.

    Parameters
    ----------
    i_loc : tuple
        Location index.
    variable : list of strs
        Names of variable to be bias-adjusted.
    data_obs_hist : list of arrays
        Historical observation data for each variable.
    data_sim_hist : list of arrays
        Historical simulation data for each variable.
    data_sim_fut : list of arrays
        Future simulation data for each variable.
    step_size: int, optional
        Step size in number of days used for bias adjustment in running-window
        mode. Setting this to 0 implies that bias adjustment is not done in 
        this mode but calendar month by calendar month.
    window_centers : array, optional
        Window centers for bias adjustment in running-window mode. In
        day-of-year units.
    months : list of ints, optional
        List of ints from {1,...,12} representing calendar months for which 
        results of bias adjustment are to be returned. Not used if bias 
        adjustment is done in running-window mode.
    halfwin_upper_bound_climatology : list of ints, optional
        Determines the lengths of running windows used in the calculations of
        climatologies of upper bounds that are used to scale values of obs_hist,
        sim_hist, and sim_fut to the interval [0,1] before bias adjustment. The
        window length is set to halfwin_upper_bound_climatology * 2 + 1 time
        steps. If halfwin_upper_bound_climatology == 0 then no rescaling is
        done.
    lower_bound : list of floats, optional
        Lower bounds of values in data.
    lower_threshold : list of floats, optional
        Lower thresholds of values in data.
    upper_bound : list of floats, optional
        Upper bounds of values in data.
    upper_threshold : list of floats, optional
        Upper thresholds of values in data.
    if_all_invalid_use : list of floats, optional
        Used to replace invalid values if there are no valid values.

    Returns
    -------
    tuple
        (i_loc, result) where result is list of bias-adjusted arrays.

    Other Parameters
    ----------------
    **kwargs : Passed on to adjust_bias_one_month.

    """
    # get local input data
    data = {}
    for key in ['obs_hist', 'sim_hist', 'sim_fut']:
        data[key] = []
        for i in range(len(variable)):
            d = eval(f'data_{key}')[i]
            # assume d.shape = spatial_shape + (time,), i_loc indexes the spatial dims
            slicer = tuple(i_loc) + (slice(None),)
            data[key].append(d[slicer])

    # abort here if there are only missing values in at least one dataset
    if uf.only_missing_values_in_at_least_one_dataset(data):
        print(i_loc, 'skipped due to missing data')
        return None

    # otherwise continue
    # print(i_loc) # Uncomment this to track progress of bias adjustment at different locations. Note that this can produce a lot of output if there are many locations.
    n_variables = len(variable)
    None_list = [None] * n_variables
    result = [d.data.copy() if isinstance(d, np.ma.MaskedArray) else d.copy()
        for d in data['sim_fut']]
    
    # scale to values in [0, 1]
    ubc = {}
    ubc_days = {}
    ubc_result = None_list.copy()
    msg = 'found nans in upper bound climatology for variable'
    for i, halfwin in enumerate(halfwin_upper_bound_climatology):
        if halfwin:
            # scale obs_hist, sim_hist, sim_fut
            for key, data_list in data.items():
                ubc[key], ubc_days[key] = uf.get_upper_bound_climatology(
                    data_list[i], days[key], halfwin)
                assert not np.any(np.isnan(ubc[key])), f'{msg} {i} in {key}'
                uf.scale_by_upper_bound_climatology(data_list[i],
                    ubc[key], days[key], ubc_days[key], divide=True)
    
            # prepare scaling of result
            ubc_result[i] = uf.ccs_transfer_sim2obs_upper_bound_climatology(
                ubc['obs_hist'], ubc['sim_hist'], ubc['sim_fut'])

    # compute mean value over all time steps for invalid value sampling
    long_term_mean = {}
    for key, data_list in data.items():
        long_term_mean[key] = [uf.average_valid_values(d, if_all_invalid_use[i],
            lower_bound[i], lower_threshold[i],
            upper_bound[i], upper_threshold[i])
            for i, d in enumerate(data_list)]

    # do local bias adjustment
    if step_size:
        # do bias adjustment in running-window mode
        data_this_window = {
        'obs_hist': None_list.copy(),
        'sim_hist': None_list.copy(),
        'sim_fut': None_list.copy()
        }
        years_this_window = {}
        for window_center in window_centers:
            # extract data for 31-day wide window around window_center
            for key, data_list in data.items():
                m = uf.window_indices_for_running_bias_adjustment(
                    days[key], window_center, 31)
                years_this_window[key] = years[key][m]
                for i in range(n_variables):
                    data_this_window[key][i] = data_list[i][m]
    
            # adjust biases and store result as list of masked arrays
            result_this_window = adjust_bias_one_month(
                data_this_window, years_this_window, long_term_mean,
                lower_bound, lower_threshold,
                upper_bound, upper_threshold, **kwargs)
    
            # put central part of bias-adjusted data into result
            m_ba = uf.window_indices_for_running_bias_adjustment(
                days['sim_fut'], window_center, 31)
            m_keep = uf.window_indices_for_running_bias_adjustment(
                days['sim_fut'], window_center, step_size, years['sim_fut'])
            m_ba_keep = np.in1d(m_ba, m_keep)
            for i, halfwin in enumerate(halfwin_upper_bound_climatology):
                # scale from values in [0, 1]
                if halfwin:
                   uf.scale_by_upper_bound_climatology(
                       result_this_window[i], ubc_result[i],
                       days['sim_fut'][m_ba], ubc_days['sim_fut'], divide=False)
    
                result[i][m_keep] = result_this_window[i][m_ba_keep]
    else:
        # do bias adjustment calendar month by calendar month
        data_this_month = {
        'obs_hist': None_list.copy(),
        'sim_hist': None_list.copy(),
        'sim_fut': None_list.copy()
        }
        years_this_month = {}
        for month in months:
            # extract data
            for key, data_list in data.items():
                m = month_numbers[key] == month
                assert np.any(m), f'no data found for month {month} in {key}'
                y = years[key]
                years_this_month[key] = None if y is None else y[m]
                for i in range(n_variables):
                    data_this_month[key][i] = data_list[i][m]
    
            # adjust biases and store result as list of masked arrays
            result_this_month = adjust_bias_one_month(
                data_this_month, years_this_month, long_term_mean,
                lower_bound, lower_threshold,
                upper_bound, upper_threshold, **kwargs)
    
            # put bias-adjusted data into result
            m = month_numbers['sim_fut'] == month
            for i, halfwin in enumerate(halfwin_upper_bound_climatology):
                # scale from values in [0, 1]
                if halfwin:
                   uf.scale_by_upper_bound_climatology(
                       result_this_month[i], ubc_result[i],
                       days['sim_fut'][m], ubc_days['sim_fut'], divide=False)
    
                result[i][m] = result_this_month[i]
    
    return (i_loc, result)


def adjust_bias(
        obs_hist_arrays, sim_hist_arrays, sim_fut_arrays,
        space_shape, **kwargs):
    """
    Adjusts biases grid cell by grid cell and returns results.

    Parameters
    ----------
    obs_hist_arrays : list of arrays
        Historical observation data for each variable.
    sim_hist_arrays : list of arrays
        Historical simulation data for each variable.
    sim_fut_arrays : list of arrays
        Future simulation data for each variable.
    space_shape : tuple
        Describes the spatial dimensions of the climate data.

    Returns
    -------
    bias_adjusted_data : dict
        Dictionary with location indices as keys and bias-adjusted data as values.

    Other Parameters
    ----------------
    **kwargs : Passed on to adjust_bias_one_location.

    """
    # Extract variable from kwargs
    variable = kwargs.pop('variable')
    
    # adjust every location individually
    i_locations = list(np.ndindex(space_shape))
    abol = partial(adjust_bias_one_location, variable=variable, data_obs_hist=obs_hist_arrays, data_sim_hist=sim_hist_arrays, data_sim_fut=sim_fut_arrays, **kwargs)
    results = list(map(abol, i_locations))
    
    # organize results by location
    bias_adjusted_data = {}
    for location, data in results:
        if location is not None:
            bias_adjusted_data[location] = data
    
    return bias_adjusted_data



def main(obs_hist_arrays, sim_hist_arrays, sim_fut_arrays, **kwargs):
    """
    Prepares and executes the bias adjustment algorithm with input arrays.

    This function works with timeseries of any length, as long as all input
    arrays have consistent shapes and the metadata arrays match the time dimension.

    Parameters
    ----------
    obs_hist_arrays : list of arrays
        Historical observation data for each variable. Shape: (lat, lon, time)
    sim_hist_arrays : list of arrays
        Historical simulation data for each variable. Shape: (lat, lon, time)
    sim_fut_arrays : list of arrays
        Future simulation data for each variable. Shape: (lat, lon, time)
    **kwargs : Additional parameters
        variable : list of strs
            Names of variables.
        years : dict
            Years for each dataset. Keys: 'obs_hist', 'sim_hist', 'sim_fut'
            Values: arrays with length matching the time dimension
        days : dict
            Days of year for each dataset. Keys: 'obs_hist', 'sim_hist', 'sim_fut'
            Values: arrays with length matching the time dimension
        month_numbers : dict
            Month numbers for each dataset. Keys: 'obs_hist', 'sim_hist', 'sim_fut'
            Values: arrays with length matching the time dimension
        And other bias correction options.

    Returns
    -------
    bias_adjusted_data : dict
        Dictionary with location indices as keys and bias-adjusted data as values.
        Each value is a list of arrays (one per variable) with shape (time,).

    Raises
    ------
    ValueError
        If input arrays have inconsistent shapes or metadata lengths don't match.
    """
    # Extract parameters from kwargs
    variable = kwargs.pop('variable')
    years_local = kwargs.pop('years')
    days_local = kwargs.pop('days')
    month_numbers_local = kwargs.pop('month_numbers')
    
    # Validate input consistency
    n_variables = len(variable)
    if len(obs_hist_arrays) != n_variables or len(sim_hist_arrays) != n_variables or len(sim_fut_arrays) != n_variables:
        raise ValueError(f"Number of arrays must match number of variables ({n_variables})")
    
    # Check that all arrays have consistent shapes
    expected_shape = obs_hist_arrays[0].shape
    time_length = expected_shape[-1]  # Last dimension is time
    
    for i, (obs, sim_hist, sim_fut) in enumerate(zip(obs_hist_arrays, sim_hist_arrays, sim_fut_arrays)):
        if obs.shape != expected_shape:
            raise ValueError(f"obs_hist array {i} has shape {obs.shape}, expected {expected_shape}")
        if sim_hist.shape != expected_shape:
            raise ValueError(f"sim_hist array {i} has shape {sim_hist.shape}, expected {expected_shape}")
        if sim_fut.shape != expected_shape:
            raise ValueError(f"sim_fut array {i} has shape {sim_fut.shape}, expected {expected_shape}")
    
    # Check metadata consistency
    for key in ['obs_hist', 'sim_hist', 'sim_fut']:
        if len(month_numbers_local[key]) != time_length:
            raise ValueError(f"{key} month_numbers has length {len(month_numbers_local[key])}, expected {time_length}")
        if len(years_local[key]) != time_length:
            raise ValueError(f"{key} years has length {len(years_local[key])}, expected {time_length}")
        if len(days_local[key]) != time_length:
            raise ValueError(f"{key} days has length {len(days_local[key])}, expected {time_length}")
    
    # Set global variables for compatibility with existing functions
    global month_numbers, years, days
    month_numbers = month_numbers_local
    years = years_local
    days = days_local

    space_shape = obs_hist_arrays[0].shape[:-1]  # assume (lat, lon, time)

    # do bias adjustment and get results
    spatial_dimensions_str = f"{space_shape[0]}x{space_shape[1]}"
    print(f'adjusting at locations in {spatial_dimensions_str} ...')
    bias_adjusted_data = adjust_bias(
        obs_hist_arrays, sim_hist_arrays, sim_fut_arrays,
        space_shape, variable=variable, **kwargs)
    
    print(f'bias adjustment completed. Returning {len(bias_adjusted_data)} locations of data.')
    return bias_adjusted_data


if __name__ == '__main__':
    print("This script is now designed for notebook use. Use main() function with arrays.")
