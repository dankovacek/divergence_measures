import os
from time import time
import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.stats import entropy, wasserstein_distance, linregress

from shapely.geometry import Point

import multiprocessing as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
STREAMFLOW_DIR = os.path.join(DATA_DIR, 'hysets_streamflow_timeseries')

hs_properties_path = os.path.join(DATA_DIR, 'BCUB_HYSETS_properties_with_climate.csv')
hs_df = pd.read_csv(hs_properties_path)
hs_df['geometry'] = hs_df.apply(lambda x: Point(x['centroid_lon_deg_e'], x['centroid_lat_deg_n']), axis=1)

class Station:
    """
    A class used to represent a Station.

    The Station class initializes an object with attributes based on a provided dictionary of station information. 
    Each key-value pair in the dictionary is unpacked and set as an attribute of the Station object.

    Attributes
    ----------
    id : str
        The official ID of the station, derived from the 'Official_ID' key in the provided dictionary.
    
    Methods
    -------
    __init__(self, station_info) -> None
        Initializes the Station object by unpacking a dictionary of station information into attributes.
    """
    def __init__(self, station_info, bitrate) -> None:
        """
        Initializes the Station object.

        Parameters
        ----------
        station_info : dict
            A dictionary containing information about the station. Each key-value pair in the dictionary 
            will be set as an attribute of the Station object. The 'Official_ID' key in the dictionary will 
            be used to set the 'id' attribute of the Station object.
        """
        # the input station_info is a dict,
        # unpack the dict into attributes
        for k, v in station_info.items():
            setattr(self, k, v)


        self.id = self.official_id
        self.sim_label = f'{self.id}_sim'
        self.obs_label = f'{self.id}'
        self.sim_log_label = f'{self.id}_sim_log10'
        self.obs_log_label = f'{self.id}_log10'
        # self.UR_label = f'{self.id}_UR'
        # self.logUR_label = f'{self.id}_logUR'
        # self.log_sim_label = f'{self.id}_sim_log10'
        self.sim_quantized_label = f'sim_quantized_{self.id}_{bitrate}b'
        self.obs_quantized_label = f'obs_quantized_{self.id}_{bitrate}b'
        # self.quantized_label_ = f'quantized_{self.id}_sim_{bitrate}b'


        
def check_processed_results(out_fpath):
    """
    Checks for existing processed results at the specified file path and loads them into a DataFrame if available.

    Parameters
    ----------
    out_fpath : str
        The file path where the processed results are expected to be found.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the loaded results if the file exists, or an empty DataFrame if the file does not exist.

    Notes
    -----
    - The function checks if a file exists at the specified path.
    - If the file exists, it is read into a DataFrame with specified data types for the 'proxy' and 'target' columns.
    - If the file does not exist, an empty DataFrame is returned.
    - The function prints a message indicating whether existing results were loaded or no existing results were found.

    Example
    -------
    >>> out_fpath = 'path/to/results.csv'
    >>> results_df = check_processed_results(out_fpath)
    >>> if not results_df.empty:
    ...     print("Results loaded successfully.")
    ... else:
    ...     print("No existing results found.")
    """    
    dtype_spec = {'proxy': str, 'target': str}
    if os.path.exists(out_fpath):
        results_df = pd.read_csv(out_fpath, dtype=dtype_spec)
        print(f'    Loaded {len(results_df)} existing results')
        return results_df
    else:
        print('    No existing results found')
        return pd.DataFrame()
    

def filter_processed_pairs(results_df, id_pairs):
    """
    Filters out pairs of IDs that have already been processed, based on a DataFrame of existing results.

    Parameters
    ----------
    results_df : pd.DataFrame
        A pandas DataFrame containing the processed results, with columns 'proxy' and 'target'.
    id_pairs : list of tuples
        A list of tuples, where each tuple contains a pair of IDs (proxy, target) to be checked against the existing results.

    Returns
    -------
    list of tuples
        A list of tuples containing the pairs of IDs (proxy, target) that have not yet been processed.

    Notes
    -----
    - The function converts the list of ID pairs into a DataFrame for easy merging with the results DataFrame.
    - It performs an outer merge and keeps only those rows that are not found in the results DataFrame, indicating unprocessed pairs.
    - The filtered DataFrame is then converted back to a list of tuples.

    Example
    -------
    >>> results_df = pd.DataFrame({
    ...     'proxy': ['A', 'B'],
    ...     'target': ['X', 'Y']
    ... })
    >>> id_pairs = [('A', 'X'), ('B', 'Z'), ('C', 'Y')]
    >>> remaining_pairs = filter_processed_pairs(results_df, id_pairs)
    >>> print(remaining_pairs)
    [('B', 'Z'), ('C', 'Y')]
    """
    # Convert list of pairs to DataFrame for easy merging
    id_pairs_df = pd.DataFrame(id_pairs, columns=['proxy', 'target'])
    
    # Perform an outer merge and keep only those rows that are NaN in results_df index
    # This indicates that these rows were not found in results_df
    merged_df = id_pairs_df.merge(results_df, on=['proxy', 'target'], how='left', indicator=True)
    filtered_df = merged_df[merged_df['_merge'] == 'left_only']
    
    # Convert the filtered DataFrame back to a list of tuples
    remaining_pairs = list(zip(filtered_df['proxy'], filtered_df['target']))
    
    return remaining_pairs

    
    
def get_timeseries_data(official_id, min_flow=1e-4):
    """
    Imports streamflow data for a given station ID, processes it to handle low flow values, 
    and returns the processed DataFrame.

    Parameters
    ----------
    official_id : str
        The official station ID for the streamflow data.
    min_flow : float, optional
        The minimum flow value to consider. Flows below this value are flagged and set to `min_flow`.
        Default is 1e-4.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the processed streamflow data with the following modifications:
        - The 'discharge' column is renamed to the station ID (`official_id`).
        - A new column named '{official_id}_low_flow_flag' is added, indicating where the discharge 
          was below the `min_flow`.
        - Discharge values below `min_flow` are clipped to `min_flow`.

    Notes
    -----
    - The function reads the streamflow data from a CSV file located in the directory specified by `STREAMFLOW_DIR`.
    - The CSV file is expected to be named as '{official_id}.csv' and should contain a column named 'discharge'.

    Example
    -------
    >>> df = import_streamflow('05010500')
    >>> df.head()
         date   12345678  12345678_low_flow_flag
    0  2020-01-01  0.0010                     False
    1  2020-01-02  0.0001                      True
    2  2020-01-03  0.0001                      True
    3  2020-01-04  0.0025                     False
    4  2020-01-05  0.0001                      True
    """
    fpath = os.path.join(STREAMFLOW_DIR, f'{official_id}.csv')
    df = pd.read_csv(fpath, parse_dates=['time'])
    df[f'{official_id}_low_flow_flag'] = df['discharge'] < min_flow    
    # assign a small flow value instead of zero
    df['discharge'] = df['discharge'].clip(lower=min_flow)
    # rename the discharge column to the station id    
    df.rename(columns={'discharge': official_id}, inplace=True)
    
    
    return df


def retrieve_nonconcurrent_data(proxy, target):
    """
    Retrieves and merges non-concurrent time series data for two stations, `proxy` and `target`, 
    and returns a combined DataFrame.

    Parameters
    ----------
    proxy : str
        The station ID for the proxy/donor time series data. The proxy is the regional 
        station that is used as a model for an ungauged location
    target : str
        The station ID for the target time series data.  The target is the location 
        where runoff is simulated.

        Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the combined time series data for both `proxy` and `target` stations. 
        The DataFrame is indexed by time and contains the data for both stations, with missing values 
        filled as NaN where there is no overlap in time. Additionally, a 'year' column is included 
        that extracts the year from the time index.

    Notes
    -----
    - The function uses `get_timeseries_data` to retrieve the time series data for each station.
    - The time series data for each station is expected to have a 'time' column, which will be set as the index.
    - The function performs an outer join to ensure that all time points from both series are included.
    - The combined DataFrame will have NaN values for non-concurrent time points where data is missing for one station.
    - The 'year' column is extracted from the time index of the combined DataFrame.

    Example
    -------
    >>> df = retrieve_nonconcurrent_data('stationA', 'stationB')
    >>> df.head()
                        stationA   stationB  year
    time                                    
    2020-01-01 00:00:00      0.1        NaN  2020
    2020-01-01 01:00:00      0.2        0.4  2020
    2020-01-01 02:00:00      NaN        0.5  2020
    2020-01-01 03:00:00      0.3        0.6  2020
    2020-01-01 04:00:00      0.4        NaN  2020
    """
    df1 = get_timeseries_data(proxy).set_index('time')
    df2 = get_timeseries_data(target).set_index('time')
    df = pd.concat([df1, df2], join='outer', axis=1)
    df['year'] = df.index.year
    return df


def transform_and_jitter(df, station):
    """
    Adds uniform noise to the data of a specified station, checks for small values, and applies a log transformation.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the station data to be transformed.
    station : object
        An object representing the station with the following attributes:
        - id: str
            The column name of the station data in the DataFrame.
        - obs_log_label: str
            The column name for the log-transformed station data to be created.

    Returns
    -------
    pd.DataFrame
        The DataFrame with added noise to the station data and an additional column with the log-transformed data.

    Notes
    -----
    - The function adds uniform noise in the range [-1e-9, 1e-9] to the data of the specified station to avoid issues with zero values.
    - If the minimum value of the data after noise addition is less than 1e-5, a warning message is printed.
    - The function creates a new column in the DataFrame with log10-transformed data of the specified station.

    Example
    -------
    >>> class Station:
    ...     def __init__(self, id, obs_log_label):
    ...         self.id = id
    ...         self.obs_log_label = obs_log_label
    ...
    >>> station = Station(id='stationA', obs_log_label='stationA_log')
    >>> df = pd.DataFrame({'stationA': [0.1, 0.01, 0.001, 0.0001]})
    >>> df = transform_and_jitter(df, station)
    >>> df
       stationA  stationA_log
    0  0.100000     -1.000000
    1  0.010000     -2.000000
    2  0.001000     -3.000000
    3  0.000100     -4.000000
    """
    depth = 1e-9
    noise = np.random.uniform(-depth, depth, size=len(df))
    df[station.id] += noise

    if df[station.id].min() < 1e-5:
        print(df[station.id].min())
        msg = f'Noise addition creates values < 1e-5 (ID: {station.id})'
        print(msg)
    
    # log transform the data
    df[station.obs_log_label] = np.log10(df[station.id])
    return df


def compute_distance(stn1, stn2):
    """
    Computes the distance in kilometers between two stations based on their geographic coordinates.

    Parameters
    ----------
    stn1 : dict
        A dictionary containing the geographic coordinates of the first station with keys:
        - 'Centroid_Lon_deg_E': float
            The longitude of the station in degrees East.
        - 'Centroid_Lat_deg_N': float
            The latitude of the station in degrees North.
    stn2 : dict
        A dictionary containing the geographic coordinates of the second station with keys:
        - 'Centroid_Lon_deg_E': float
            The longitude of the station in degrees East.
        - 'Centroid_Lat_deg_N': float
            The latitude of the station in degrees North.

    Returns
    -------
    float
        The distance between the two stations in kilometers.

    Notes
    -----
    - The function uses the geodesic coordinates of the stations to create Point geometries.
    - The points are then transformed from the WGS 84 coordinate system (EPSG:4326) to the NAD83 / BC Albers coordinate system (EPSG:3005).
    - The distance is computed in the projected coordinate system (EPSG:3005) and converted to kilometers.

    Example
    -------
    >>> stn1 = {'Centroid_Lon_deg_E': -123.3656, 'Centroid_Lat_deg_N': 48.4284}
    >>> stn2 = {'Centroid_Lon_deg_E': -123.3000, 'Centroid_Lat_deg_N': 48.4500}
    >>> distance = compute_distance(stn1, stn2)
    >>> print(f"Distance: {distance:.2f} km")
    Distance: 4.91 km
    """
    p1 = Point(stn1['Centroid_Lon_deg_E'], stn1['Centroid_Lat_deg_N'])
    p2 = Point(stn2['Centroid_Lon_deg_E'], stn2['Centroid_Lat_deg_N'])
    gdf = gpd.GeoDataFrame(geometry=[p1, p2], crs='EPSG:4326')
    gdf = gdf.to_crs('EPSG:3005')
    distance = gdf.distance(gdf.shift()) / 1000
    return distance.values[1]


def compute_observed_series_entropy(row, bitrate):
    """
    Quantizes the observed time series data for a given station to a specified bitrate and computes the entropy of the quantized data.

    Parameters
    ----------
    row : pd.Series
        A pandas Series containing information about the station. It must include the key 'official_id' which specifies the station ID.
    bitrate : int
        The number of bits to use for quantizing the observed series.

    Returns
    -------
    float
        The entropy of the quantized time series data.

    Notes
    -----
    - The function retrieves the time series data for the specified station ID.
    - It drops any rows with missing data for the station.
    - The observed values are quantized into bins based on the specified bitrate.
    - The entropy of the quantized values is computed and returned.

    Example
    -------
    >>> row = pd.Series({'official_id': 'station123'})
    >>> bitrate = 8
    >>> H = quantize_observed_series(row, bitrate)
    >>> print(f'Entropy: {H:.4f}')
    """
    # get data
    stn_id = row['official_id']
    df = get_timeseries_data(stn_id)
    df = df.dropna(subset=[stn_id])
    min_q, max_q = df[stn_id].min(), df[stn_id].max()
    # use equal width bins in log10 space
    log_edges = np.linspace(np.log10(min_q), np.log10(max_q), 2**bitrate)
    linear_edges = [10**e for e in log_edges]
    df[f'{bitrate}_bits_quantized'] = np.digitize(df[stn_id], linear_edges)
    H = entropy(df[f'{bitrate}_bits_quantized'], base=2)
    return H
    
    
def compute_cdf(data):
    """
    Computes the cumulative distribution function (CDF) for a given dataset.

    Parameters
    ----------
    data : array-like
        An array or list of numerical values for which the CDF is to be computed.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing two numpy arrays:
        - The sorted data values.
        - The CDF values corresponding to the sorted data values.

    Notes
    -----
    The CDF values are computed by sorting the data and calculating the cumulative probabilities for each sorted value.

    Example
    -------
    >>> data = [1, 3, 2, 4, 5]
    >>> sorted_data, cdf_values = compute_cdf(data)
    >>> print(sorted_data)
    [1, 2, 3, 4, 5]
    >>> print(cdf_values)
    [0.2, 0.4, 0.6, 0.8, 1.0]
    """
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data)+1) / float(len(sorted_data))
    return sorted_data, yvals


def process_pairwise_comparisons(inputs, bitrate, out_fname, batch_size):
    """
    Processes pairwise comparisons in batches, saving results to CSV files and handling already processed batches.

    Parameters
    ----------
    inputs : list
        A list of input pairs to be processed.
    bitrate : int
        The bitrate to be used for processing.
    out_fname : str
        The base output file name for saving the results.
    batch_size : int
        The number of pairs to process in each batch.

    Returns
    -------
    list
        A list of file paths to the processed batch result files.

    Notes
    -----
    - If the input list is empty, the function prints a message and returns None.
    - The function divides the input pairs into batches and processes each batch separately.
    - Already processed batches are skipped to avoid redundant computation.
    - Results of each batch are saved to a CSV file, and the paths to these files are returned.

    Example
    -------
    >>> inputs = [('A', 'X'), ('B', 'Y'), ('C', 'Z')]
    >>> bitrate = 4
    >>> out_fname = 'results.csv'
    >>> batch_size = 500
    >>> batch_files = process_pairwise_comparisons(inputs, bitrate, out_fname, batch_size)
    >>> print(batch_files)
    ['path/to/results_batch_0.csv', 'path/to/results_batch_1.csv']
    """
    t0 = time()
    if len(inputs) == 0:
        print('    No new pairs to process')
        return None
    
    
    n_batches = max(len(inputs) // batch_size, 1)
    batch_inputs = np.array_split(np.array(inputs, dtype=object), n_batches)

    print(f'    Processing {len(inputs)} pairs in {n_batches} batches at {bitrate} bits')
    batch_no = 0
    batch_files = []

    for batch in batch_inputs[:2]:
        batch_output_fname = out_fname.replace('.csv', f'_batch_{batch_no}.csv')
        batch_output_path = os.path.join(DATA_DIR, 'temp_batches', batch_output_fname)
        if os.path.exists(batch_output_path):
            print(f'    Batch {batch_no} already processed.')
            batch_no += 1
            batch_files.append(batch_output_path)
            continue

        # results = []
        # for b in batch[:2]:
        #     result = process_batch(b)
        #     results.append(result)

        with mp.Pool() as pool:
            results = pool.map(process_batch, batch)
            results = [r for r in results if r is not None]

        if len(results) == 0:
            print('    No new results in current batch to save')
            continue
        
        t1 = time()
        print(f'    Processed batch {batch_no}/{len(batch_inputs)} -- {batch_size} pairs ({len(results)} good results) in {t1 - t0:.1f} seconds')
        
        new_results_df = pd.DataFrame(results)
        new_results_df['target'] = new_results_df['target'].astype(str)
        new_results_df['proxy'] = new_results_df['proxy'].astype(str)

        if len(new_results_df) == 0:
            print('    No new results in current batch to save')
            continue
        else:
            pass
            new_results_df.to_csv(batch_output_path, index=False)
            print(f'    Saved {len(new_results_df)} new results to file.')
            batch_files.append(batch_output_path)
        batch_no += 1
    return batch_files


def check_if_nested(proxy_data, target_data):
    """
    Create an attribute to identify when
    catchments are nested.
    Returns: 
        0: No intersection
        1: The donor/proxy station is downstream of the target
        2: The donor/proxy station is upstream of the target.
    """
    proxy, target = proxy_data['geometry'], target_data['geometry']
    pid = proxy_data['official_id']
    tid = target_data['official_id']
    nested = 0
    a, b, c = proxy.intersects(target), proxy.contains(target), target.contains(proxy)
    
    if proxy.intersects(target):
        # if polygons intersect, check if the smaller one is mostly within the bigger one
        # can't use 'contained' because bounds may not align perfectly
        diff = target.intersection(proxy)
        target_pct_diff = diff.area / target.area
        proxy_pct_diff = diff.area / proxy.area
        nested = 1
        
        if target.area > proxy.area:
            nested = 2
            
        if (proxy_pct_diff < 0.25) & (target_pct_diff < 0.25):
            nested = 0
            # print(pid, tid)
            # print(f'Proxy: {proxy.area/1e6:.1f} km2')
            # print(f'Target: {target.area/1e6:.1f} km2')
            # print(f'    Polygons intersect but may not be nested. target pct diff = {target_pct_diff:.3f} proxy pct diff {proxy_pct_diff:.3f}')
            # raise Exception('Polygons intersect but may not be nested.')    
    
    return nested



def uniform_log_bins(data, proxy, bitrate, epsilon=1e-9):
    """
    Creates uniform bins in log space for quantizing time series data.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the time series data.
    proxy : object
        An object representing the proxy station, with an attribute `obs_label` that specifies the observed data column name.
    bitrate : int
        The number of bits used for quantizing the observed series.
    epsilon : float, optional
        A small value added to the maximum log value to ensure the maximum data point falls within the last bin (default is 1e-9).

    Returns
    -------
    list
        A list of bin edges in linear space for quantizing the data.

    Notes
    -----
    - The function reserves two bins for out-of-range values at the left and right edges.
    - It computes the minimum and maximum log values of the observed data to define the range for binning.
    - Bin edges are evenly spaced in log space and then converted back to linear space.
    - An exception is raised if the minimum and maximum log values are equal.
    - The function ensures that there are `n_bins + 1` bin edges, where `n_bins` is `2**bitrate - 2`.

    Example
    -------
    >>> data = pd.DataFrame({'proxy_obs': np.random.rand(100)})
    >>> proxy = lambda: None
    >>> setattr(proxy, 'obs_label', 'proxy_obs')
    >>> bin_edges = uniform_log_bins(data, proxy, 8)
    >>> print(bin_edges)
    [1e-06, 1.0023672938784246e-06, 1.0047394197324785e-06, ..., 0.9976342115601912, 1.0]

    Raises
    ------
    Exception
        If the minimum and maximum log values are equal.
    """
    # reserve two bins for out of range values at left and right
    n_bins = 2**bitrate - 2
    min_log_val = np.log10(data[proxy.obs_label].min())
    # shift the right edge slightly to ensure the maximum value 
    # falls in the last bin
    max_log_val = np.log10(data[proxy.obs_label].max())
    
    if min_log_val == max_log_val:
        raise Exception('Min. and max. log values should not be equal.')
        # print('   Min and max log values are the same.  Adding small amount of noise.')
        # max_log_val += epsilon

    # set the bin edges to be evenly spaced between the
    # observed range of the proxy/donor series
    # np.digitize will assign 0 for out-of-range values at left
    # and n_bins + 1 for out-of-range values at right
    log_bin_edges = np.linspace(
        min_log_val,
        max_log_val,
        n_bins + 1,
    ).flatten()

    # convert back to linear space
    bin_edges = [10**e for e in log_bin_edges]

    # there should be n_bins edges which define n_bins - 1 bins
    # this is to reserve 2 bin for out-of-range values to the right
    assert len(bin_edges) == n_bins + 1
    
    return bin_edges


def error_adjusted_fractional_bin_counts(values, bin_edges, error_factor=0.1):
    """
    Computes error-adjusted fractional bin counts for a given set of values by considering an error factor.

    Parameters
    ----------
    values : array-like
        An array or list of numerical values for which the fractional bin counts are to be computed.
    bin_edges : np.ndarray
        The edges of the bins used for quantizing the values.
    error_factor : float, optional
        The factor of uniformly distributed error to be considered for each value (default is 0.1).

    Returns
    -------
    np.ndarray
        An array of fractional bin counts adjusted for the specified error factor.

    Notes
    -----
    - The function calculates lower and upper bounds for each value by applying the error factor.
    - It computes the overlap of each value's bounds with each bin and calculates fractional counts based on this overlap.
    - The bin counts are normalized to ensure the sum is equal to the total number of observations.

    Example
    -------
    >>> values = np.random.rand(100)
    >>> bin_edges = np.linspace(0, 1, 11)
    >>> counts = error_adjusted_fractional_bin_counts(values, bin_edges, error_factor=0.05)
    >>> print(counts)
    """
    lower_bounds = values * (1 - error_factor)
    upper_bounds = values * (1 + error_factor)
    
    # Calculate bin midpoints
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initialize the bin counts
    bin_counts = np.zeros(len(bin_midpoints))
    
    # Create a matrix of bin edges
    bin_edges_lower = bin_edges[:-1]
    bin_edges_upper = bin_edges[1:]

    # Compute the overlap for each value with each bin
    try:
        overlap_matrix = np.maximum(0, np.minimum(upper_bounds[:, None], bin_edges_upper) - np.maximum(lower_bounds[:, None], bin_edges_lower))
    except Exception as ex:
        print(bin_edges)

        raise Exception('Error calculating overlap matrix')
    # Compute bin widths
    bin_widths = bin_edges_upper - bin_edges_lower

    # Fractional counts based on the overlap
    fractional_counts = overlap_matrix / bin_widths

    # Sum fractional counts for each bin
    bin_counts = np.sum(fractional_counts, axis=0)

    # Normalize bin counts to ensure the sum is equal to the number of observations
    total_observations = len(values)
    bin_counts *= (total_observations / np.sum(bin_counts))

    return bin_counts


def compute_error_adjusted_probabilities(df, series, bin_edges, error_factor=0.1):
    """
    Computes error-adjusted probabilities for a given series in a DataFrame by adding uniformly distributed error.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time series data.
    series : str
        The column name of the series for which to compute the probabilities.
    bin_edges : np.ndarray
        The edges of the bins used for quantizing the series.
    error_factor : float, optional
        The factor of uniformly distributed error to be added to the observed data (default is 0.1).

    Returns
    -------
    np.ndarray
        An array of error-adjusted frequencies corresponding to each bin.

    Notes
    -----
    - The function first computes the bin numbers for the series values based on the provided bin edges.
    - It then calculates fractional bin counts adjusted for measurement error.
    - The sum of the fractional error counts is checked to ensure it is approximately equal to the sum of the original bin counts.
    - The function returns the error-adjusted frequencies, ensuring that their sum is approximately 1.

    Example
    -------
    >>> data = {'series': np.random.rand(100)}
    >>> df = pd.DataFrame(data)
    >>> bin_edges = np.linspace(0, 1, 11)
    >>> frequencies = compute_error_adjusted_probabilities(df, 'series', bin_edges, error_factor=0.05)
    >>> print(frequencies)
    """
    n_sample = len(df[series])
    # bin_edges = np.array([10**e for e in log_bin_edges])
    bin_nos = np.digitize(df[series], bin_edges)
    unique, bin_counts = np.unique(bin_nos, return_counts=True)
    
    fractional_error_counts = error_adjusted_fractional_bin_counts(df[series], bin_edges, error_factor)
        
    # ensure that the number of observations hasn't changed
    try:
        sum_difference = abs(sum(bin_counts) - sum(fractional_error_counts))
        assert sum_difference < 0.01
    except AssertionError:
        print(f'simple and linear counts are not equal')
    
    error_adjusted_frequencies = fractional_error_counts / n_sample
    
    # frequency_mapping = {k:v for k, v in zip(unique, error_frequencies)}
    
    # Check if the sum of frequencies is within 0.01 of 1
    if (abs(sum(error_adjusted_frequencies) - 1) >= 0.01):
        raise ValueError("Sum of frequencies is not within 0.01 of 1")

    return error_adjusted_frequencies



def compute_probabilities(df, target, bitrate, pseudo_counts, concurrent_data, bin_edges):
    """
    Computes the observed and simulated probabilities for a given target station using quantization and pseudo counts.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time series data for the target station.
    target : object
        An object representing the target station, with attributes `obs_label`, `sim_label`, `obs_quantized_label`, and `sim_quantized_label`.
    bitrate : int
        The number of bits used for quantizing the observed series.
    pseudo_counts : list of float
        A list of pseudo counts to be added to each bin for calculating posterior probabilities.
    concurrent_data : bool
        A flag indicating whether the data is concurrent.
    bin_edges : np.ndarray
        The edges of the bins used for quantizing the data.

    Returns
    -------
    tuple
        A tuple containing:
        - p_obs (np.ndarray): The observed probabilities.
        - q_df (pd.DataFrame): A DataFrame containing the simulated probabilities and posterior probabilities with different pseudo counts.
        - bin_edges (np.ndarray): The bin edges used for quantizing the data.

    Notes
    -----
    - The function quantizes the observed and simulated series using the provided bin edges.
    - It counts the occurrences of each quantized value and normalizes them to obtain probabilities.
    - If `concurrent_data` is True, it checks that the number of observations and simulations match.
    - It adds one pseudo count to each bin to represent the uniform prior and adjusts probabilities if necessary.
    - Posterior probabilities are computed based on a range of pseudo counts to test sensitivity.

    Example
    -------
    >>> data = {'time': pd.date_range(start='1/1/2020', periods=100, freq='D'),
    ...         'obs': np.random.rand(100),
    ...         'sim': np.random.rand(100)}
    >>> df = pd.DataFrame(data)
    >>> target = lambda: None
    >>> setattr(target, 'obs_label', 'obs')
    >>> setattr(target, 'sim_label', 'sim')
    >>> setattr(target, 'obs_quantized_label', 'obs_quantized')
    >>> setattr(target, 'sim_quantized_label', 'sim_quantized')
    >>> bin_edges = np.linspace(0, 1, 17)
    >>> pseudo_counts = [0.1, 0.5, 1.0]
    >>> p_obs, q_df = compute_probabilities(df, target, 8, pseudo_counts, True, bin_edges)
    >>> print(p_obs)
    >>> print(q_df)
    """
    try:
        # note that digitize is 1-indexed
        # print('n bin edges ', len(bin_edges))
        df[target.obs_quantized_label] = np.digitize(df[target.obs_label], bin_edges)
        df[target.sim_quantized_label] = np.digitize(df[target.sim_label], bin_edges)
        # uns, scount = np.unique(df[target.sim_quantized_label], return_counts=True)
        # uno, ocount = np.unique(df[target.obs_quantized_label], return_counts=True)
        # print(uns)
        # print(uno)
    except Exception as e:
        print(f'Error digitizing series: {e}')
        raise Exception('Error digitizing series')
    # count the occurrences of each quantized value
    # the "simulated" series is the proxy/donor series
    # and the "observed" series is the target location 
    obs_count_df = df.groupby(target.obs_quantized_label).count()
    sim_count_df = df.groupby(target.sim_quantized_label).count()
    og_sim_count = sim_count_df[target.sim_label].copy().sum()
    
    n_obs = np.nansum(obs_count_df[target.obs_label])
    n_sim = np.nansum(sim_count_df[target.sim_label])
    
    count_df = pd.DataFrame(index=range(2**bitrate))
    count_df[target.obs_label] = 0
    count_df[target.sim_label] = 0

    count_df[target.obs_label] += obs_count_df[target.obs_label]
    count_df[target.sim_label] += sim_count_df[target.sim_label]
    count_df.fillna(0, inplace=True)
    
    # obs_count = .join(obs_count_df[target.obs_label]).fillna(0)
    # sim_count = pd.DataFrame(index=range(2**bitrate)).join(sim_count_df[target.sim_label]).fillna(0)

    if concurrent_data:
        assert n_obs == n_sim, 'Number of observations and simulations do not match.'
    
    foo = count_df.sum()
    
    if concurrent_data is True:
        if foo[target.sim_label] != foo[target.obs_label]:
            print(f'Concurrent: {concurrent_data}')
            print(f'sim count:')
            print(foo)
            # print('obs count')
            # print(obs_count)
            print('og sim count')
            print(og_sim_count)
            raise Exception('Counts do not match.')
      
    p_obs = count_df[target.obs_label].values / n_obs
    p_sim = count_df[target.sim_label].values / n_sim

    # print(concurrent_data, n_obs, n_sim, round(p_obs.sum(), 2), round(p_sim.sum(), 2))
    
    # here we add one pseudo-count to each bin to represent the uniform prior
    # doesn't matter what label you use, all columns are the same
    q_df = pd.DataFrame()
    q_df['q_sim_no_prior'] = p_sim
    q_df['q_uniform'] = 1.0 / 2.0**bitrate

    # check for rounding errors, p and q should sum to 1
    # if not, renormalize the probabilities
    sum_q = np.sum(q_df['q_uniform'])
    if round(sum_q, 2) != 1.0:
        diff = 1.0 - sum_q
        q_df['q_uniform'] += diff / 2**bitrate
        new_sum = np.sum(q_df['q_uniform'])
        print(f'adjusted q_uniform: {new_sum:.3f}')

    sum_p = np.sum(p_obs)
    if round(sum_p, 2) != 1.0:
        diff = 1.0 - sum_p
        p_obs += diff / 2**bitrate
        new_sum = np.sum(p_obs)
        if round(new_sum, 2) != 1.0:
            raise Exception('Renormalization of observed probabilities failed.')
    
    # compute the posterior probabilities based on 
    # a wide range of priors to test sensitivity
    for pseudo_counts in pseudo_counts:
        adjusted_counts = [x + 10**pseudo_counts for x in count_df[target.sim_label]]
        tot_adjusted_counts = np.nansum(adjusted_counts)
        q_df[f'q_post_{pseudo_counts}R'] = adjusted_counts / tot_adjusted_counts
        assert np.round(q_df[f'q_post_{pseudo_counts}R'].sum(), 2) == 1, 'Posterior probabilities do not sum to 1.'
    
    return p_obs, q_df


def process_probabilities(df, proxy, target, bitrate, concurrent_data, partial_counts, pseudo_counts):
    """
    Processes the probabilities of observed and simulated data for a given proxy and target station.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time series data for the proxy and target stations.
    proxy : object
        An object representing the proxy station.
    target : object
        An object representing the target station. It should have attributes `obs_label` and `sim_label`.
    bitrate : int
        The number of bits used for quantizing the observed series.
    concurrent_data : bool
        A flag indicating whether the data is concurrent.
    partial_counts : bool
        A flag indicating whether to use partial observation counts with error adjustment.

    Returns
    -------
    tuple
        A tuple containing three elements:
        - p_obs (dict): The probabilities of the observed data.
        - p_sim (dict): The probabilities of the simulated data.
        - bin_edges (np.ndarray): The bin edges used for quantizing the data.

    Notes
    -----
    - The function computes the bin edges based on equal width in log space using `uniform_log_bins`.
    - If `partial_counts` is True, it computes error-adjusted probabilities by adding uniformly distributed error to the observed data.
    - If `partial_counts` is False, it computes the observed and simulated distribution probabilities directly using `compute_probabilities`.

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'time': pd.date_range(start='1/1/2020', periods=100, freq='D'),
    ...     'proxy': np.random.rand(100),
    ...     'target_obs': np.random.rand(100),
    ...     'target_sim': np.random.rand(100)
    ... })
    >>> proxy = Station(id='proxy')
    >>> target = Station(id='target', obs_label='target_obs', sim_label='target_sim')
    >>> p_obs, p_sim, bin_edges = process_probabilities(df, proxy, target, 8, True, False)
    >>> print(p_obs, p_sim, bin_edges)
    """    
    # compute the bin edges based on equal width in log space
    bin_edges = uniform_log_bins(df, proxy, bitrate)    
        
    if partial_counts == True:
        # add a uniformly distributed error to the observed data
        # and compute probabilities from partial observation counts
        # where counts are divided based on the proportion of the bin 
        # that the measurement error falls within
        p_obs = compute_error_adjusted_probabilities(df, target.obs_label, bin_edges, error_factor=0.1)
        p_sim = compute_error_adjusted_probabilities(df, target.sim_label, bin_edges, error_factor=0.1)
    else:
        # computes the observed P and simulation Q distribution probabilities 
        # as dicts by bin number, probability key-value pairs
        # test a wide range of uniform priors via pseudo counts
        p_obs, p_sim = compute_probabilities(
            df, target, bitrate, pseudo_counts, concurrent_data, bin_edges)
            
    return p_obs, p_sim, bin_edges


def compute_cod(df, obs, sim):
    """
    Computes the coefficient of determination (R^2) between observed and simulated series.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time series data.
    obs : object
        An object representing the observed station, with an attribute `id` specifying the observed data column name.
    sim : object
        An object representing the simulated station, with an attribute `id` specifying the simulated data column name.

    Returns
    -------
    float
        The coefficient of determination (R^2) between the observed and simulated series. Returns 0 if linear regression fails.

    Notes
    -----
    - The function uses `scipy.stats.linregress` to compute the linear regression between the observed and simulated series.
    - It returns the square of the correlation coefficient (R^2) as the coefficient of determination.
    - If linear regression fails, the function prints an error message and returns 0.

    Example
    -------
    >>> df = pd.DataFrame({'obs': [1, 2, 3], 'sim': [1.1, 1.9, 3.2]})
    >>> obs = lambda: None
    >>> setattr(obs, 'id', 'obs')
    >>> sim = lambda: None
    >>> setattr(sim, 'id', 'sim')
    >>> r_squared = compute_cod(df, obs, sim)
    >>> print(r_squared)
    0.996...
    """
    try:
        _, _, r_value, _, _ = linregress(df[obs.id], df[sim.id])
        return r_value**2
    except Exception as ex:
        print(f'Linear regression failed on {obs.id} and {sim.id} (N={len(df)})')
        return 0
    

def compute_nse(df, obs, sim):
    """
    Computes the Nash-Sutcliffe Efficiency (NSE) between observed and simulated series.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time series data.
    obs : object
        An object representing the observed station, with an attribute `id` specifying the observed data column name.
    sim : object
        An object representing the simulated station, with an attribute `id` specifying the simulated data column name.

    Returns
    -------
    float
        The Nash-Sutcliffe Efficiency (NSE) between the observed and simulated series.

    Notes
    -----
    - NSE is computed as 1 minus the ratio of the sum of squared differences between observed and simulated values 
    to the sum of squared differences between observed values and the mean of observed values.
    - The function handles cases where the mean of the observed values is used in the denominator.

    Example
    -------
    >>> df = pd.DataFrame({'obs': [1, 2, 3], 'sim': [1.1, 1.9, 3.2]})
    >>> obs = lambda: None
    >>> setattr(obs, 'id', 'obs')
    >>> sim = lambda: None
    >>> setattr(sim, 'id', 'sim')
    >>> nse = compute_nse(df, obs, sim)
    >>> print(nse)
    0.995...
    """
    obs_mean = df[obs.id].mean()
    return 1 - (df[sim.id] - df[obs.id]).pow(2).sum() / (df[obs.id] - obs_mean).pow(2).sum()
    obs_mean = df[obs.id].mean()
    return 1 - (df[sim.id] - df[obs.id]).pow(2).sum() / (df[obs.id] - obs_mean).pow(2).sum()


def compute_kge(df, obs, sim):
    """
    Computes the Kling-Gupta Efficiency (KGE) between observed and simulated series.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time series data.
    obs : object
        An object representing the observed station, with an attribute `id` specifying the observed data column name.
    sim : object
        An object representing the simulated station, with an attribute `id` specifying the simulated data column name.

    Returns
    -------
    float
        The Kling-Gupta Efficiency (KGE) between the observed and simulated series.

    Notes
    -----
    - KGE is computed using the correlation coefficient, the ratio of the means, and the ratio of the coefficients of 
    variation between observed and simulated series.
    - The function combines these components into the KGE formula to evaluate the efficiency.

    Example
    -------
    >>> df = pd.DataFrame({'obs': [1, 2, 3], 'sim': [1.1, 1.9, 3.2]})
    >>> obs = lambda: None
    >>> setattr(obs, 'id', 'obs')
    >>> sim = lambda: None
    >>> setattr(sim, 'id', 'sim')
    >>> kge = compute_kge(df, obs, sim)
    >>> print(kge)
    0.99...
    """
    # compute the Kling-Gupta Efficiency
    # between the obs and sim series
    obs_mean = df[obs.id].mean()
    sim_mean = df[sim.id].mean()
    obs_std = df[obs.id].std()
    sim_std = df[sim.id].std()
    r = np.corrcoef(df[obs.id], df[sim.id])[0,1]
    beta = sim_mean / obs_mean
    gamma = (sim_std / sim_mean) / (obs_std / obs_mean)
    return 1 - np.sqrt((r-1)**2 + (beta - 1)**2 + (gamma - 1)**2)



def process_batch(inputs):
    """
    Processes a batch of input data to compute various hydrological metrics and store the results.

    Parameters
    ----------
    inputs : tuple
        A tuple containing the following elements:
        - proxy (str): The proxy station ID.
        - target (str): The target station ID.
        - bitrate (int): The bitrate to be used for processing.
        - completeness_threshold (float): The threshold for data completeness.
        - min_years (int): The number of years for which data is considered.
        - partial_counts (bool): A flag indicating whether partial counts are used.

    Returns
    -------
    dict or None
        A dictionary containing the computed metrics and results for the batch, or None if the batch is not processed.

    Notes
    -----
    - The function retrieves station characteristics and checks if the catchments are nested.
    - It computes the spatial distance between catchment centroids and skips processing if the distance exceeds 1000 km.
    - Data completeness is checked, and the function raises an error if required information is missing.
    - Concurrent and non-concurrent data is retrieved, transformed, and jittered.
    - Simulated flow at the target station is calculated based on unit area runoff scaling.
    - Various efficiency metrics such as Coefficient of Determination (COD), Nash-Sutcliffe Efficiency (NSE), 
    and Kling-Gupta Efficiency (KGE) are computed for the concurrent data.
    - Probability Mass Functions (PMFs) and divergences are processed for both concurrent and non-concurrent data 
    if sufficient observations are available.

    Example
    -------
    >>> inputs = ('proxy_station', 'target_station', 16, 0.9, 10, True)
    >>> result = process_batch(inputs)
    >>> if result is not None:
    ...     print("Processed results:", result)
    ... else:
    ...     print("No results processed.")
    """
    proxy, target, bitrate, completeness_threshold, min_years, partial_counts, attr_cols, climate_cols, pseudo_counts = inputs
        
    proxy_id, target_id = str(proxy), str(target)
    bitrate = int(bitrate)
    completeness_threshold = float(completeness_threshold)
    min_years = int(min_years)
    min_observations = min_years * completeness_threshold * 365

    result = {'proxy': str(proxy_id), 'target': str(target_id), 'bitrate': bitrate, 'completeness_threshold': completeness_threshold}    
    
    station_info = {
        'proxy': hs_df[hs_df['official_id'] == proxy_id].copy().to_dict(orient='records')[0],
        'target': hs_df[hs_df['official_id'] == target_id].copy().to_dict(orient='records')[0]
    }

    # check if the polygons are nested
    result['nested_catchments'] = check_if_nested(station_info['proxy'], station_info['target'])

    # we don't need to add the attributes, these can be retrieved later.
    # HOWEVER, sacrificing a bit of disk space now 
    # saves substantial computation time later
    for l in ['proxy', 'target']:
        for c in attr_cols + climate_cols:
            result[f'{l}_{c.lower()}'] = station_info[l][c.lower()]
    
    # for stn in pair:
    proxy = Station(station_info['proxy'], bitrate)
    target = Station(station_info['target'], bitrate)
    
    # compute spatial distance
    p1, p2 = station_info['proxy']['geometry'].centroid, station_info['target']['geometry'].centroid
    # compute the distance between catchment centroids (km)
    centroid_distance = p1.distance(p2) / 1000
    result['centroid_distance'] = round(centroid_distance, 2)
    if centroid_distance > 1000: return None

    if np.isnan(target.drainage_area_km2):
        raise ValueError(f'No drainage area for {target_id}')
    if np.isnan(proxy.drainage_area_km2):
        raise ValueError(f'No drainage area for {proxy_id}')

    # Retrieve the data for both stations
    adf = retrieve_nonconcurrent_data(proxy_id, target_id)
    
    for stn in [proxy, target]:
        adf = transform_and_jitter(adf, stn)

    # simulate flow at the target based on equal unit area runoff scaling
    adf[target.sim_label] = adf[proxy.id] * (target.drainage_area_km2 / proxy.drainage_area_km2)

    # filter for the concurrent data
    df = adf.copy().dropna(subset=[proxy_id, target_id], how='any')
    counts = df[[proxy_id, target_id]].count(axis=0)

    if counts[proxy_id] != counts[target_id]:
        raise ValueError(f'Unequal counts for {proxy_id} and {target_id}')
    
    result['num_concurrent_obs'] = len(df)
    counts = adf.count(axis=0)

    proxy.n_obs, target.n_obs = counts[proxy_id], counts[target_id]
    result[f'proxy_n_obs'] = proxy.n_obs
    result[f'target_n_obs'] = target.n_obs

    result[f'proxy_frac_concurrent'] = len(df) / proxy.n_obs
    result[f'target_frac_concurrent'] = len(df) / target.n_obs
    
    if (counts[proxy_id] == 0) or (counts[target_id] == 0):
        print(f'   Zero observation count detected.  Skipping.')
        return None
            
    if len(df) > min_observations:        
        # compute coefficient of determination
        result['cod'] = compute_cod(df, proxy, target)

        # compute Nash-Sutcliffe efficiency
        result['nse'] = compute_nse(df, proxy, target)

        # compute the Kling-Gupta efficiency
        result['kge'] = compute_kge(df, proxy, target)
    
    # process the PMFs and divergences for concurrent data 
    # using a range of uniform priors via pseudo counts
    if len(df) > min_observations:
        # print('processing concurrent data')
        p_obs, p_sim, bin_edges = process_probabilities(df, proxy, target, bitrate, True, partial_counts, pseudo_counts)
        result = process_divergences(p_obs, p_sim, bin_edges, bitrate, True, result)
    
    if (target.n_obs > min_observations) & (proxy.n_obs > min_observations):
        # process the PMFs and divergences for concurrent=False
        # print('processing non-concurrent data')        
        p_obs, p_sim, bin_edges = process_probabilities(adf, proxy, target, bitrate, False, partial_counts, pseudo_counts)
        result = process_divergences(p_obs, p_sim, bin_edges, bitrate, False, result)            
        
    return result



def check_distribution(p, q, c):
    """
    Checks whether the provided probability distributions sum to 1.

    Parameters
    ----------
    p : np.ndarray
        The observed probability distribution.
    q : np.ndarray
        The simulated probability distribution.
    c : str
        A label or context string used in the error message if the distributions do not sum to 1.

    Raises
    ------
    AssertionError
        If the sum of probabilities in `p` or `q` is not equal to 1.

    Example
    -------
    >>> p = np.array([0.2, 0.3, 0.5])
    >>> q = np.array([0.1, 0.4, 0.5])
    >>> check_distribution(p, q, 'Test')
    """
    sum_p = round(np.nansum(p), 2)
    sum_q = round(np.nansum(q), 2)

    msg = f'{c}: sum(p)={sum_p:.2f} sum(q)={sum_q:.2f}'        
    assert sum_p == 1.0 and sum_q == 1.0, msg


    
def process_KL_divergence(p_obs, p_sim, bitrate, concurrent_data):
    """
    Processes the Kullback-Leibler (KL) divergence between observed and simulated probability distributions.

    Parameters
    ----------
    p_obs : np.ndarray
        The observed probability distribution.
    p_sim : pd.DataFrame
        A DataFrame containing the simulated probability distributions with different priors.
    bitrate : int
        The number of bits used for quantizing the observed series.
    concurrent_data : bool
        A flag indicating whether the data is concurrent.

    Returns
    -------
    pd.Series
        A series containing the sum of KL divergences for each simulated distribution.

    Raises
    ------
    Exception
        If any value in the simulated distribution is zero, which should not happen due to the addition of pseudo-counts.

    Notes
    -----
    - The function computes the KL divergence for each simulated distribution in `p_sim`.
    - It ensures that the probability distributions sum to 1 before computing the divergence.
    - If the data is concurrent, the divergence labels are prefixed with 'dkl_concurrent_', otherwise 'dkl_nonconcurrent_'.

    Example
    -------
    >>> p_obs = np.array([0.2, 0.3, 0.5])
    >>> p_sim = pd.DataFrame({'q_post_0.1R': [0.1, 0.4, 0.5], 'q_post_0.5R': [0.2, 0.3, 0.5]})
    >>> bitrate = 3
    >>> concurrent_data = True
    >>> sum_dkl = process_KL_divergence(p_obs, p_sim, bitrate, concurrent_data)
    >>> print(sum_dkl)
    """
    # dkl_df = uf.compute_kl_divergence(p_obs, p_sim, bitrate, concurrent_data)
    df = pd.DataFrame()
    df['bin'] = range(1, 2**bitrate+1)
    df.set_index('bin', inplace=True)

    for c in p_sim.columns:
        if c == 'q_sim_no_prior':
            continue
        q = p_sim[c].values
        p = p_obs

        # ensure that the probabilities sum to 1
        check_distribution(p, q, c)

        label = 'dkl_nonconcurrent_' + '_'.join(c.split('_')[1:])
        if concurrent_data is True:
            label = 'dkl_concurrent_' + '_'.join(c.split('_')[1:])

        kld = 0
        for i in range(len(p)):
            if (p[i] == 0) | (q[i] == 0):
                # we define 0 * log(0/0) = 0
                # but q[i] should not be zero because 
                # we added a pseudo-count (Dirichlet) prior
                continue
            if q[i] == 0:
                raise Exception(f'q[i] is zero at i={i}')
            else:
                kld = p[i] * np.log2(p[i] / q[i])
                df.loc[i+1, label] = kld

    sum_dkl = df.sum()

    if any(sum_dkl.values) <= 0:
        print(f'negative or zero dkl')
        print(sum_dkl.values)
    
    return sum_dkl


def compute_tvd(p, q, q_uniform, concurrent_data):
    """
    Computes the Total Variation Distance (TVD) between observed and simulated probability distributions.

    Parameters
    ----------
    p : np.ndarray
        The observed probability distribution.
    q : np.ndarray
        The simulated probability distribution.
    q_uniform : np.ndarray
        The uniform probability distribution.
    concurrent_data : bool
        A flag indicating whether the data is concurrent.

    Returns
    -------
    dict
        A dictionary containing the TVD between the observed and simulated distributions and the TVD 
        between the observed and uniform distributions.

    Notes
    -----
    - TVD is computed as the sum of absolute differences between two distributions divided by 2.
    - The function labels the results based on whether the data is concurrent or non-concurrent.

    Example
    -------
    >>> p = np.array([0.2, 0.3, 0.5])
    >>> q = np.array([0.1, 0.4, 0.5])
    >>> q_uniform = np.array([0.33, 0.33, 0.34])
    >>> concurrent_data = True
    >>> result = compute_tvd(p, q, q_uniform, concurrent_data)
    >>> print(result)
    {'tvd_concurrent': 0.1, 'tvd_concurrent_max': 0.165}
    """
    results = {}
    tvd_label = f'tvd_nonconcurrent'
    if concurrent_data is True:
        tvd_label = f'tvd_concurrent'
    results[tvd_label] = np.sum(np.abs(np.subtract(p, q))) / 2
    results[tvd_label + '_max'] = np.sum(np.abs(np.subtract(p, q_uniform))) / 2
    return results


def compute_wasserstein_distance(bin_edges, p, q, q_uniform, concurrent_data):
    """
    Computes the Wasserstein distance between observed and simulated probability distributions.

    Parameters
    ----------
    bin_edges : np.ndarray
        The edges of the bins used for quantizing the data.
    p : np.ndarray
        The observed probability distribution.
    q : np.ndarray
        The simulated probability distribution.
    q_uniform : np.ndarray
        The uniform probability distribution.
    concurrent_data : bool
        A flag indicating whether the data is concurrent.

    Returns
    -------
    dict
        A dictionary containing the Wasserstein distance between the observed and simulated distributions 
        and the Wasserstein distance between the observed and uniform distributions.

    Notes
    -----
    - Wasserstein distance is computed using the bin midpoints in linear space.
    - The function labels the results based on whether the data is concurrent or non-concurrent.
    - An exception is raised if there is an error in computing the Wasserstein distance.

    Example
    -------
    >>> bin_edges = np.linspace(0, 1, 11)
    >>> p = np.array([0.1, 0.2, 0.3, 0.4])
    >>> q = np.array([0.1, 0.25, 0.25, 0.4])
    >>> q_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    >>> concurrent_data = False
    >>> result = compute_wasserstein_distance(bin_edges, p, q, q_uniform, concurrent_data)
    >>> print(result)
    {'wasserstein_nonconcurrent': 0.05, 'wasserstein_nonconcurrent_max': 0.1}
    """
    result = {}
    wasserstein_label = f'wasserstein_nonconcurrent'
    if concurrent_data is True:
        wasserstein_label = f'wasserstein_concurrent'
    
    # Compute the bin midpoints in the linear space
    # this represents the water volume in the bin
    bin_midpoints = list(np.add(bin_edges[:-1], bin_edges[1:]) / 2)

    # append the lowest and highest edges to the bin midpoints 
    # to serve as the weight (volume) of the highest bin
    bin_midpoints = [bin_edges[0]] + bin_midpoints + [bin_edges[-1]]
    # check for monitonicity
    assert np.all(np.diff(bin_midpoints) >= 0)

    try:
        result[wasserstein_label] = wasserstein_distance(bin_midpoints, bin_midpoints, p, q) 
        result[wasserstein_label + '_max'] = wasserstein_distance(bin_midpoints, bin_midpoints, p, q_uniform)   
        return result
    except ValueError as e:
        print(f'Error computing Wasserstein distance: {e}')
        print(f'p: {len(p)}')
        print(p)
        print(f'q: {len(q)}')
        print(q)
        print(f'bin_midpoints: {len(bin_midpoints)}')
        print(f'bin_edges: {len(bin_edges)}')
        raise Exception('Wasserstein distance computation failed')
        # result[wasserstein_label] = np.nan
        # result[wasserstein_label + '_max'] = np.nan


def process_divergences(p_obs, p_sim, bin_edges, bitrate, concurrent_data, result):
    """
    Processes the divergences between observed and simulated probability distributions, including KL divergence, TVD, and Wasserstein distance.

    Parameters
    ----------
    p_obs : np.ndarray
        The observed probability distribution.
    p_sim : pd.DataFrame
        A DataFrame containing the simulated probability distributions with different priors.
    bin_edges : np.ndarray
        The edges of the bins used for quantizing the data.
    bitrate : int
        The number of bits used for quantizing the observed series.
    concurrent_data : bool
        A flag indicating whether the data is concurrent.
    result : dict
        A dictionary to store the computed divergence results.

    Returns
    -------
    dict
        A dictionary containing the computed divergence results, including KL divergence, TVD, and Wasserstein distance.

    Notes
    -----
    - The function computes KL divergence using `process_KL_divergence`.
    - It computes TVD and Wasserstein distance and updates the result dictionary with these values.

    Example
    -------
    >>> p_obs = np.array([0.2, 0.3, 0.5])
    >>> p_sim = pd.DataFrame({
    ...     'q_sim_no_prior': [0.1, 0.4, 0.5],
    ...     'q_uniform': [0.33, 0.33, 0.34],
    ...     'q_post_0.1R': [0.2, 0.3, 0.5]
    ... })
    >>> bin_edges = np.linspace(0, 1, 4)
    >>> bitrate = 3
    >>> concurrent_data = True
    >>> result = {}
    >>> result = process_divergences(p_obs, p_sim, bin_edges, bitrate, concurrent_data, result)
    >>> print(result)
    """
    dkl_by_prior = process_KL_divergence(p_obs, p_sim, bitrate, concurrent_data)

    p = p_obs
    q = p_sim['q_sim_no_prior'].values
    q_uniform = p_sim['q_uniform'].values

    tvd_result = compute_tvd(p, q, q_uniform, concurrent_data)
    wd_result = compute_wasserstein_distance(bin_edges, p, q, q_uniform, concurrent_data)

    result.update(tvd_result)
    result.update(wd_result)
    result.update(dkl_by_prior.to_dict())

    return result