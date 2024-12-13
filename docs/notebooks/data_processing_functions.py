import os
from time import time
import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.stats import entropy, wasserstein_distance, linregress
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    f1_score, d2_log_loss_score, confusion_matrix, fbeta_score
)
from sklearn.model_selection import StratifiedKFold

from shapely.geometry import Point

import xgboost as xgb
import multiprocessing as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

STREAMFLOW_DIR = '/media/danbot2/Samsung_T5/geospatial_data/HYSETS_data/hysets_series'

class Station:
    """
    A class used to represent a Station.

    The Station class initializes an object with attributes based on a provided dictionary of station information.
    Each key-value pair in the dictionary is unpacked and set as an attribute of the Station object.

    Attributes
    ----------f
    id : str
        The official ID of the station, derived from the 'Official_ID' key in the provided dictionary.

    Methods
    -------
    __init__(self, station_info) -> None
        Initializes the Station object by unpacking a dictionary of station information into attributes.
    """

    def __init__(self, station_info) -> None:
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
        self.sim_label = f"{self.id}_sim"
        self.sim_label_parametric = f"{self.id}_sim_parametric"
        self.obs_label = f"{self.id}"
        self.sim_log_label = f"{self.id}_sim_log10"
        self.obs_log_label = f"{self.id}_log10"
        # self.UR_label = f'{self.id}_UR'
        # self.logUR_label = f'{self.id}_logUR'
        # self.log_sim_label = f'{self.id}_sim_log10'
        # self.sim_quantized_label = f"sim_quantized_{self.id}_{bitrate}b"
        # self.obs_quantized_label = f"obs_quantized_{self.id}_{bitrate}b"
        # self.quantized_label_ = f'quantized_{self.id}_sim_{bitrate}b'


# Add custom CSS for styling (optional)
table_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400&display=swap');

.styled-table {
    border-collapse: collapse;
    margin: 25px 0;
    font-size: 0.9em;
    font-family: Roboto, sans-serif;
    width: 100%;
    text-align: left;
}
.styled-table th {
    background-color: white;
    color: black;
    padding: 12px 15px;
}
.styled-table td {
    padding: 12px 15px;
}
.styled-table tr {
    border-bottom: 1px solid #dddddd;
}
.styled-table tr:nth-of-type(even) {
    background-color: #f3f3f3;
}
.styled-table tr:last-of-type {
    border-bottom: 2px solid #009879;
}
</style>
"""

def format_fig_fonts(fig, font_size=20, font='Bitstream Charter', legend=True):
    fig.xaxis.axis_label_text_font_size = f'{font_size}pt'
    fig.yaxis.axis_label_text_font_size = f'{font_size}pt'
    fig.xaxis.major_label_text_font_size = f'{font_size-2}pt'
    fig.yaxis.major_label_text_font_size = f'{font_size-2}pt'
    fig.yaxis.axis_label_text_font = font
    fig.xaxis.axis_label_text_font = font
    fig.xaxis.major_label_text_font = font
    fig.yaxis.major_label_text_font = font
    if legend == True:
        fig.legend.label_text_font_size = f'{font_size-2}pt'
        fig.legend.label_text_font = font
    return fig
    

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
    dtype_spec = {"proxy": str, "target": str}
    if os.path.exists(out_fpath):
        results_df = pd.read_csv(out_fpath, dtype=dtype_spec)
        print(f"    Loaded {len(results_df)} existing results")
        return results_df
    else:
        print("    No existing results found")
        return pd.DataFrame()


def add_attributes(attr_df, df_relations, attribute_cols):
    """
    Adds attributes from the df_attributes to the df_relations based on the 'proxy' and 'target' columns
    using map for efficient lookups.

    Parameters:
    df_attributes (pd.DataFrame): DataFrame with 'id' and attribute columns.
    df_relations (pd.DataFrame): DataFrame with 'proxy' and 'target' columns.
    attribute_cols (list of str): List of attribute columns to add to df_relations.

    Returns:
    pd.DataFrame: Updated df_relations with added attribute columns.
    """
    # Create dictionaries for each attribute for quick lookup
    attr_dicts = {col: attr_df.set_index('official_id')[col].to_dict() for col in attribute_cols}

    # Add target attributes
    for col in attribute_cols:
        df_relations[f'target_{col}'] = df_relations['target'].map(attr_dicts[col])

    # Add proxy attributes
    for col in attribute_cols:
        df_relations[f'proxy_{col}'] = df_relations['proxy'].map(attr_dicts[col])

    return df_relations


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
    id_pairs_df = pd.DataFrame(id_pairs, columns=["proxy", "target"])

    # Perform an outer merge and keep only those rows that are NaN in results_df index
    # This indicates that these rows were not found in results_df
    merged_df = id_pairs_df.merge(
        results_df, on=["proxy", "target"], how="left", indicator=True
    )
    filtered_df = merged_df[merged_df["_merge"] == "left_only"]

    # Convert the filtered DataFrame back to a list of tuples
    remaining_pairs = list(zip(filtered_df["proxy"], filtered_df["target"]))

    return remaining_pairs


def count_complete_years(df, date_column, value_column):
    # Ensure the date column is a datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Set the date column as the index for easier grouping
    df.set_index(date_column, inplace=True)

    # Group the data by year and month
    monthly_data = df.groupby([df.index.year, df.index.month])[value_column]

    # Function to check if a month is complete (no more than 10% missing)
    def is_complete_month(group):
        total_days = group.size
        missing_days = group.isna().sum()
        return missing_days / total_days <= 0.10

    # Create a DataFrame indicating which months are complete
    monthly_completeness = monthly_data.apply(is_complete_month).unstack()

    # Find complete years (where all 12 months are complete)
    complete_years = monthly_completeness[monthly_completeness.sum(axis=1) == 12].index

    # Return the number of complete years
    return len(complete_years)


def check_if_seasonal(df, date_column, value_column):
    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Set the date column as the index
    df.dropna(subset=[value_column], inplace=True)
    df.set_index(date_column, inplace=True)

    # Extract the months and check if all months (1 to 12) are present in the record
    months_present = df.index.month.unique()
    df.reset_index(inplace=True)

    # Check if all months 1 to 12 are present
    return len(list(set(months_present))) < 12


def get_timeseries_data(official_id, min_flow=1e-4, try_counter=0):
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
    fpath = os.path.join(STREAMFLOW_DIR, f"{official_id}.csv")
    if not os.path.exists(fpath):
        new_fpath = os.path.join(STREAMFLOW_DIR, f"0{official_id}.csv")
        try_counter += 1
        if try_counter > 3:
            raise FileNotFoundError(f"Check station number {official_id}, file not found after {try_counter} attempts")
        else:
            return get_timeseries_data(f"0{official_id}", try_counter=try_counter)
            
    df = pd.read_csv(fpath, parse_dates=["time"])
    df[f"{official_id}_low_flow_flag"] = (df["discharge"] < min_flow).astype(bool)
    # assign a small flow value instead of zero
    df["discharge"] = df["discharge"].clip(lower=min_flow)
    # rename the discharge column to the station id
    df.rename(columns={"discharge": official_id}, inplace=True)
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
    df1 = get_timeseries_data(proxy).set_index("time")
    df2 = get_timeseries_data(target).set_index("time")
    df = pd.concat([df1, df2], join="outer", axis=1)
    df["year"] = df.index.year
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
        msg = f"Noise addition creates values < 1e-5 (ID: {station.id})"
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
    p1 = Point(stn1["Centroid_Lon_deg_E"], stn1["Centroid_Lat_deg_N"])
    p2 = Point(stn2["Centroid_Lon_deg_E"], stn2["Centroid_Lat_deg_N"])
    gdf = gpd.GeoDataFrame(geometry=[p1, p2], crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3005")
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
    stn_id = row["official_id"]
    df = get_timeseries_data(stn_id)
    df.dropna(subset=[stn_id], inplace=True)
    if df.empty:
        raise Exception(f'{stn_id} is empty')
    min_q, max_q = df[stn_id].min() - 1e-6, df[stn_id].max() + 1e-6
    assert min_q > 0
    # use equal width bins in log10 space
    log_edges = np.linspace(np.log10(min_q), np.log10(max_q), 2**bitrate)
    linear_edges = [10**e for e in log_edges]
    df[f"{bitrate}_bits_quantized"] = np.digitize(df[stn_id], linear_edges)
    unique, counts = np.unique(df[f"{bitrate}_bits_quantized"], return_counts=True)
    count_dict = {k: 1 / v for k, v in zip(unique, counts)}
    frequencies = [
        count_dict[e] if e in count_dict else 0 for e in range(1, 2**bitrate)
    ]
    normed_frequencies = frequencies / sum(frequencies)
    return entropy(normed_frequencies, base=2)


def format_features(input_attributes):
    features = []

    for a in input_attributes:
        features.append(f"proxy_{a}".lower())
        features.append(f"target_{a}".lower())
    # add the distance feature
    features.append("centroid_distance")
    return features


def train_test_split_by_official_id(holdout_pct, stations, nfolds):
    """
    Splits a list of stations into training and holdout test sets, and further splits the training set into cross-validation folds, randomly shuffling the order of training set stations to generate random cross validation subsets.

    Parameters:
    -----------
    holdout_pct : float
        The percentage of stations to be used as the holdout test set. This should be a float between 0 and 1.

    stations : list
        A list of station identifiers.

    nfolds : int
        The number of folds for cross-validation splitting of the training set.

    Returns:
    --------
    training_cv_sets : list of numpy.ndarray
        A list containing `nfolds` arrays, each representing a fold of training stations for cross-validation.

    holdout_test_stns : numpy.ndarray
        An array of station identifiers that are reserved for the holdout test set.

    Example:
    --------
    >>> stations = ['station1', 'station2', 'station3', 'station4', 'station5']
    >>> train_test_split_by_official_id(0.2, stations, 3)
    (array([['station2', 'station1'], ['station3'], ['station4']], dtype='<U8'),
    array(['station5'], dtype='<U8'))

    Notes:
    ------
    - The function randomly selects stations for the holdout test set without replacement.
    - The remaining stations are shuffled and split into `nfolds` approximately equal sets for cross-validation.
    - The randomness in selecting and shuffling the stations can be controlled by setting a random seed before calling this function.
    """
    n_holdout = int(len(stations) * holdout_pct)
    holdout_test_stns = np.random.choice(stations, n_holdout, replace=False)
    training_stations = [e for e in stations if e not in holdout_test_stns]
    np.random.shuffle(training_stations)
    training_cv_sets = np.array_split(np.array(training_stations), nfolds)
    return training_cv_sets, holdout_test_stns


def filter_input_data_by_official_id(df, stations):
    """
    Filters a DataFrame to include only rows where both 'proxy' and 'target' columns match a given list of station identifiers.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be filtered. It must include 'proxy' and 'target' columns.

    stations : list
        A list of station identifiers to filter the DataFrame by.

    Returns:
    --------
    pandas.DataFrame
        A filtered copy of the input DataFrame where both 'proxy' and 'target' columns contain values from the given list of station identifiers.

    Example:
    --------
    >>> import pandas as pd
    >>> data = {'proxy': ['station1', 'station2', 'station3', 'station4'],
                'target': ['station1', 'station2', 'station5', 'station4'],
                'value': [10, 20, 30, 40]}
    >>> df = pd.DataFrame(data)
    >>> stations = ['station1', 'station2', 'station4']
    >>> filter_input_data_by_official_id(df, stations)
         proxy   target  value
    0  station1  station1     10
    1  station2  station2     20
    3  station4  station4     40

    Notes:
    ------
    - The function returns a copy of the filtered DataFrame to avoid modifying the original DataFrame.
    - The filtering is performed using the 'proxy' and 'target' columns to ensure that both columns' values are in the specified list of stations.
    """
    return df[df["proxy"].isin(stations) & df["target"].isin(stations)].copy()


def train_xgb_model(
    input_data, train_stns, test_stns, attributes, target, params, num_boost_rounds,
):
    

    X_train = train_data[attributes].values
    Y_train = train_data[target].values
    X_test = test_data[attributes].values
    Y_test = test_data[target].values

    model = xgb.XGBRegressor(**params)

    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)

    eval_list = [(dtrain, "train"), (dtest, "eval")]

    bst = xgb.train(
        params,
        dtrain,
        num_boost_rounds,
        evals=eval_list,
        verbose_eval=0,
        early_stopping_rounds=20,
    )

    predicted_y = bst.predict(dtest)

    test_results = pd.DataFrame(
        {
            "predicted": predicted_y,
            "actual": Y_test,
        }
    )

    rmse = root_mean_squared_error(predicted_y, Y_test)
    mae = mean_absolute_error(predicted_y, Y_test)

    return bst, rmse, mae, test_results


def train_binary_xgb_model(
    input_data, train_stns, test_stns, attributes, target, params, num_boost_rounds,
):
    train_data = filter_input_data_by_official_id(input_data, train_stns)
    test_data = filter_input_data_by_official_id(input_data, test_stns)

    X_train = train_data[attributes].values
    Y_train = train_data[target].values
    X_test = test_data[attributes].values
    Y_test = test_data[target].values

    model = xgb.XGBClassifier(**params)

    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)

    eval_list = [(dtrain, "train"), (dtest, "eval")]

    bst = xgb.train(
        params,
        dtrain,
        num_boost_rounds,
        evals=eval_list,
        verbose_eval=0,
        early_stopping_rounds=20,
    )

    predicted_y = bst.predict(dtest)

    test_results = pd.DataFrame(
        {
            "predicted": predicted_y,
            "actual": Y_test,
        }
    )
    test_results['predicted'] = test_results['predicted'].astype(int)
    test_results['actual'] = test_results['actual'].astype(int)

    return bst, test_results


def compute_empirical_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf


def run_xgb_CV_trials(
    set_name,
    features,
    target,
    input_data,
    n_optimization_rounds,
    nfolds,
    num_boost_rounds,
    results_folder,
    loss='reg:squarederror',
    random_seed=42
):
    input_data['group'] = input_data[['proxy','target']].astype(str).agg('_'.join, axis=1)
    groups = input_data['group'].unique()
    y_grouped = input_data.groupby('group')[target].first().values

    # Step 3: Create stratified KFold object
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    X_input, Y_input = (
        input_data.loc[:, features].values,
        input_data.loc[:, target].values,
    )
    # Bin the continuous target variable into quantiles
    # y_binned = pd.qcut(Y_input, q=20, labels=False)  # Quantile-based discretization

    # split the data into folds such that the target
    # distribution is similar across each fold
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=int(random_seed))

    sample_choices = np.arange(0.1, 0.9, 0.02)
    lr_choices = np.arange(0.001, 0.12, 0.0005)
    learning_rates = np.random.choice(lr_choices, n_optimization_rounds)
    subsamples = np.random.choice(sample_choices, n_optimization_rounds)
    colsamples = np.random.choice(sample_choices, n_optimization_rounds)

    all_results = []
    best_result = (None, np.inf, None)
    best_params = None
    best_mean_test_perf = np.inf
    best_convergence_df = pd.DataFrame()
    best_trial_test_predictions = None
    output_target_cdfs = None
    for trial in range(n_optimization_rounds):
        lr, ss, cs = learning_rates[trial], subsamples[trial], colsamples[trial]
        results_fname = f"{set_name}_{lr:.3f}_lr_{ss:.3f}_sub_{cs:.3f}_col.csv"
        results_fpath = os.path.join(results_folder, results_fname)
        params = {
            "objective": loss,
            "eta": lr,
            "max_depth": 6,  # use default max_depth
            "subsample": ss,
            "colsample_bytree": cs,
            "seed": random_seed,
            "device": "cuda",  # note, change this to 'cpu' if your system doesn't have a CUDA GPU
            "sampling_method": "gradient_based",
            "tree_method": "hist",
            "validate_parameters": True,
        }

        # Split into N equal segments
        fold_scores = []
        n_samples = len(X_input)
        fold_no = 0
        best_train_fold_perf, best_test_fold_perf, best_test_rounds = [], [], []
        all_fold_results = []
        fold_scores, fold_arrays = [], []
        target_cdfs = []
        for train_index, test_index in skf.split(np.zeros(n_samples), y_binned):
    
            X_train, X_test = X_input[train_index], X_input[test_index]
            Y_train, Y_test = Y_input[train_index], Y_input[test_index]
            train_ids = input_data.iloc[train_index]['official_id'].copy()
            test_ids = input_data.iloc[test_index]['official_id'].copy()
            assert len(np.intersect1d(train_ids, test_ids)) == 0

            ordered_data, fold_cdf = compute_empirical_cdf(Y_test)
            target_cdfs += [(ordered_data, fold_cdf)]
            
            if loss == 'reg:quantileerror':
                params['eval_metric'] = None
                params['quantile_alpha'] = np.array([0.05, 0.5, 0.95])
                # dtrain = xgb.QuantileDMatrix(X_train, label=Y_train)   
                dtrain = xgb.QuantileDMatrix(X_train, label=Y_train)   
                dtest = xgb.QuantileDMatrix(X_test, label=Y_test)
            else:
                dtrain = xgb.DMatrix(X_train, label=Y_train)
                dtest = xgb.DMatrix(X_test, label=Y_test)
            
            # Initialize evals_result as a DataFrame
            evals_result = {}
            eval_list = [(dtrain, "train"), (dtest, "eval")]
            model = xgb.train(
                params,
                dtrain,
                num_boost_rounds,
                evals=eval_list,
                evals_result=evals_result,
                verbose_eval=0,
                # early_stopping_rounds=20,
            )
            eval_keys = list(evals_result['train'].keys())
            if len(eval_keys) > 1:
                print(f' setting eval key to {eval_keys[0]} from {eval_keys}')
            eval_key = eval_keys[0]
            # Convert the lists to NumPy arrays with dtype=object
            train_perf = np.array(evals_result['train'][eval_key], dtype=object)
            test_perf = np.array(evals_result['eval'][eval_key], dtype=object)  
                        
            fold_progress = pd.DataFrame({
                'train': evals_result['train'][eval_key],#train_rmse,
                'test': evals_result['eval'][eval_key],#test_rmse,
                'fold': [fold_no]*len(evals_result['train'][eval_key]),
            })
            fold_arrays.append(fold_progress)
            predicted_y = model.predict(dtest)
            test_results = pd.DataFrame(
                {"predicted": predicted_y, "actual": Y_test, "official_id": test_ids}
            )
            
            # Get the round with the best validation score (out-of-sample performance)
            best_perf_round_train, best_perf_round_test = np.argmin(train_perf), np.argmin(test_perf)
            
            # Store the metrics at the best round
            train_perf_best = train_perf[best_perf_round_train]
            test_perf_best = test_perf[best_perf_round_test]

            best_train_fold_perf.append(train_perf_best)
            best_test_fold_perf.append(test_perf_best)
            best_test_rounds.append(best_perf_round_test)
            all_fold_results.append(test_results)            
            fold_no += 1

        all_test_predictions_df = pd.concat(all_fold_results)
        convergence_df = pd.concat(fold_arrays)    
        
        mean_test_perf = np.mean(best_test_fold_perf)
        stdev_test_perf = np.std(best_test_fold_perf)
        # # track the trial error metrics
        results_dict = {
            'trial': trial,
            f'test_{eval_key}_mean': mean_test_perf,
            f'test_{eval_key}_stdev': stdev_test_perf,
        }
        
        results_cols = list(results_dict.keys())
        results_dict.update(params)
        all_results.append(results_dict)
        if (trial > 0) & (trial % 10 == 0):
            print(f"   completed {trial}/{n_optimization_rounds}")
            
        if mean_test_perf < best_mean_test_perf:
            best_params = params
            best_mean_test_perf = mean_test_perf
            best_convergence_df = convergence_df
            best_trial_test_predictions = all_test_predictions_df
            output_target_cdfs = target_cdfs
            print(f'    New best result: {eval_key}={mean_test_perf:.2f} (trial {trial})')

    # save the best trial results
    best_trial_test_predictions.to_csv(results_fpath)
    results_all_trials = pd.DataFrame(all_results)
    # get the mean and standard deviation of the error metrics over all trials
    all_trials_mean = results_all_trials[f"test_{eval_key}_mean"].mean()
    all_trials_stdev = results_all_trials[f"test_{eval_key}_mean"].std()
    print(
        f"    {all_trials_mean:.2f} ± {all_trials_stdev:.3f} mean {eval_key} (of {len(results_all_trials)} hyperparameter optimization rounds.)"
    )
    return best_params, best_mean_test_perf, best_convergence_df, best_trial_test_predictions, results_all_trials, output_target_cdfs
    

def run_xgb_trials_custom_CV(
    bitrate,
    set_name,
    attributes,
    target,
    input_data,
    fold_dict, 
    n_optimization_rounds,
    num_boost_rounds,
    results_folder,
    loss='reg:squarederror',
    random_seed=42
):
    """
    Custom CV refers to cross validation.  Custom cross validation means the 
    held-out set must be determined in a more robust way to avoid "data leakage".
    That is, the pairs making up the training, validation, and test sets must 
    be made up of pairings from unique sets of stations.
    """
    # select random hyperparameters for n_optimization_rounds
    sample_choices = np.arange(0.5, 0.9, 0.02)  # subsample and colsample percentages
    lr_choices = np.arange(0.001, 0.1, 0.0005)  # learning rates
    learning_rates = np.random.choice(lr_choices, n_optimization_rounds)
    subsamples = np.random.choice(sample_choices, n_optimization_rounds)
    colsamples = np.random.choice(sample_choices, n_optimization_rounds)
    num_boost_rounds = num_boost_rounds

    all_results = []
    best_result = (None, np.inf, None)
    best_params = None
    best_mean_test_perf = np.inf
    best_convergence_df = pd.DataFrame()
    best_trial_test_predictions = None
    output_target_cdfs = None
    for trial in range(n_optimization_rounds):
        lr, ss, cs = learning_rates[trial], subsamples[trial], colsamples[trial]
        params = {
            "objective": loss,
            "eta": lr,
            # "max_depth": 6,  # use default max_depth
            # "min_child_weight": 1, # use colsample and subsample instead of min_child_weight
            "subsample": ss,
            "colsample_bytree": cs,
            "seed": random_seed,
            "device": "cuda",  # note, change this to 'cpu' if your system doesn't have a CUDA GPU
            "sampling_method": "gradient_based",
            "tree_method": "hist",
        } 

        results_fname = (
            f"{set_name}_{bitrate}_bits_{lr:.3f}_lr_{ss:.3f}_sub_{cs:.3f}_col.csv"
        )
        results_fpath = os.path.join(results_folder, results_fname)
        
        for fold_no, cv_test_indices in fold_dict.items():
            
            cv_model, rmse, mae, cv_test = train_xgb_model(
                input_data,
                train_stns,
                cv_test_indices,
                attributes,
                target,
                params,
                num_boost_rounds,
            )
            cv_rmses.append(rmse)
            cv_maes.append(mae)

        cv_mean_rmse, cv_std_rmse = np.mean(cv_rmses), np.std(cv_rmses)
        cv_mean_mae, cv_std_mae = np.mean(cv_maes), np.std(cv_maes)

        results_dict = {
            "trial_num": trial,
            "test_mae": cv_mean_mae,
            "test_rmse": cv_mean_rmse,
            "mae_stdev": cv_std_mae,
            "rmse_stdev": cv_std_rmse,
        }
        results_cols = list(results_dict.keys())
        results_dict.update(params)

        all_results.append(results_dict)
        if (trial > 0) & (trial % 20 == 0):
            print(f"   completed {trial}/{n_optimization_rounds}")

    # save the trial results
    trial_results = pd.DataFrame(all_results)
    trial_results.to_csv(results_fpath)
    trial_mean_rmse = trial_results["test_rmse"].mean()
    trial_stdev_rmse = trial_results["rmse_stdev"].mean()
    trial_mean_mae = trial_results["test_mae"].mean()
    trial_stdev_mae = trial_results["mae_stdev"].mean()
    
    param_cols = list(params.keys())

    # get the optimal hyperparameters
    optimal_rmse_idx = trial_results["test_rmse"].idxmin()
    optimal_mae_idx = trial_results["test_mae"].idxmin()

    best_rmse_params = trial_results.loc[optimal_rmse_idx, param_cols].to_dict()
    best_mae_params = trial_results.loc[optimal_mae_idx, param_cols].to_dict()

    if loss.endswith('squarederror'):
        print(
            f"    {trial_mean_rmse:.2f} ± {trial_stdev_rmse:.3f} mean RMSE on the (held-out) test set ({len(trial_results)} hyperparameter optimization rounds.)"
        )
        best_params = best_rmse_params
    elif loss.endswith('absoluteerror'):
        print(
            f"    {trial_mean_mae:.2f} ± {trial_stdev_mae:.3f} mean MAE on the (held-out) test set ({len(trial_results)} hyperparameter optimization rounds.)"
        )
        best_params = best_mae_params

    final_model, rmse, mae, test_results = train_xgb_model(
        input_data,
        all_training_stations,
        test_stations,
        attributes,
        target,
        best_params,
        2 * num_boost_rounds,
    )

    return trial_results, test_results


def run_binary_xgb_trials_custom_CV(
    bitrate,
    set_name,
    attributes,
    target,
    input_data,
    n_optimization_rounds,
    nfolds,
    num_boost_rounds,
    results_folder,
    loss='binary:hinge',
    eval_metric="error",
):
    """
    Custom CV refers to cross validation.  Custom cross validation means the 
    held-out set must be determined in a more robust way to avoid "data leakage".
    That is, the pairs making up the training, validation, and test sets must 
    be made up of pairings from unique sets of stations.  In addition, the distribution
    of the target variable should be similar between training and test sets.
    """
    # select random hyperparameters for n_optimization_rounds
    sample_choices = np.arange(0.5, 0.9, 0.02)  # subsample and colsample percentages
    lr_choices = np.arange(0.001, 0.1, 0.0005)  # learning rates
    learning_rates = np.random.choice(lr_choices, n_optimization_rounds)
    subsamples = np.random.choice(sample_choices, n_optimization_rounds)
    colsamples = np.random.choice(sample_choices, n_optimization_rounds)
    num_boost_rounds = num_boost_rounds

    all_training_stations = np.array([item for sublist in train_stn_cv_sets for item in sublist])

    # compute train-test split before optimization rounds to keep sets constant across trials
    X_input, Y_input = (
        input_data.loc[:, features].values,
        input_data.loc[:, target].values,
    )
    # Bin the continuous target variable into quantiles
    y_binned = pd.qcut(Y_input, q=20, labels=False)  # Quantile-based discretization

    # split the data into folds such that the target
    # distribution is similar across each fold
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=int(random_seed))
    
    all_results = []
    best_result = (None, np.inf, None)
    best_params = None
    best_mean_test_perf = np.inf
    best_convergence_df = pd.DataFrame()
    best_trial_test_predictions = None
    output_target_cdfs = None
    for trial in range(n_optimization_rounds):
        lr, ss, cs = learning_rates[trial], subsamples[trial], colsamples[trial]

        params = {
            "objective": loss,
            "eval_metric": eval_metric,
            "eta": lr,
            # "n_estimators": num_boost_rounds,
            # "max_depth": 6,  # use default max_depth
            # "min_child_weight": 1, # use colsample and subsample instead of min_child_weight
            "subsample": ss,
            "colsample_bytree": cs,
            "seed": 42,
            "device": "cuda",  # note, change this to 'cpu' if your system doesn't have a CUDA GPU
            "sampling_method": "gradient_based",
            "tree_method": "hist",
        } 

        # we need to manually do CV because we're separating by stations
        # to prevent data leakage across training rounds
        fold_scores = []
        n_samples = len(X_input)
        fold_no = 0
        best_train_fold_perf, best_test_fold_perf, best_test_rounds = [], [], []
        all_fold_results = []
        fold_scores, fold_arrays = [], []
        target_cdfs = []
        for train_index, test_index in skf.split(np.zeros(n_samples), y_binned):
            
            train_stns = [e for e in all_training_stations if e not in cv_test_stns]

            assert len(np.intersect1d(train_stns, cv_test_stns)) == 0

            cv_model, cv_test = train_binary_xgb_model(
                input_data,
                train_stns,
                cv_test_stns,
                attributes,
                target,
                params,
                num_boost_rounds,
            )
            obs, pred = cv_test['actual'].values, cv_test['predicted'].values
            obs_set, obs_counts = np.unique(obs, return_counts=True)
            if (obs == pred).all() & (len(obs_set) == 1):
                # print('    All observations have the same class.')
                accuracy = 1.0
            else:
                tn, fp, fn, tp = confusion_matrix(obs, pred).ravel()
                accuracy = (tp + tn) / (tp + fp + fn + tn) 
            
            # error = fbeta_score(obs, pred, 1)
            # print(f'Accuracy: {accuracy:.2f}')#, F1 beta: {error:.2f}')
            cv_accuracies.append(accuracy)

        cv_mean, cv_std = np.mean(cv_accuracies), np.std(cv_accuracies)

        results_dict = {
            "trian_num": trial,
            "trial_accuracy": cv_mean,
            "trial_accuracy_stdev": cv_std,
        }
        results_cols = list(results_dict.keys())
        results_dict.update(params)

        all_results.append(results_dict)
        if (trial > 0) & (trial % 20 == 0):
            print(f"   completed {trial}/{n_optimization_rounds}")

    # save the trial results
    trial_results = pd.DataFrame(all_results)
    # trial_results.to_csv(results_fpath)
    trial_mean = trial_results["trial_accuracy"].mean()
    trial_stdev = trial_results["trial_accuracy_stdev"].mean()

    print(
        f"    {trial_mean:.2f} ± {trial_stdev:.3f} mean accuracy over ({len(trial_results)}x{nfolds}-fold CV hyperparameter optimization rounds.)"
    )

    param_cols = list(params.keys())

    # get the optimal hyperparameters
    # for accuracy, we want the maximum value
    optimal_accuracy_idx = trial_results["trial_accuracy"].idxmax()
    best_params = trial_results.loc[optimal_accuracy_idx, param_cols].to_dict()

    final_model, test_results = train_binary_xgb_model(
                input_data,
                all_training_stations,
                test_stations,
                attributes,
                target,
                best_params,
                2 * num_boost_rounds,
            )

    lr_final, ss_final, cs_final = best_params['eta'], best_params['subsample'], best_params['colsample_bytree']
    results_fname = f"{set_name}_{bitrate}_bits_{lr_final:.3f}_lr_{ss_final:.3f}_sub_{cs_final:.3f}_col.json"
    results_fpath = os.path.join(results_folder, results_fname)
    test_results.to_csv(results_fpath)

    return trial_results, test_results


def train_test_split(df, holdout_pct, random_seed=42):
    """
    Split the input data into training and test sets.
    The proportion of test data is holdout_pct.
    Return the data as arrays.
    """
    n_holdout = int(holdout_pct * len(df))
    np.random.seed(random_seed)
    test_idxs = np.random.choice(df.index.values, n_holdout, replace=False)
    train_idxs = [i for i in df.index.values if i not in test_idxs]

    common_idxs = np.intersect1d(train_idxs, test_idxs)
    assert len(common_idxs) == 0

    training_stns = df.loc[train_idxs, 'official_id'].values
    test_stns = df.loc[test_idxs, 'official_id'].values

    return training_stns, test_stns


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
    yvals = np.arange(1, len(sorted_data) + 1) / float(len(sorted_data))
    return sorted_data, yvals


def compute_mean_runoff(row):
    """
    Retrieves daily average flow time series and computes mean runoff for months
    with fewer than 5 missing days.

    Parameters
    ----------
    row : pd.Series
        A pandas Series containing information about the station, including 'official_id'.

    Returns
    -------
    float
        The mean unit runoff.

    """
    # Retrieve necessary data
    stn_id = row["official_id"]
    drainage_area = row['drainage_area_km2']
    df = get_timeseries_data(stn_id)
    
    # Convert flow to mm/day from m³/s based on drainage area
    df[stn_id] = (86.4 / drainage_area) * df[stn_id]
    
    # Resample the time series to daily frequency, filling missing dates
    df = df.set_index('time').resample('D').asfreq()
    
    # Add year and month columns
    df['year'] = df.index.year
    df['month'] = df.index.month
    
    # Drop rows where the target column is NaN
    df.dropna(subset=[stn_id], inplace=True)
    
    # Group by year and month and check for months with fewer than 5 missing days
    monthly_data = df.groupby(['year', 'month'])[stn_id].agg(['count', 'mean'])
    days_in_month = df.groupby(['year', 'month']).size()
    
    # Boolean mask: check for months with fewer than 5 missing days
    complete_months = (days_in_month - monthly_data['count']) < 5
    
    # Filter the months and compute the mean runoff
    filtered_mean = monthly_data.loc[complete_months, 'mean']
    
    return filtered_mean.mean()


def check_if_nested(proxy_data, target_data):
    """
    Create an attribute to identify when
    catchments are nested.
    Returns:
        0: No intersection
        1: The donor/proxy station is downstream of the target
        2: The donor/proxy station is upstream of the target.
    """
    proxy, target = proxy_data["geometry"], target_data["geometry"]
    
    pid = proxy_data["official_id"]
    tid = target_data["official_id"]
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


def uniform_log_bins(data, station, bitrate, epsilon=1e-9):
    """
    Creates uniform bins in log space for quantizing time series data.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the time series data.
    station : object
        An object representing the station, with an attribute `obs_label` that specifies the observed data column name.
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
    >>> station = lambda: None
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
    min_log_val = np.log10(data[station.obs_label].min())
    max_log_val = np.log10(data[station.obs_label].max())

    if min_log_val == max_log_val:
        raise Exception("Min. and max. log values should not be equal.")
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


def error_adjusted_fractional_bin_counts(
    observations, bin_edges, bitrate, error_factor=0.1
):
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

    # drop nan values
    values = observations[~np.isnan(observations)].values

    lower_bounds = values * (1 - error_factor)
    upper_bounds = values * (1 + error_factor)

    # Compute bin widths
    value_widths = upper_bounds - lower_bounds

    # Initialize the bin counts
    bin_counts = np.zeros(2**bitrate)

    # Create a matrix of bin edges
    bin_edges_lower = bin_edges[:-1]
    bin_edges_upper = bin_edges[1:]

    # Compute the amount each value +/- error overlaps the bins
    # np.minimum(upper_bounds[:, None], bin_edges_upper)
    #    ---> takes the smaller of the upper bin edge and the upper bound of the value range
    # np.maximum(lower_bounds[:, None], bin_edges_lower)
    #    ---> takes the larger of the lower bin edge and the lower bound of the value range
    overlap_matrix = np.maximum(
        0,
        np.minimum(upper_bounds[:, None], bin_edges_upper)
        - np.maximum(lower_bounds[:, None], bin_edges_lower),
    )

    # handle cases where bounds are outside bin ranges
    min_bin_edge, max_bin_edge = bin_edges_lower[0], bin_edges_upper[-1]

    # Extend the overlap matrix to include virtual bins
    extended_overlap_matrix = np.zeros((len(values), 2**bitrate))

    # Copy the original overlaps into the extended matrix
    extended_overlap_matrix[:, 1:-1] = overlap_matrix
    
    # Vectorized handling of out-of-bounds lower bounds
    out_of_bounds_lower = lower_bounds < min_bin_edge
    extended_overlap_matrix[out_of_bounds_lower, 0] = min_bin_edge - lower_bounds[out_of_bounds_lower]

    # Vectorized handling of out-of-bounds upper bounds
    out_of_bounds_upper = upper_bounds > max_bin_edge
    extended_overlap_matrix[out_of_bounds_upper, -1] = upper_bounds[out_of_bounds_upper] - max_bin_edge

    # Vectorized handling of completely out-of-bounds cases
    completely_out_of_bounds_left = upper_bounds < min_bin_edge
    completely_out_of_bounds_right = lower_bounds > max_bin_edge
    
    # Set the adjusted widths to the value widths for completely out-of-bounds cases
    extended_overlap_matrix[completely_out_of_bounds_left, 0] = value_widths[completely_out_of_bounds_left]
    extended_overlap_matrix[completely_out_of_bounds_right, -1] = value_widths[completely_out_of_bounds_right]

    # assert that the overlap matrix sums to the value bounds along rows
    assert np.allclose(
        np.sum(extended_overlap_matrix, axis=1), value_widths, atol=1e-2
    ), f"overlaps dont add up to value width"

    # Fractional counts based on the overlap
    fractional_counts = extended_overlap_matrix / value_widths[:, None]

    # Sum fractional counts for each bin
    bin_counts = np.sum(fractional_counts, axis=0)

    # find where fractional counts is greater than zero in the left-most bin
    assert np.allclose(
        np.sum(fractional_counts, axis=1), 1, atol=1e-2
    ), f"Fractional counts don't sum to 1."

    assert round(sum(bin_counts), 0) == len(
        values
    ), f"Error in bin counts: {sum(bin_counts)} != {len(values)}"

    return bin_counts


def compute_unadjusted_counts(df, target, bin_edges, bitrate, concurrent_data):
    try:
        # note that np.digitize is 1-indexed
        df[target.obs_quantized_label] = np.digitize(
            df[target.obs_label], bin_edges
        )  # P(X)
        df[target.sim_quantized_label] = np.digitize(
            df[target.sim_label], bin_edges
        )  # Q(X)
    except Exception as e:
        print(f"Error digitizing series: {e}")
        raise Exception("Error digitizing series")
    # count the occurrences of each quantized value
    # the "simulated" series is the proxy/donor series
    # and the "observed" series is the target location
    obs_count_df = df.groupby(target.obs_quantized_label).count()
    sim_count_df = df.groupby(target.sim_quantized_label).count()
    # og_sim_count = sim_count_df[target.sim_label].copy().sum()

    count_df = pd.DataFrame(index=range(2**bitrate))
    count_df[target.obs_label] = 0
    count_df[target.sim_label] = 0

    count_df[target.obs_label] += obs_count_df[target.obs_label]
    count_df[target.sim_label] += sim_count_df[target.sim_label]
    count_df.fillna(0, inplace=True)

    return count_df


def test_probability_distribution_sums_to_one(p, bitrate):
    # check for rounding errors, p should sum to 1
    # if not, renormalize the probabilities
    sum_p = np.sum(p)
    if round(sum_p, 2) != 1.0:
        diff = 1.0 - sum_p
        p += diff / 2**bitrate
        new_sum = round(np.sum(p), 2)
        
        if new_sum != 1.0:
            raise Exception(f"Renormalization of probabilities failed: {new_sum}")
    return p


def compute_posterior_Q_probabilities(
    count_df, target, bitrate, pseudo_counts, concurrent_data, bin_edges
):
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
    - p_obs is the observed (target) frequencies P(X)
    - p_sim is the simulated (proxy/donor) frequencies Q(X)

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
    n_obs = np.nansum(count_df[target.obs_label])
    n_sim = np.nansum(count_df[target.sim_label])

    if concurrent_data:
        assert round(n_obs, 0) == round(n_sim, 0), f"Number of observations and simulations do not match. n_obs={n_obs}, n_sim={n_sim}"

    # normalize p_obs and p_sim
    p_obs = count_df[target.obs_label].values / n_obs
    p_sim = count_df[target.sim_label].values / n_sim
    assert round(np.sum(p_sim), 2) == 1.0, "p_sim does not sum to 1.0"

    # create a dataframe to store the posterior Q after assuming different priors
    q_df = pd.DataFrame()
    q_df["q_sim_no_prior"] = p_sim

    uniform_p = [1.0 / 2.0**bitrate for _ in range(2**bitrate)]
    q_df["q_uniform"] = test_probability_distribution_sums_to_one(uniform_p, bitrate)

    # compute the posterior probabilities based on
    # a wide range of priors to test sensitivity
    for pseudo_counts in pseudo_counts:
        adjusted_counts = [x + 10**pseudo_counts for x in count_df[target.sim_label]]
        tot_adjusted_counts = np.nansum(adjusted_counts)
        q_df[f"q_post_{pseudo_counts}R"] = adjusted_counts / tot_adjusted_counts
        assert (
            np.round(q_df[f"q_post_{pseudo_counts}R"].sum(), 5) == 1
        ), "Posterior probabilities do not sum to 1."

    return p_obs, q_df


def process_probabilities(
    df, proxy, target, bitrate, concurrent_data, partial_counts, pseudo_counts
):
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

    # if partial_counts == False:
        # computes the observed P and simulation Q distribution probabilities
        # as dicts by bin number, probability key-value pairs
        # test a wide range of uniform priors via pseudo counts
    count_df = compute_unadjusted_counts(
        df, target, bin_edges, bitrate, concurrent_data
    )
    # else:
    # add a uniformly distributed error to the observed data
    # and compute probabilities from partial observation counts
    # where counts are divided based on the proportion of the bin
    # that the measurement error falls within
    fractional_obs_counts = error_adjusted_fractional_bin_counts(
        df[target.obs_label], np.array(bin_edges), bitrate, error_factor=0.1
    )
    fractional_sim_counts = error_adjusted_fractional_bin_counts(
        df[target.sim_label], np.array(bin_edges), bitrate, error_factor=0.1
    )

    count_df = pd.DataFrame(index=range(2**bitrate))
    count_df[target.obs_label] = 0
    count_df[target.sim_label] = 0
    count_df[target.obs_label] += fractional_obs_counts
    count_df[target.sim_label] += fractional_sim_counts
    count_df.fillna(0, inplace=True)

    # compute posteriors for a rang of priors (pseudo-counts)
    p_obs, p_sim = compute_posterior_Q_probabilities(
        count_df, target, bitrate, pseudo_counts, concurrent_data, bin_edges
    )
    # the prior adds some amount of noise to the distribution
    # we should compute this noise (via the KL divergence 
    # between the model q and the posterior r
    
    # set a flag where the simulation counts zero in any state
    # where the a posteriori observation counts are > 0.
    # it's these cases where the prior should have the greatest 
    # influence on the KL divergence.
    underspecified_flag = (
        (count_df[f'{target.official_id}_sim'] == 0) &   # the simulated series can be out of range of the target,
        (count_df[target.official_id] > 0)
    ).any()
    
    return p_obs, p_sim, bin_edges, underspecified_flag


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
        print(f"Linear regression failed on {obs.id} and {sim.id} (N={len(df)})")
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
    return (
        1
        - (df[sim.id] - df[obs.id]).pow(2).sum() / (df[obs.id] - obs_mean).pow(2).sum()
    )


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
    r = np.corrcoef(df[obs.id], df[sim.id])[0, 1]
    beta = sim_mean / obs_mean
    gamma = (sim_std / sim_mean) / (obs_std / obs_mean)
    return 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)


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
        - attr_cols (list): list of attributes (features)
        - climate_cols (list): climate attribute columns
        - pseudo_counts (list): array of integers representing prior distribution pseudo-counts

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
    (
        proxy,
        target,
        bitrate,
        error_df,
        min_concurrent_years,
        partial_counts,
        attr_cols,
        climate_cols,
        pseudo_counts,
    ) = inputs

    proxy_id, target_id = proxy['official_id'], target['official_id']
    bitrate = int(bitrate)

    # create a result dict object for tracking results of the batch comparison
    result = {
        "proxy": proxy_id,
        "target": target_id,
        "bitrate": bitrate,
        "min_concurrent_years": min_concurrent_years,
    }

    station_info = {"proxy": proxy, "target": target}

    # check if the polygons are nested
    result["nested_catchments"] = check_if_nested(
        proxy, target
    )

    # for stn in pair:
    proxy = Station(station_info["proxy"], bitrate)
    target = Station(station_info["target"], bitrate)

    # compute spatial distance
    p1, p2 = (
        station_info["proxy"]["geometry"].centroid,
        station_info["target"]["geometry"].centroid,
    )
    # compute the distance between catchment centroids (km)
    centroid_distance = p1.distance(p2) / 1000
    result["centroid_distance"] = round(centroid_distance, 2)
    if centroid_distance > 1000:
        return None

    if np.isnan(target.drainage_area_km2):
        raise ValueError(f"No drainage area for {target_id}")
    if np.isnan(proxy.drainage_area_km2):
        raise ValueError(f"No drainage area for {proxy_id}")

    # Retrieve the data for both stations
    # this is all data, including non-concurrent
    adf = retrieve_nonconcurrent_data(proxy_id, target_id)

    assert ~adf.empty, "No data returned."

    for stn in [proxy, target]:
        adf = transform_and_jitter(adf, stn)

    # simulate flow at the target based on equal unit area runoff scaling
    adf[target.sim_label] = adf[proxy.id] * (
        target.drainage_area_km2 / proxy.drainage_area_km2
    )

    # filter for the concurrent data
    df = adf.copy().dropna(subset=[proxy_id, target_id], how="any")
    result["num_concurrent_obs"] = len(df)
    
    if df.empty:
        num_complete_concurrent_years = 0
    else:
        df.reset_index(inplace=True)
        num_complete_concurrent_years = count_complete_years(df, 'time', proxy_id)
        
    counts = df[[proxy_id, target_id]].count(axis=0)
    counts = adf.count(axis=0)
    proxy.n_obs, target.n_obs = counts[proxy_id], counts[target_id]
    result[f"proxy_n_obs"] = proxy.n_obs
    result[f"target_n_obs"] = target.n_obs
    result[f"proxy_frac_concurrent"] = len(df) / proxy.n_obs
    result[f"target_frac_concurrent"] = len(df) / target.n_obs

    if (counts[proxy_id] == 0) or (counts[target_id] == 0):
        print(f"   Zero observations.  Skipping.")
        return None

    # process the PMFs and divergences for concurrent data
    # using a range of uniform priors via pseudo counts
    if num_complete_concurrent_years > min_concurrent_years:
        # compute coefficient of determination
        result["cod"] = compute_cod(df, proxy, target)

        # compute Nash-Sutcliffe efficiency
        result["nse"] = compute_nse(df, proxy, target)

        # compute the Kling-Gupta efficiency
        result["kge"] = compute_kge(df, proxy, target)

        # df is concurrent data, so the results
        # are updating concurrent data here
        # df, proxy, target, bitrate, concurrent_data, partial_counts, pseudo_counts
        concurrent_data = True
        p_obs, p_sim, bin_edges, underspecified_flag = process_probabilities(
            df, proxy, target, bitrate, concurrent_data, partial_counts, pseudo_counts
        )
        
        result = process_divergences(
            result, p_obs, p_sim, bin_edges, bitrate, concurrent_data
        )

    if (target.n_obs > 365 * 0.9) & (proxy.n_obs > 365 * 0.9):
        # adf is all data (includes non-concurrent), so the results
        # are updated if both series meet the minimum length
        concurrent_data = False
        p_obs, p_sim, bin_edges, underspecified_flag = process_probabilities(
            adf, proxy, target, bitrate, concurrent_data, partial_counts, pseudo_counts
        )
        result = process_divergences(
            result, p_obs, p_sim, bin_edges, bitrate, concurrent_data
        )
    result['underspecified_model_flag'] = underspecified_flag
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

    msg = f"{c}: sum(p)={sum_p:.2f} sum(q)={sum_q:.2f}"
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
    df["bin"] = range(1, 2**bitrate + 1)
    df.set_index("bin", inplace=True)

    for c in p_sim.columns:
        if c == "q_sim_no_prior":
            continue

        # explicitly set data types before vectorization
        p = np.array(p_obs, dtype=np.float64)
        q = np.array(p_sim["q_sim_no_prior"].values, dtype=np.float64)
        r = np.array(p_sim[c].values, dtype=np.float64) # the posterior
        
        # Raise error if any r[i] == 0, as log(p/r) is undefined for those cases
        if np.any(r == 0):
            raise ValueError("Posterior R contains zero entries, which is not allowed.")

        # ensure that the probabilities sum to 1
        check_distribution(p, r, c)

        label = "dkl_nonconcurrent_" + "_".join(c.split("_")[1:])        
        if concurrent_data is True:
            label = "dkl_concurrent_" + "_".join(c.split("_")[1:])

        # compute the distortion due to the prior assumed on Q
        mask = (p > 0) & (r > 0)
        kld_array = np.zeros_like(p)
        kld_array[mask] = p[mask] * np.log2(p[mask] / r[mask])
        df[label] = kld_array
            
        # compute DKL(Q||R), or the distortion of Q 
        # due to the assumed prior
        mask = (q > 0) & (r > 0)
        distortion = np.zeros_like(q)
        distortion[mask] = q[mask] * np.log2(q[mask] / r[mask])
        label = "dkl_prior_distortion_" + "_".join(c.split("_")[1:])
        df[label] = distortion

    sum_dkl = df.sum()

    if any(sum_dkl.values) <= 0:
        print(f"negative or zero dkl")
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
    tvd_label = f"tvd_nonconcurrent"
    if concurrent_data is True:
        tvd_label = f"tvd_concurrent"
    results[tvd_label] = np.sum(np.abs(np.subtract(p, q))) / 2
    results[tvd_label + "_max"] = np.sum(np.abs(np.subtract(p, q_uniform))) / 2
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
    wasserstein_label = f"wasserstein_nonconcurrent"
    if concurrent_data is True:
        wasserstein_label = f"wasserstein_concurrent"

    # Compute the bin midpoints in the linear space
    # this represents the water volume in the bin
    bin_midpoints = list(np.add(bin_edges[:-1], bin_edges[1:]) / 2)

    # append the lowest and highest edges to the bin midpoints
    # to serve as the weight (volume) of the highest bin
    bin_midpoints = [bin_edges[0]] + bin_midpoints + [bin_edges[-1]]
    # check for monitonicity
    assert np.all(np.diff(bin_midpoints) >= 0)

    try:
        result[wasserstein_label] = wasserstein_distance(
            bin_midpoints, bin_midpoints, p, q
        )
        result[wasserstein_label + "_max"] = wasserstein_distance(
            bin_midpoints, bin_midpoints, p, q_uniform
        )
        return result
    except ValueError as e:
        print(f"Error computing Wasserstein distance: {e}")
        print(f"p: {len(p)}")
        print(p)
        print(f"q: {len(q)}")
        print(q)
        print(f"bin_midpoints: {len(bin_midpoints)}")
        print(f"bin_edges: {len(bin_edges)}")
        raise Exception("Wasserstein distance computation failed")
        # result[wasserstein_label] = np.nan
        # result[wasserstein_label + '_max'] = np.nan


def process_divergences(result, p_obs, p_sim, bin_edges, bitrate, concurrent_data):
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
    q = p_sim["q_sim_no_prior"].values
    q_uniform = p_sim["q_uniform"].values

    tvd_result = compute_tvd(p, q, q_uniform, concurrent_data)
    wd_result = compute_wasserstein_distance(
        bin_edges, p, q, q_uniform, concurrent_data
    )

    result.update(tvd_result)
    result.update(wd_result)
    result.update(dkl_by_prior.to_dict())

    return result
