import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMFLOW_DIR = os.path.join(BASE_DIR, 'hysets_streamflow_timeseries')

def import_streamflow(official_id, min_flow=1e-4):
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
    >>> df = import_streamflow('12345678')
    >>> df.head()
         date   12345678  12345678_low_flow_flag
    0  2020-01-01  0.0010                     False
    1  2020-01-02  0.0001                      True
    2  2020-01-03  0.0001                      True
    3  2020-01-04  0.0025                     False
    4  2020-01-05  0.0001                      True
    """
    fpath = os.path.join(STREAMFLOW_DIR, f'{official_id}.csv')
    df = pd.read_csv(fpath, engine='pyarrow', )
    df[f'{official_id}_low_flow_flag'] = df['discharge'] < min_flow    
    
    df['discharge'] = df['discharge'].clip(lower=min_flow)
    # rename the discharge column to the station id    
    df.rename(columns={'discharge': official_id}, inplace=True)
    return df


def retrieve_nonconcurrent_data(proxy, target):
    df1 = get_timeseries_data(proxy).set_index('time')
    df2 = get_timeseries_data(target).set_index('time')
    df = pd.concat([df1, df2], join='outer', axis=1)

    # get the year from the time index
    df['year'] = df.index.year
    return df