import os
import pandas as pd

def aggregate_user_data(df):
    """
    Aggregates data for each user (MSISDN/Number).
    
    Parameters:
    df (DataFrame): The input dataframe containing telecom session data.
    
    Returns:
    DataFrame: Aggregated data per user.
    """
    # Perform the groupby and aggregation
    user_data = df.groupby('MSISDN/Number').agg(
        number_of_sessions=('MSISDN/Number', 'size'),
        total_duration=('Dur. (ms)', 'sum'),
        total_dl_data=('Total DL (Bytes)', 'sum'),
        total_ul_data=('Total UL (Bytes)', 'sum')
    ).reset_index()
    
    # Calculate total_data_volume after aggregation
    user_data['total_data_volume'] = user_data['total_dl_data'] + user_data['total_ul_data']
    
    return user_data



