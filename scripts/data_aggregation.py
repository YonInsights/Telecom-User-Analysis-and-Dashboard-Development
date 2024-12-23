import pandas as pd

def aggregate_user_data(df):
    """
    Aggregates data for each user (MSISDN/Number) with specific metrics.

    Parameters:
    df (DataFrame): The input dataframe containing telecom session data.

    Returns:
    DataFrame: Aggregated data per user.
    """
    # Clean 'Handset Type' column: Replace undefined or missing values with 'Unknown'
    df['Handset Type'] = df['Handset Type'].fillna('Unknown')
    df['Handset Type'] = df['Handset Type'].replace('undefined', 'Unknown')

    # Perform the groupby and aggregation for the requested metrics
    user_data = df.groupby('MSISDN/Number').agg(
        average_rtt_dl=('Avg RTT DL (ms)', 'mean'),
        average_rtt_ul=('Avg RTT UL (ms)', 'mean'),
        average_bearer_tp_dl=('Avg Bearer TP DL (kbps)', 'mean'),
        average_bearer_tp_ul=('Avg Bearer TP UL (kbps)', 'mean'),
        average_tcp_dl_retrans=('TCP DL Retrans. Vol (Bytes)', 'mean'),
        average_tcp_ul_retrans=('TCP UL Retrans. Vol (Bytes)', 'mean'),
        handset_type=('Handset Type', pd.Series.mode)  # Using mode for categorical data
    ).reset_index()

    # Handle missing or outlier values by replacing with mean (if needed)
    user_data['average_rtt_dl'] = user_data['average_rtt_dl'].fillna(user_data['average_rtt_dl'].mean())
    user_data['average_rtt_ul'] = user_data['average_rtt_ul'].fillna(user_data['average_rtt_ul'].mean())
    user_data['average_bearer_tp_dl'] = user_data['average_bearer_tp_dl'].fillna(user_data['average_bearer_tp_dl'].mean())
    user_data['average_bearer_tp_ul'] = user_data['average_bearer_tp_ul'].fillna(user_data['average_bearer_tp_ul'].mean())
    user_data['average_tcp_dl_retrans'] = user_data['average_tcp_dl_retrans'].fillna(user_data['average_tcp_dl_retrans'].mean())
    user_data['average_tcp_ul_retrans'] = user_data['average_tcp_ul_retrans'].fillna(user_data['average_tcp_ul_retrans'].mean())

    return user_data
