
import pandas as pd
def calculate_throughput(dataframe):
    """
    Calculate Total Throughput (kbps) by summing Avg Bearer TP DL (kbps) and Avg Bearer TP UL (kbps).

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame containing throughput columns.

    Returns:
        pd.DataFrame: DataFrame with an added 'Total Throughput (kbps)' column.
    """
    dataframe['Total Throughput (kbps)'] = dataframe['Avg Bearer TP DL (kbps)'] + dataframe['Avg Bearer TP UL (kbps)']
    return dataframe
def calculate_latency_qoe(dataframe):
    """
    Calculate Overall Average RTT (ms) using Avg RTT DL (ms) and Avg RTT UL (ms).
    Handles missing values by imputing with the column median.

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame containing RTT columns.

    Returns:
        pd.DataFrame: DataFrame with an added 'Overall Avg RTT (ms)' column.
    """
    # Impute missing values with column median
    dataframe['Avg RTT DL (ms)'] = dataframe['Avg RTT DL (ms)'].fillna(dataframe['Avg RTT DL (ms)'].median())
    dataframe['Avg RTT UL (ms)'] = dataframe['Avg RTT UL (ms)'].fillna(dataframe['Avg RTT UL (ms)'].median())
    
    # Calculate Overall Average RTT
    dataframe['Overall Avg RTT (ms)'] = (dataframe['Avg RTT DL (ms)'] + dataframe['Avg RTT UL (ms)']) / 2
    return dataframe
def categorize_qoe(dataframe):
    """
    Categorize QoE based on Overall Avg RTT (ms).

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame containing 'Overall Avg RTT (ms)' column.

    Returns:
        pd.DataFrame: DataFrame with an added 'QoE Category' column.
    """
    conditions = [
        dataframe['Overall Avg RTT (ms)'] <= 50,
        (dataframe['Overall Avg RTT (ms)'] > 50) & (dataframe['Overall Avg RTT (ms)'] <= 100),
        (dataframe['Overall Avg RTT (ms)'] > 100) & (dataframe['Overall Avg RTT (ms)'] <= 200),
        dataframe['Overall Avg RTT (ms)'] > 200
    ]
    categories = ['Excellent', 'Good', 'Fair', 'Poor']
    
    # Assign categories
    dataframe['QoE Category'] = pd.cut(
        dataframe['Overall Avg RTT (ms)'], 
        bins=[-float('inf'), 50, 100, 200, float('inf')], 
        labels=categories
    )
    return dataframe
