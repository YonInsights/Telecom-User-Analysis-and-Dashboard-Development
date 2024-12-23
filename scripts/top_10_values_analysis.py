import pandas as pd

# Function to compute the top 10 values for a given column
def top_10_values(df, column_name):
    """
    Returns the top 10 largest values from the specified column of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to analyze.
    
    Returns:
    pd.Series: The top 10 largest values from the column.
    """
    return df[column_name].nlargest(10)
# Function to compute the bottom 10 values for a given column
def bottom_10_values(df, column_name):
    """
    Returns the bottom 10 smallest values from the specified column of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to analyze.
    
    Returns:
    pd.Series: The bottom 10 smallest values from the column.
    """
    return df[column_name].nsmallest(10)
# Function to compute the most frequent values (mode) for a given column
def most_frequent_values(df, column_name):
    """
    Returns the most frequent values (mode) from the specified column of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to analyze.
    
    Returns:
    pd.Series: The most frequent values (mode) from the column.
    """
    return df[column_name].mode()
# Function to summarize top, bottom, and mode values
def improved_summary(df, column_name):
    """
    Displays the top 10, bottom 10, and most frequent values for a given column.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to analyze.
    
    Returns:
    pd.DataFrame: A DataFrame summarizing the top 10, bottom 10, and most frequent values in a cleaner format.
    """
    top_10 = df[column_name].nlargest(10).reset_index()
    bottom_10 = df[column_name].nsmallest(10).reset_index()
    most_frequent = df[column_name].mode().reset_index()
    
    top_10 = top_10[['index', column_name]].rename(columns={'index': 'Index', column_name: 'Top 10 Values'})
    bottom_10 = bottom_10[['index', column_name]].rename(columns={'index': 'Index', column_name: 'Bottom 10 Values'})
    
    # Most frequent value(s) are shown as a single row
    most_frequent = most_frequent[['index', column_name]].rename(columns={'index': 'Index', column_name: 'Most Frequent Value(s)'})
    
    # Merge all results into one DataFrame
    result = pd.merge(top_10, bottom_10, on='Index', how='outer')
    result = pd.merge(result, most_frequent, on='Index', how='outer')
    
    return result
def improved_summary_v2(df, column_name):
    """
    Displays the top 10, bottom 10, and most frequent values for a given column,
    formatted cleanly for easier interpretation.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to analyze.
    
    Returns:
    pd.DataFrame: A DataFrame summarizing the top 10, bottom 10, and most frequent values in a cleaner format.
    """
    # Get the Top 10 and Bottom 10 values
    top_10 = df[column_name].nlargest(10).reset_index(drop=True)
    bottom_10 = df[column_name].nsmallest(10).reset_index(drop=True)
    
    # Get the Most Frequent Value(s)
    most_frequent = df[column_name].mode().reset_index(drop=True)
    
    # Create a DataFrame for the result
    summary = pd.DataFrame({
        'Top 10 Values': top_10,
        'Bottom 10 Values': bottom_10,
        'Most Frequent Value(s)': most_frequent[0]  # Only show the first mode value
    })
    
    return summary













