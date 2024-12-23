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








