import os
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load your data (assuming you have a DataFrame named 'data_for_bivariate')
# data_for_bivariate = pd.read_csv("your_dataset.csv")


def plot_histograms(df, numeric_cols, cols_per_row=5, rows_per_fig=10):
    """
    Plots histograms for each numeric column in the DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric columns in the DataFrame.
    cols_per_row (int): Number of columns per row in the grid.
    rows_per_fig (int): Number of rows per figure.
    """
    plots_per_fig = cols_per_row * rows_per_fig
    num_cols = len(numeric_cols)
    num_figs = math.ceil(num_cols / plots_per_fig)

    for fig_idx in range(num_figs):
        start_idx = fig_idx * plots_per_fig
        end_idx = min(start_idx + plots_per_fig, num_cols)
        cols_to_plot = numeric_cols[start_idx:end_idx]
        num_plots = len(cols_to_plot)
        num_rows = math.ceil(num_plots / cols_per_row)  # Number of rows needed

        plt.figure(figsize=(cols_per_row * 4, num_rows * 4))  # Adjust figure size
        for i, col in enumerate(cols_to_plot, 1):
            plt.subplot(num_rows, cols_per_row, i)
            df[col].hist(bins=30, edgecolor='black')
            plt.title(col)
            plt.tight_layout()

        plt.show()

def plot_boxplots(df, numeric_cols):
    """
    Plots box plots for each numeric column in the DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric columns in the DataFrame.
    """
    num_cols = len(numeric_cols)
    num_rows = math.ceil(num_cols / 5)

    plt.figure(figsize=(20, num_rows * 4))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(num_rows, 5, i)
        df.boxplot(column=col)
        plt.title(col)
        plt.tight_layout()

    plt.show()

def plot_total_data_volume(user_data, top_n=10):
    """
    Plots a bar chart of total data volume for each user (MSISDN/Number).
    
    Parameters:
    - user_data (DataFrame): The aggregated data per user.
    - top_n (int): The number of top users to plot. Default is 10.
    """
    # Sort the data based on total_data_volume in descending order and select top_n rows
    top_users = user_data.sort_values('total_data_volume', ascending=False).head(top_n)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(top_users['MSISDN/Number'].astype(str), top_users['total_data_volume'], color='skyblue')

    # Add labels and title
    plt.xlabel('MSISDN/Number')
    plt.ylabel('Total Data Volume (Bytes)')
    plt.title(f'Top {top_n} Users with Highest Total Data Volume')
    plt.xticks(rotation=90)  # Rotate the MSISDN/Number labels for better readability

    # Add the data values on top of each bar
    for i, bar in enumerate(plt.gca().patches):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, 
                 f'{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=9)

    # Show the plot
    plt.tight_layout()
    plt.show()

def filter_top_users(df, metric, top_n=10):
    """
    Filters the top N users based on a given metric.
    :param df: DataFrame containing user data.
    :param metric: Column name to sort and filter by.
    :param top_n: Number of top users to keep.
    :return: Filtered DataFrame.
    """
    return df.nlargest(top_n, metric)

def plot_xdr_sessions_top(df):
    """Visualize the number of xDR sessions for the top users."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='MSISDN/Number', y='Number of xDR Sessions', data=df)
    plt.xticks(rotation=45)
    plt.title('Number of xDR Sessions (Top Users)')
    plt.xlabel('MSISDN/Number')
    plt.ylabel('Number of xDR Sessions')
    plt.show()

def plot_total_dl_data_top(df):
    """Visualize the total download data (DL) for the top users."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='MSISDN/Number', y='Total DL Data (Bytes)', data=df)
    plt.xticks(rotation=45)
    plt.title('Total Download Data (Top Users)')
    plt.xlabel('MSISDN/Number')
    plt.ylabel('Total DL Data (Bytes)')
    plt.show()

def plot_total_ul_data_top(df):
    """Visualize the total upload data (UL) for the top users."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='MSISDN/Number', y='Total UL Data (Bytes)', data=df)
    plt.xticks(rotation=45)
    plt.title('Total Upload Data (Top Users)')
    plt.xlabel('MSISDN/Number')
    plt.ylabel('Total UL Data (Bytes)')
    plt.show()

def plot_data_volume_vs_duration_top(df):
    """Visualize the relationship between total data volume and session duration for top users."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Total Data Volume (Bytes)', y='Total Session Duration (ms)', data=df)
    plt.title('Total Data Volume vs Session Duration (Top Users)')
    plt.xlabel('Total Data Volume (Bytes)')
    plt.ylabel('Total Session Duration (ms)')
    plt.show()
def generate_graphical_univariate_analysis(data):
    # List of quantitative variables in the dataset
    quantitative_vars = data.columns

    # Set up the plotting grid
    fig, axes = plt.subplots(len(quantitative_vars), 3, figsize=(18, 5 * len(quantitative_vars)))
    fig.suptitle('Graphical Univariate Analysis', fontsize=16, y=0.92)

    # Loop through each variable and create the plots
    for i, var in enumerate(quantitative_vars):
        plot_histogram(data, var, axes[i, 0])
        plot_boxplot(data, var, axes[i, 1])
        plot_density(data, var, axes[i, 2])

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # Function to plot histogram
def plot_histogram(data, var, ax):
    sns.histplot(data[var], bins=30, kde=False, ax=ax, color='skyblue')
    ax.set_title(f'{var} - Histogram')
    ax.set_xlabel(var)
    ax.set_ylabel('Frequency')

# Function to plot boxplot
def plot_boxplot(data, var, ax):
    sns.boxplot(x=data[var], ax=ax, color='orange')
    ax.set_title(f'{var} - Boxplot')
    ax.set_xlabel(var)

# Function to plot density plot
def plot_density(data, var, ax):
    sns.kdeplot(data[var], ax=ax, fill=True, color='green')
    ax.set_title(f'{var} - Density Plot')
    ax.set_xlabel(var)

# Main function to generate all plots
def generate_graphical_univariate_analysis(data):
    # List of quantitative variables in the dataset
    quantitative_vars = data.columns

    # Set up the plotting grid
    fig, axes = plt.subplots(len(quantitative_vars), 3, figsize=(18, 5 * len(quantitative_vars)))
    fig.suptitle('Graphical Univariate Analysis', fontsize=16, y=0.92)

    # Loop through each variable and create the plots
    for i, var in enumerate(quantitative_vars):
        plot_histogram(data, var, axes[i, 0])
        plot_boxplot(data, var, axes[i, 1])
        plot_density(data, var, axes[i, 2])

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # Function to create scatter plots
# Define the plot function (for reference)
def plot_scatter(x, y, ax, title, xlabel, ylabel):
    ax.scatter(x, y, color='blue')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

# Create a figure and axes for the subplots (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

def plot_cumulative_variance(cumulative_variance, figsize=(8, 5), marker='o', linestyle='--'):
    """
    Plots the cumulative explained variance from PCA.

    Parameters:
        cumulative_variance (list or numpy array): Array of cumulative variance values from PCA.
        figsize (tuple): Size of the plot figure.
        marker (str): Marker style for the plot.
        linestyle (str): Line style for the plot.

    Returns:
        None
    """
    plt.figure(figsize=figsize)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
             marker=marker, linestyle=linestyle)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by PCA Components')
    plt.grid(True)
    plt.show()


