import os
import matplotlib.pyplot as plt
import math

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



