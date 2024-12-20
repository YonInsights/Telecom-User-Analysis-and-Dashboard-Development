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

# Example usage:
# df = pd.read_csv("path_to_your_csv_file.csv")
# numeric_cols = df.select_dtypes(include='number').columns
# plot_histograms(df, numeric_cols)
