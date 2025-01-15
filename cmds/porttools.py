import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def heatmap_vector(s,labs=None,Ngrid=None,plottitle=None):
    if labs is None:
        labs = s.index

    if Ngrid is None:
        Ngrid = np.sqrt(len(s))

    target_rows = Ngrid
    target_cols = Ngrid
    target_size = target_rows * target_cols

    n_missing = target_size - len(s)
    if n_missing < 0:
        raise ValueError("Series is larger than target shape!")

    values_padded = pd.concat([s, pd.Series([np.nan] * n_missing)], ignore_index=True)
    index_padded = s.index.tolist() + [None] * n_missing  # or some placeholder strings

    values_2d = values_padded.values.reshape(target_rows, target_cols)
    labels_2d = np.array(index_padded, dtype=object).reshape(target_rows, target_cols)

    df_values = pd.DataFrame(values_2d)


    plt.figure(figsize=(12, 12))
    sns.heatmap(
        df_values, 
        #cmap="viridis",
        annot_kws={"fontsize": 10},
        xticklabels=False,  # Hide horizontal axis labels
        yticklabels=False,   # Hide vertical axis labels
        annot=labels_2d,   # pass the 2D array of label strings
        fmt="",            # no special numeric format
        cbar=False          # optional color bar
    )

    if plottitle is None:
        plottitle=''
        
    plt.title(plottitle)
    plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def scatterplot_outliers_iqr(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    k: float = 1.5,
    figsize=(6, 6),
    title: str = "",
    x_label: str = None,
    y_label: str = None
):
    """
    Plots a scatterplot of df[col_x] vs. df[col_y], labeling outliers in either direction.
    Outliers are determined by the IQR method with multiplier k.
    
    :param df: DataFrame containing the data.
    :param col_x: Column name for the x-axis.
    :param col_y: Column name for the y-axis.
    :param k: The multiplier for the IQR method. (Default 1.5)
    :param figsize: Tuple for figure size. (Default (6,6))
    :param title: Title of the plot. (Default "Scatterplot with Outliers Labeled")
    :param x_label: Label for the x-axis. (Default None -> use col_x)
    :param y_label: Label for the y-axis. (Default None -> use col_y)
    """

    def iqr_bounds(series, k_value):
        """Returns (lower_bound, upper_bound) for outlier detection based on IQR."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k_value * iqr
        upper = q3 + k_value * iqr
        return lower, upper

    # Compute bounds for x and y using IQR
    x_lower, x_upper = iqr_bounds(df[col_x], k)
    y_lower, y_upper = iqr_bounds(df[col_y], k)

    # Identify outliers (in either col_x or col_y)
    mask_outliers = (
        (df[col_x] < x_lower) | (df[col_x] > x_upper) |
        (df[col_y] < y_lower) | (df[col_y] > y_upper)
    )
    outliers = df[mask_outliers]

    # Plot
    plt.figure(figsize=figsize)
    plt.scatter(df[col_x], df[col_y], color="blue", label="All points")
    plt.scatter(outliers[col_x], outliers[col_y], color="red", label="Outliers")

    # Annotate outliers with index
    for idx, row in outliers.iterrows():
        plt.annotate(
            text=idx,
            xy=(row[col_x], row[col_y]),
            xytext=(5, 5),  # offset so labels don't overlap dots
            textcoords="offset points",
            fontsize=8
        )

    plt.xlabel(x_label if x_label else col_x)
    plt.ylabel(y_label if y_label else col_y)
    plt.title(title)
    #plt.legend()
    plt.show()
