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
