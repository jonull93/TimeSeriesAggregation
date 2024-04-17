from TSA.algorithms.pctpc import PCTPCAggregator
from TSA.data_importer import get_filepath
import TSA
import numpy as np
import pandas as pd
import os

"""
conda create -n fresh python=3.12
conda activate fresh
pip install "C:\...\TimeSeriesAggregator"
pip install openpyxl
python "C:\...\TimeSeriesAggregator\docs\example_usage.py"
"""

## Construct sample DataFrame 
# read Load and Temp from "C:\Users\vijulk\OneDrive - Vitecsoftware Group AB\workspace.xlsx" Sheet2
excel_path = get_filepath("example_usage.xlsx")
df = pd.read_excel(excel_path, sheet_name="Sheet2") # 4 days of hourly Load and Temp
nr_rows = len(df) # 96 rows
#df2 = pd.DataFrame(np.random.randint(0,nr_rows,size=(nr_rows, 2)), columns=list('AB')) # Free-rider columns A and B that do not affect dissimilarity calculation
#df = pd.concat([df, df2], axis=1) # Load, Temp, A, B

## Set up parameters for the PCTPC algorithm
final_length = nr_rows // 2 # Half of the initial length, in the form of an integer
priority_columns = [0] # Maintain extrema for the Load column
similarity_columns = [0, 1] # Consider Load and Temp columns for similarity calculation
similarity_weights = [1, 1] # Weights for Load and Temp columns, respectively, when calculating dissimilarity between timesteps

if similarity_weights: # if similarity_weights is not empty
    if len(similarity_weights) < len(similarity_columns):
        similarity_weights += [0] * (len(similarity_columns) - len(similarity_weights)) # Fill up with 1s if weights are missing
    similarity_dict = dict(zip(similarity_columns, similarity_weights)) # e.g. {0: 1, 1: 2} if cols=[0, 1] and weights=[1, 2]
# Note that the dissimilarity includes (centroid_i - centroid_j) ** 2, meaning that the weights are effectively squared

## Run the PCTPC algorithm
new_df, weight_list = TSA.from_df(df,
                            index_method='first',
                            columns_for_priority=priority_columns,
                            columns_for_similarity=similarity_dict,
                            clusters_nr_final=final_length,
                            verbose=True)

new_df["Weights"] = weight_list

# Initialize an empty DataFrame with the same columns as agg_df
decompressed_df = pd.DataFrame(columns=new_df.columns) # for plotting

# Loop through each row in agg_df
for index, row in new_df.iterrows():
    # Repeat the current row 'Weights' times and append to the decompressed_df
    # np.repeat creates an array of the index, repeated 'Weights' times
    repeated_rows = [row.values.tolist() for _ in range(int(row["Weights"]))]
    decompressed_rows = pd.DataFrame(repeated_rows, columns=new_df.columns)
    decompressed_df = pd.concat([decompressed_df, decompressed_rows], ignore_index=True)

with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    df.to_excel(writer, sheet_name="Original", index=False)
    new_df.to_excel(writer, sheet_name="Aggregated", index=False)
    decompressed_df.to_excel(writer, sheet_name="Decompressed", index=False)

print("-- Aggregated data saved to Excel --")