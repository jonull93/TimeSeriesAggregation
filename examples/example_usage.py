from TSA.algorithms.pctpc import PCTPCAggregator
from TSA.data_importer import get_filepath
from TSA.data_processor import process_indata, clusters_to_df, decompress_df
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
excel_path = get_filepath("example_usage.xlsx")
df = pd.read_excel(excel_path, sheet_name="Sheet2") # 4 days of hourly Load and Temp
nr_rows = len(df) # 96 rows

# Path to save a figure illustrating the aggregation
fig_path = os.path.join(__file__.split('src')[0],'data/output/example_usage.png') # Save the figure to the output folder of the module

## Set up parameters for the PCTPC algorithm
final_length = nr_rows // 2 # Half of the initial length, in the form of an integer
priority_columns = [0] # Maintain extrema for the Load column
similarity_columns = [0, 1] # Consider Load and Temp columns for similarity calculation
similarity_weights = [1, 1] # Weights for Load and Temp columns, respectively, when calculating dissimilarity between timesteps
global_prio_period = 48 # Period for global priority calculation (the highest and lowest levels in each period are preserved)
# 48 means daily in this case because the resolution is 30 min
# once per every few days might be a good general rule-of-thumb

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
                            global_prio_period=global_prio_period,
                            show_fig=True,
                            fig_path=fig_path,
                            verbose=True)

new_df["Weights"] = weight_list
decompressed_df = decompress_df(new_df)

with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    df.to_excel(writer, sheet_name="Original", index=False)
    new_df.to_excel(writer, sheet_name="Aggregated", index=False)
    decompressed_df.to_excel(writer, sheet_name="Decompressed", index=False)

print("-- Aggregated data saved to Excel --")