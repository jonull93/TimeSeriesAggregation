from src.aggregation_algorithms.pctpc import PCTPCAggregator
import numpy as np
import pandas as pd
import os

## Construct sample DataFrame 
# read Load and Temp from "C:\Users\vijulk\OneDrive - Vitecsoftware Group AB\workspace.xlsx" Sheet2
dirpath = os.path.dirname(__file__)
df = pd.read_excel(f"{dirpath}\\data\\input\\example_usage.xlsx", sheet_name="Sheet2")
nr_rows = len(df) # 96 rows
df2 = pd.DataFrame(np.random.randint(0,nr_rows,size=(nr_rows, 2)), columns=list('AB')) # Free-rider columns A and B that do not affect dissimilarity calculation
df = pd.concat([df, df2], axis=1) # Load, Temp, A, B

## Convert to numpy array (for faster processing)
df_as_array = df.to_numpy()

## Set up parameters for the PCTPC algorithm
initial_length = len(df_as_array)
final_length = initial_length // 2 # Half of the initial length, in the form of an integer
priority_columns = [0] # Maintain extrema for the Load column
similarity_columns = [0, 1] # Consider Load and Temp columns for similarity
# weights = [2, 1] # Weights for Load and Temp columns, respectively. 
# Note that the dissimilarity includes (centroid_i - centroid_j) ** 2, meaning that the weights are effectively squared

## Run the PCTPC algorithm
aggregator = PCTPCAggregator(df_as_array, 
                             clusters_nr_final=final_length, 
                             verbose=True, 
                             columns_for_priority=priority_columns, 
                             columns_for_similarity=similarity_columns)

clusters = aggregator.aggregate() 
# clusters is a list of clusters, where each cluster is a dictionary with the following keys:
# - 'centroid': the centroid vector of the cluster (a numpy array of the parameter values)
# - 'vectors': a list of the vectors in the cluster (a list of numpy arrays)
# - 'original_indices': a list of the original indices of the vectors in the cluster (a list of integers)
# - 'priority': the priority value of the cluster (1, 2 or 3, where 1 is the lowest priority)

## Post-processing: Decompress the aggregated data and save to Excel
agg_df = pd.DataFrame([clusters[i]['centroid'] for i in range(len(clusters))], columns=df.columns) # 48 rows
agg_df["Weights"] = [len(clusters[i]['vectors']) for i in range(len(clusters))]
agg_df["Priority"] = [clusters[i]['priority'] for i in range(len(clusters))]

# Initialize an empty DataFrame with the same columns as agg_df
decompressed_df = pd.DataFrame(columns=agg_df.columns) # for plotting

# Loop through each row in agg_df
for index, row in agg_df.iterrows():
    # Repeat the current row 'Weights' times and append to the decompressed_df
    # np.repeat creates an array of the index, repeated 'Weights' times
    repeated_rows = [row.values.tolist() for _ in range(int(row["Weights"]))]
    decompressed_rows = pd.DataFrame(repeated_rows, columns=agg_df.columns)
    decompressed_df = pd.concat([decompressed_df, decompressed_rows], ignore_index=True)

with pd.ExcelWriter("data\\input\\example_usage.xlsx", mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    df.to_excel(writer, sheet_name="Original", index=False)
    agg_df.to_excel(writer, sheet_name="Aggregated", index=False)
    decompressed_df.to_excel(writer, sheet_name="Decompressed", index=False)