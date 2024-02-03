# data_importer.py
import pandas as pd
import numpy as np
import re

class TimeseriesReader():
    def __init__(self, filepath):
        self.filepath = filepath

    def read(self):
        # Read data from file and return a pandas DataFrame
        pass

class CsvDataReader(TimeseriesReader):
    def read(self):
        return pd.read_csv(self.filepath)

class XlsxDataReader(TimeseriesReader):
    def read(self):
        return pd.read_excel(self.filepath)
    
class incDataReader(TimeseriesReader):
    @staticmethod
    def read_file(filename):
        data = []
        with open(filename, "r") as reader:
            for i_l, line in enumerate(reader):
                if line.startswith("*"):
                    continue
                items = re.split(r'\s', line.strip())
                items = [item for item in items if item not in ["","."]]  # Remove empty strings
                if len(items) == 0:
                    continue
                data.append(items)
        return data
    
    def read(self):
        return pd.DataFrame(self.read_file(self.filepath))
    
class DataFramefixer():
    def __init__(self, df):
        self.df = df

    def fix(self):
        # find the column where the timestep is ("h0001" or "d001a")
        row = self.df.iloc[0]
        timestep_col = None
        value_col = None
        for i, item in enumerate(row):
            # check if the item contains "h" followed by 4 digits or "d" followed by 3 digits
            if re.match(r"h\d{4}.*", item) or re.match(r"d\d{3}.*", item):
                timestep_col = i
                print(f"timestep_col: {timestep_col}")
            #elif can be converted to float
            elif re.match(r"\d+.*", item) and "-" not in item:
                value_col = i
                print(f"value_col: {value_col}")
        # set the timestep column as the index
        self.df.set_index(timestep_col, inplace=True)
        # make all non-value columns into a multi-index column
        multi_index_levels = [self.df[col].unique() for col in self.df.columns if col != value_col]
        multi_index = pd.MultiIndex.from_product(multi_index_levels, names=[f"level_{i}" for i in range(len(multi_index_levels))])
        self.df = pd.DataFrame(self.df[value_col].values, index=self.df.index, columns=multi_index)

        return self.df



if __name__ == "__main__":
    # Example usage
    inc_reader = incDataReader(r"C:\git\TimeSeriesAggregation\data\input\hourly_hydro_inflow_1980-1981.inc")
    data = inc_reader.read()
    print(data.head())
    df_fixer = DataFramefixer(data)
    fixed_df = df_fixer.fix()
    print(fixed_df.head())


def load_timeseries_data(file_path):
    # Load and return timeseries data from file_path
    # The data should be turned into a pandas DataFrame with time as the index and regions+technologies as multi-index columns (if there are more than one non-time dimensions)
    # if there only is one non-time dimension, the columns should be a multi-index with the first level being the a custom label (e.g. "heat", "elec", "inflow") and the second level being the regions
    pass

def load_technology_weights(file_path):
    # Load and return technology weights from file_path
    pass

def load_region_specs(file_path):
    # Load and return region specifications from file_path
    # The region specifications should be a dictionary with strings as keys and a list of regions to aggregate as values
    pass

def import_data(profiles, weights, regions_to_aggregate):
    # Load data from file paths and return a dictionary with the loaded data
    # The following timeseries are expected: wind profiles, PV profiles, elec and heat load profiles, hydro inflow profiles
    # Load and inflow profiles will need custom column labels:
    #   heat if "heat" is in the file name, 
    #   elec if "elec" is in the file name or ("heat" is not in the file name and ("load" is in the file name or "demand" is in the file name)),
    #   inflow if "inflow" is in the file name or "hydro" is in the file name
    timeseries_data = pd.DataFrame()
    for file_path in profiles:
        data = load_timeseries_data(file_path)
        timeseries_data = pd.concat([timeseries_data, data], axis=1)
    if weights:
        technology_weights = load_technology_weights(weights)
    if regions_to_aggregate:
        region_specs = load_region_specs(regions_to_aggregate)
    return {
        "timeseries_data": timeseries_data,
        "technology_weights": technology_weights,
        "region_specs": region_specs
    }

# Additional utility functions as needed