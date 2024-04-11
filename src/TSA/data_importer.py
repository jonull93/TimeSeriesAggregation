# data_importer.py
import pandas as pd
import openpyxl
import numpy as np
import re
import pycountry
import sys
#if __name__ == "__main__": sys.path.append(r"./") # so that the tech_names can be imported from the data.input.possible_tech_names module
from utils import tech_names, VRE
import os
from pathlib import Path, WindowsPath

# if ../data/input/additional_tech_names.txt exists, add the tech names to the list
p = Path("../data/input/additional_tech_names.txt")
if p.exists():
    with open(p.resolve(), "r") as reader:
        for line in reader:
            if line[0] in ["#","*"]:
                continue
            # split the line by spaces, commas and semicolons
            line = re.split(r'[\s,;]', line)
            # remove empty strings
            line = [item for item in line if item]
            tech_names += line

def looks_like_year(item):
    if re.match(r"(19|20)\d{2}", item) or re.match(r"\d{4}-\d{4}", item):
        return True
    return False

def looks_like_tech(item):
    if item in tech_names:
        return True
    return False
  
def get_filepath(file_path):
    # check if the file exists, otherwise, check if the file exists in the input folder
    p = Path(file_path)
    if not p.exists():
        glob_result = list(Path("..").glob(f"**/{file_path}")) # search for the file in the parent directory and its subdirectories
        if len(glob_result) > 0:
            return str(glob_result[0].resolve())
        else:
            raise ValueError(f"File {file_path} not found in the input folder or the current directory.")
    return file_path
class TimeseriesReader():
    def __init__(self, filepath):
        # check if the file exists, otherwise, check if the file exists in the input folder
        self.filepath = get_filepath(filepath)

    def read(self):
        # Read data from file and return a pandas DataFrame
        pass

    def timeseries_name(self):
        # infer the timeseries name from the file name
        # The following timeseries are expected: wind profiles, PV profiles, elec and heat load profiles, hydro inflow profiles
        # Load and inflow profiles will need custom column labels:
        #   heat if "heat" is in the file name, 
        #   elec if "elec" is in the file name or ("heat" is not in the file name and ("load" is in the file name or "demand" is in the file name)),
        #   inflow if "inflow" is in the file name or "hydro" is in the file name
        filename = Path(self.filepath).name # get the filename from the path
        if "heat" in filename:
            return "heat"
        elif "elec" in filename or ("heat" not in filename and ("load" in filename or "demand" in filename)):
            return "elec"
        elif "inflow" in filename or "hydro" in filename:
            return "inflow"
        elif "VRE" in filename:
            return "VRE"
        elif "wind" in filename.lower():
            return "wind"
        elif "PV" in filename:
            return "PV"
        elif "gen" in filename:
            return "gen"
        else:
            return "unknown"


class CsvDataReader(TimeseriesReader):
    def read(self):
        df = pd.read_csv(self.filepath)
        df.name = self.timeseries_name()
        return df

class XlsxDataReader(TimeseriesReader):
    def read(self):
        df = pd.read_excel(self.filepath)
        df.name = self.timeseries_name()
        return df
    
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
        df = pd.DataFrame(self.read_file(self.filepath))
        df.name = self.timeseries_name()
        return df
    
class DataFrameLoader():
    def __init__(self, filepath, **kwargs):
        self.filepath = filepath
        self.df = None

    def load(self):
        # call the appropriate reader based on the file extension
        if self.filepath.endswith(".csv"):
            reader = CsvDataReader(self.filepath, **self.kwargs)
        elif self.filepath.endswith(".xlsx"):
            reader = XlsxDataReader(self.filepath)
        elif self.filepath.endswith(".inc"):
            reader = incDataReader(self.filepath)
        else:
            raise ValueError("Unsupported file format")
        self.df = reader.read()
        self.fix()
        return self.df
        
    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
        
    def name_columns(self):
        row = self.df.iloc[0]
        country_names = [i.name for i in pycountry.countries]
        for i, item in enumerate(row):
            if re.match(r"h\d{4}.*", item) or re.match(r"d\d{3}.*", item):
                # set the column name to "timestep"
                self.df.rename(columns={i: "timestep"}, inplace=True)
            #elif 4 digits starting with 19 or 20, OR 4 digits followed by a dash and 4 digits
            elif looks_like_year(item):
                # set the column name to "year"
                self.df.rename(columns={i: "year"}, inplace=True)
            elif item in tech_names:
                # set the column name to "technology"
                self.df.rename(columns={i: "tech"}, inplace=True)
            elif item in country_names or item.isupper():
                # set the column name to "region"
                self.df.rename(columns={i: "region"}, inplace=True)
            #elif item can be converted to float
            elif self.is_number(item) and "-" not in item:
                # set the column name to "value"
                self.df.rename(columns={i: self.df.name}, inplace=True)

    def pivot(self):
        # assume one of the columns is called "timestep"
        # the column called "timestep" should be the index, and the columns whose name does not match the df.name should be made into multi-level columns
        value_col = self.df.name
        timestep_col = "timestep"
        other_cols = [col for col in self.df.columns if col != value_col and col != timestep_col]
        self.df = self.df.pivot(index=timestep_col, values=value_col, columns=other_cols)
        # the columns are now multiindexed, but the order is wrong
        #reorder the columns to the following order: year, region, tech (but only if the columns are present)
        reordered_cols = [i for i in ["year", "region", "tech"] if i in self.df.columns.names]+[i for i in self.df.columns.names if i not in ["year", "region", "tech"]]
        self.df = self.df.reorder_levels(reordered_cols, axis=1)
        self.df.name = value_col # the name is lost in the pivot operation, so we need to set it again
    
    def fix(self):
        self.name_columns()
        self.pivot()
        #add a top level to the columns, with the name being the name of the timeseries
        tech_in_col = None
        for i,level in enumerate(self.df.columns[0]):
            if looks_like_tech(level):
                tech_in_col = i
                break
        if tech_in_col != None:
            # move the tech level to the first level
            non_tech_levels = [i for i in range(len(self.df.columns.levels)) if i != tech_in_col]
            self.df = self.df.reorder_levels([tech_in_col]+non_tech_levels, axis=1)
        else:
            self.df.columns = pd.MultiIndex.from_tuples([(self.df.name, *col) for col in self.df.columns])
        # make all the values numeric
        self.df = self.df.apply(pd.to_numeric)
        return self.df

class WeightReader():
    def __init__(self, filepath, version=None):
        self.filepath = filepath
        # check if the file exists, otherwise, check if the file exists in the input folder
        self.filepath = get_filepath(filepath)
        self.version = version
        self.weights = None
        self.expected_techs = ["WON", "wind", "PV", "solar", "PVPA2", "WOFF5", "WONA5", "WONA2"] # at least one of these should be in the first or second column


class XlsxWeightReader(WeightReader):
    def __init__(self, filepath, version=None, sheet_name=None, verbose=False):
        super().__init__(filepath, version)
        self.loaded_df = pd.read_excel(self.filepath, sheet_name=None)
        self.verbose = verbose
        if sheet_name:
            self.sheet = sheet_name
        else:
            self.sheet = self.find_sheet()
            if self.verbose: print(f"Using sheet {self.sheet}")
        if self.verbose: print(self.loaded_df[self.sheet])

    def techs_in_sheet(self, sheet):
        df = self.loaded_df[sheet]
        for tech in self.expected_techs:
            for col in df.columns:
                if tech in df[col].values:
                    return True
    
    def find_sheet(self):
        sheet_names = list(self.loaded_df.keys())
        for sheet in sheet_names:
            if sheet.lower() in ["capacity","cap"] and self.techs_in_sheet(sheet):
                return sheet
            if self.version and self.version in sheet and self.techs_in_sheet(sheet):
                return sheet
        if self.techs_in_sheet(sheet_names[0]):
            return sheet_names[0]
        raise LookupError(f"No appropriate sheet found in {self.filepath}")
    
    def find_head_row(self, df):
        # check if the column namnes contain "cap" or "capacity" or "value"
        col_to_use = None
        for i,col in enumerate(df.columns):
            if "cap" in col.lower() or "capacity" in col.lower() or "value" in col.lower():
                return 0, col_to_use
            if self.version and self.version in col:
                col_to_use = [i]
        # find the first row that contains "cap" or "capacity" or "value"
        for i, row in df.iterrows():
            if self.verbose: print(f"Checking row {i}: {row}")
            #look for "cap" or "capacity" or "value" in the row
            if "cap" in row.values or "capacity" in row.values or "value" in row.values:
                return i+1, col_to_use
        
    @staticmethod
    def exists_in_iterable(value, iterable):
        for item in iterable:
            if item in value:
                return True
        return False
    
    def find_index_and_value_cols(self, df):
        # return the column indices that should be used as the index columns and a list of all columns to load
        cols_to_use = None # None equals all columns
        version_col = None
        cap_names = ["cap", "capacity", "value", "installed"]
        for i,col in enumerate(df.columns):
            if self.exists_in_iterable(col.lower(), cap_names):
                return list(range(i)), cols_to_use
            if self.version and self.version in col:
                version_col = i
        if self.version and version_col == None:
            raise ValueError(f"Unknown header without any {self.version} column in {self.filepath}")
        elif version_col == None:
            version_col = len(df.columns)-1
            print(f"Using last column ({df.columns[version_col]}) as version column")
        if self.verbose: print(f"Checking first row: {df.iloc[0].values}")
        for i, value in enumerate(df.iloc[0].values):
            #look for "cap" or "capacity" in the row
            if self.exists_in_iterable(value, cap_names):
                index_cols = list(range(i))
                cols_to_use = index_cols + [version_col]
                return index_cols, cols_to_use
            
    def filter_nonVRE_techs(self, df):
        # filter out non-VRE techs from the index level called tech
        techs = df.index.get_level_values("tech")
        techs = [tech for tech in techs if tech in VRE]
        return df.loc[techs]
            
    def load(self):
        header, col_to_use = self.find_head_row(self.loaded_df[self.sheet])
        if self.verbose: print(f"Using header {header} and column {col_to_use} as the value column")
        index_cols, cols_to_use = self.find_index_and_value_cols(self.loaded_df[self.sheet])
        if self.verbose: print(self.loaded_df[self.sheet])
        if self.verbose: print(f"Using index columns {index_cols}, and columns {cols_to_use} as the value column")
        df = pd.read_excel(self.filepath, sheet_name=self.sheet, header=header, index_col=index_cols, usecols=cols_to_use)
        # if the column name is capacity.# or cap.#, remove the dot and everything after it
        df.columns = [re.sub(r"\.\d+","",col) for col in df.columns]
        df = self.filter_nonVRE_techs(df)
        if self.verbose: print(df)
        # if version is defined, look for a column with the version name
        self.weights = df
        return df

def DataLoader(profiles, weights=False):
    # Load all timeseries found in the filepaths in profiles and combine them into a single DataFrame
    # Load the weights from the filepath in weights and return a DataFrame
    timeseries_data = pd.DataFrame()
    for file_path in profiles:
        if type(file_path) == WindowsPath: file_path = str(file_path)
        data = DataFrameLoader(file_path).load()
        # if a colum level looks like a year, move it to the first level
        first_col = data.columns[0]
        for i, col in enumerate(first_col):
            if looks_like_year(col):
                data = data.swaplevel(i,0,axis=1)
                break
        timeseries_data = pd.concat([timeseries_data, data], axis=1)
    if weights:
        technology_weights = XlsxWeightReader(weights, verbose=False).load()
    else:
        technology_weights = None
    return timeseries_data, technology_weights


if __name__ == "__main__":
    # Example usage
    inflow_file = r"hourly_hydro_inflow_1980-1981.inc"
    load_file = r"hourly_load_1980-1981.inc"
    heat_file = r"hourly_heat_demand_1980-1981.inc"
    VRE_file = r"gen_profile_VRE_1980-1981.inc"
    #profiles = [inflow_file, load_file, heat_file, VRE_file]
    # get all file names in ../data/input/profiles to load
    profiles = list(Path("../data/input/profiles to load").glob("*")) # get all files in the folder
    print(profiles)
    cap_file = r"capacity_mix.xlsx"
    inflow_df = DataFrameLoader(inflow_file).load()
    print("inflow df:\n",inflow_df.head())
    cap_df = XlsxWeightReader(cap_file, verbose=True).load()
    print("cap df:\n",cap_df.head())
    DataLoader(profiles, cap_file)
    print("done")
    # 



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

def import_data2(profiles, weights, regions_to_aggregate):
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