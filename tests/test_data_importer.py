import unittest
from TSA.data_importer import DataFrameLoader, XlsxWeightReader, DataLoader

r""" 
    # Example usage
    inflow_file = r"C:\git\TimeSeriesAggregation\data\input\hourly_hydro_inflow_1980-1981.inc"
    load_file = r"hourly_load_1980-1981.inc"
    inflow_df = DataFrameLoader(inflow_file).load()
"""
class TestDataImporter(unittest.TestCase):
    def test_inc_import(self):
        # Test importing .inc files
        inflow_file = r"hourly_hydro_inflow_1980-1981.inc"
        importer = DataFrameLoader(inflow_file)
        data = importer.load()
        self.assertIsNotNone(data)  # basic check to ensure data is not None
        self.assertTrue(data.shape[0] > 0)
        self.assertTrue(data.sum().sum() > 0)
        print(f"Successfully loaded {inflow_file}")

    def test_XlsxWeightReader(self):
        # Test importing XLSX files
        weight_file = r"capacity_mix.xlsx"
        importer = XlsxWeightReader(weight_file, verbose=False, version="allyears")
        data = importer.load()
        self.assertIsNotNone(data)

    def test_DataLoader(self):
        inflow_file = r"hourly_hydro_inflow_1980-1981.inc"
        load_file = r"hourly_load_1980-1981.inc"
        profiles = [inflow_file, load_file]
        weight_file = r"capacity_mix.xlsx"
        df_profiles, df_weights = DataLoader(profiles, weights=weight_file)
        self.assertIsNotNone(df_profiles)
        self.assertIsNotNone(df_weights)
        self.assertTrue(df_profiles.shape[0] > 0)
        self.assertTrue(df_weights.shape[0] > 0)
        self.assertTrue(df_profiles.sum().sum() > 0)
        self.assertTrue(df_weights.sum().sum() > 0)
        print(f"Successfully ran DataLoader")


if __name__ == '__main__':
    unittest.main()
