import unittest
from TSA.main import from_df
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3, 4, 2, 6, 1], 'B': [4, 5, 6, 7, 8, 9, 10]}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

class TestMain(unittest.TestCase):
    def test_from_df(self):
        # Test from_df
        new_df, weights = from_df(df)
        print(new_df)
        self.assertIsNotNone(new_df)
        self.assertGreater(df.shape[0], new_df.shape[0])
        self.assertEqual(new_df.columns.tolist(), df.columns.tolist())
        self.assertTrue('g' in new_df.index)
        self.assertEqual(sum(weights), len(df))