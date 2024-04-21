import unittest
from TSA.data_processor import process_indata, process_outdata_array, clusters_to_df
import pickle
import pandas as pd

data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])

class TestDataProcessor(unittest.TestCase):
    def test_process_indata(self):
        # Test process_indata
        arr, header, index, scaling_factors = process_indata(data)
        self.assertIsNotNone(arr)
        self.assertIsNotNone(header)
        self.assertIsNotNone(index)
        self.assertIsNotNone(scaling_factors)
        self.assertEqual(arr.shape, data.shape)
        self.assertEqual(header.tolist(), data.columns.tolist())
        self.assertEqual(index.tolist(), data.index.tolist())
        self.assertEqual(scaling_factors, [max(abs(data.iloc[:,i])) for i in range(data.shape[1])])

    def test_process_outdata_array(self):
        # Test process_outdata
        arr, header, index, scaling_factors = process_indata(data)
        new_data = process_outdata_array(arr, header, scaling_factors, index)
        self.assertIsNotNone(new_data)
        self.assertEqual(new_data.shape, data.shape)
        self.assertEqual(new_data.columns.tolist(), data.columns.tolist())
        self.assertEqual(new_data.index.tolist(), data.index.tolist())
        self.assertTrue((new_data.values == data.values).all().all())
        new_data_noIndex = process_outdata_array(arr, header, scaling_factors)
        self.assertIsNotNone(new_data_noIndex)
        self.assertEqual(new_data_noIndex.shape, data.shape)
        self.assertEqual(new_data_noIndex.columns.tolist(), data.columns.tolist())
        self.assertEqual(new_data_noIndex.index.tolist(), list(range(len(data))))
        self.assertTrue((new_data_noIndex.values == data.values).all().all())

    def test_clusters_to_df(self):
        # Test clusters_to_df
        import pickle
        import numpy as np
        clusters = pickle.load(open('tests/clusters.pickle', 'rb'))
        scaling_factors = np.array([1, 1, 1, 1])
        original_df = pd.read_pickle('tests/original_df.pickle')
        header = original_df.columns
        new_data = clusters_to_df(clusters, header, scaling_factors, index_method='first')
        self.assertIsNotNone(new_data)
        self.assertEqual(new_data.shape[0], len(clusters))
        self.assertEqual(new_data.columns.tolist(), header.tolist())
        self.assertEqual(new_data.index.tolist(), [clusters[i]['original_indices'][0] for i in range(len(clusters))])
        new_data_with_refindex = clusters_to_df(clusters, header, scaling_factors, index_method='first', ref_index=original_df.index)
        print(new_data_with_refindex)
        self.assertIsNotNone(new_data_with_refindex)
        self.assertEqual(new_data_with_refindex.shape[0], len(clusters))
        self.assertEqual(new_data_with_refindex.columns.tolist(), header.tolist())
        ind_list = [clusters[i]['original_indices'][0] for i in range(len(clusters))]
        self.assertEqual(new_data_with_refindex.index.tolist(), original_df.index[ind_list].tolist())
