import unittest
from TSA.data_processor import process_indata, process_outdata_array, process_outdata_clusters
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

    def process_outdata_array(self):
        # Test process_outdata
        arr, header, index, scaling_factors = process_indata(data)
        new_data = process_outdata_array(arr, header, scaling_factors, index)
        self.assertIsNotNone(new_data)
        self.assertEqual(new_data.shape, data.shape)
        self.assertEqual(new_data.columns.tolist(), data.columns.tolist())
        self.assertEqual(new_data.index.tolist(), data.index.tolist())
        self.assertTrue((new_data == data).all().all())
        new_data_noIndex = process_outdata_array(arr, header, scaling_factors)
        self.assertIsNotNone(new_data_noIndex)
        self.assertEqual(new_data_noIndex.shape, data.shape)
        self.assertEqual(new_data_noIndex.columns.tolist(), data.columns.tolist())
        self.assertEqual(new_data_noIndex.index.tolist(), list(range(len(data))))
        self.assertTrue((new_data_noIndex == data).all().all())

    def process_outdata_clusters(self):
        # Test process_outdata_clusters
        import pickle
        clusters = pickle.load(open('tests/cluster.pickle', 'rb'))
        original_df = pd.read_pickle('tests/original_df.pickle')
        header = original_df.columns
        new_data = process_outdata_clusters(clusters, header, 'first')
        self.assertIsNotNone(new_data)
        self.assertEqual(new_data.shape[0], len(clusters))
        self.assertEqual(new_data.columns.tolist(), header.tolist())
        self.assertEqual(new_data.index.tolist(), [clusters[i]['original_indices'][0] for i in range(len(clusters))])
