import unittest
import numpy as np
from src.aggregation_algorithms.pctpc import PCTPCAggregator

class TestPCTPCAggregator(unittest.TestCase):
    
    def setUp(self):
        # Sample data for testing
        self.sample_data = np.random.rand(300, 10)  # 100 timesteps, 10 dimensions
        self.sample_data[2,2] = 2 # Add a global maximum
        self.clusters_nr_final = 30  # Number of clusters to be created

    def test_initialization(self):
        aggregator = PCTPCAggregator(data=self.sample_data, clusters_nr_final=self.clusters_nr_final)
        self.assertEqual(aggregator.data.shape, self.sample_data.shape)
        self.assertEqual(aggregator.clusters_nr_final, self.clusters_nr_final)

    # Additional setup and initialization tests

    def test_priority_assignment(self):
        aggregator = PCTPCAggregator(data=self.sample_data, clusters_nr_final=self.clusters_nr_final)
        aggregator.assign_priorities()
        # Assert statements based on your priority assignment logic
        # For example, check if global maxima and minima are identified correctly
        self.assertEqual(len(aggregator.priorities), len(self.sample_data))
        self.assertEqual(len(aggregator.priorities), len(aggregator.data))
        self.assertGreaterEqual(sum(aggregator.priorities), 6) # there should be at least 2 global extreme, both of which have prio=3
        self.assertEqual(aggregator.priorities[2], 3) # the global maximum should have prio=3
        
    def test_cluster_initialization(self):
        aggregator = PCTPCAggregator(data=self.sample_data, clusters_nr_final=self.clusters_nr_final)
        aggregator.initialize_clusters()
        self.assertEqual(len(aggregator.clusters), len(self.sample_data))
        # Additional checks for the structure and content of the clusters

    def test_dissimilarity_computation(self):
        aggregator = PCTPCAggregator(data=self.sample_data, clusters_nr_final=self.clusters_nr_final)
        aggregator.initialize_clusters()
        dissimilarity = aggregator.compute_dissimilarity(aggregator.clusters[0], aggregator.clusters[1])
        # Assert statements based on expected dissimilarity

    def test_cluster_merging(self):
        aggregator = PCTPCAggregator(data=self.sample_data, clusters_nr_final=self.clusters_nr_final)
        aggregator.assign_priorities()
        aggregator.initialize_clusters()
        original_cluster_count = len(aggregator.clusters)
        aggregator.merge_clusters()
        self.assertLess(len(aggregator.clusters), original_cluster_count)
        # Additional checks to validate merging rules

    def test_full_aggregation_process(self):
        aggregator = PCTPCAggregator(data=self.sample_data, clusters_nr_final=self.clusters_nr_final, verbose=False, columns_for_priority=[0,(1,2)], columns_for_similarity=[0,(1,2)])
        final_clusters = aggregator.aggregate()
        self.assertEqual(len(final_clusters), self.clusters_nr_final)
        self.assertEqual(len(final_clusters[0]["centroid"]), len(self.sample_data[0]))
        clusters_to_print = final_clusters if len(final_clusters) < 50 else final_clusters[:50]
        print(f"Successfully clustered indices:", [cluster["original_indices"] for cluster in final_clusters])
        # Additional assertions can be made based on the expected properties of the final clusters

if __name__ == '__main__':
    unittest.main()
