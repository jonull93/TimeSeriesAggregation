# pctpc.py
from TSA.algorithms.base import AggregationAlgorithm
import numpy as np

class PCTPCAggregator(AggregationAlgorithm):
    """
    PCTPC: 
    Take a "database" ΩN of N number of timesteps , where the data for each timestep is represented as a multidimensional vector x, and reduce it to K timesteps/vectors.
    1) Assign one of three priority levels to each timestep, e.g. depending on whether it is a global extreme, local extreme or neither. Three subdatasets are thus formed: ΩH, ΩM and ΩL.
    2) Create N clusters, one for each vector of the database ΩN. Note that, at this point, the centroid of each cluster is equal to the vector contained in itself.
    3) Compute the dissimilarity between each pair of adjacent clusters I and J according to Ward's method: D(I, J)= 2|I||J| /( |I| + |J| ) ||¯xI - ¯xJ||^2
    where ¯xI and ¯xJ are defined as the centroids of clusters I and J, respectively.
    4) Identify the two adjacent clusters I∗ andJ∗ with minimum dissimilarity between them.
    5) Compute the centroid xbar of merging clusters I∗ and J∗ by following the rules below: 
      a) If both clusters I∗ and J∗ contain a vector of subset ΩH, set D(I∗,J∗) at a sufficiently large positive value and go to Step 4) without merging them.
      b) If one of the clusters contains a vector of subset ΩH and the other does not, they are merged and the resulting centroid is equal to the vector with high-priority values.
      c) If both clusters contain a vector of subset ΩM and none of them contain vectors of subset ΩH, they are merged and the resulting centroid is computed using the following equation:
    xbar = (|I|xbar_I + |J|xbar_J) / (|I| + |J|)
      d) If one of the clusters contains only vectors of subsets ΩM and ΩL, and the other contains only vectors of subset ΩL, they are merged and the resulting centroid is equal to the centroid of the former.
      e) If both clusters contain only vectors of subset ΩL, they are merged and the resulting centroid is computed using the formula in c).
    """
    def __init__(self, data:np.ndarray, clusters_nr_final=False, acceptable_dissimilarity_percentile=False, verbose=False, 
                 columns_for_priority=None, columns_for_similarity=False, nan=None):
        super().__init__()
        if clusters_nr_final and acceptable_dissimilarity_percentile:
            raise ValueError("Only one of num_final_clusters or acceptable_dissimilarity_percentile can be specified")
        elif not clusters_nr_final and not acceptable_dissimilarity_percentile:
            #default is halving the number of timesteps
            clusters_nr_final = len(data) // 2
            if verbose:
                print(f"Using default number of final clusters: {clusters_nr_final}")
        elif acceptable_dissimilarity_percentile:
            if type(acceptable_dissimilarity_percentile) == str:
                acceptable_dissimilarity_percentile = float(acceptable_dissimilarity_percentile.strip("%"))
            if type(acceptable_dissimilarity_percentile) not in [int, float]:
                raise ValueError("acceptable_dissimilarity_percentile must be a number")
        if type(columns_for_similarity) == dict:
            # make sure that the keys are integers or tuples of integers
            if not all([type(k) in [int, tuple] for k in columns_for_similarity.keys()]):
                raise ValueError("The keys of columns_for_similarity must be integers or tuples of integers")
            # make sure that the values are integers
            if not all([type(v) == int for v in columns_for_similarity.values()]):
                raise ValueError("The values of columns_for_similarity must be integers")
            self.weights_for_similarity = np.array(list(columns_for_similarity.values()))
            self.columns_for_similarity = list(columns_for_similarity.keys())
        else:
            if columns_for_similarity:
                self.weights_for_similarity = np.array([1]*len(columns_for_similarity))
                self.columns_for_similarity = columns_for_similarity
            else:
                self.weights_for_similarity = np.array([1]*data.shape[1])
                self.columns_for_similarity = list(range(data.shape[1]))
        #if columns_for_priority and len(columns_for_priority) != data.shape[1]:
        #    raise ValueError("The number of columns for priority computation must match the number of columns in the data")
        #if columns_for_similarity and len(columns_for_similarity) != data.shape[1]:
        #    raise ValueError("The number of columns for similarity computation must match the number of columns in the data")
        self.data = data  # Assumed to be a numpy array where each row represents a timestep and each column represents a dimension of the vector
        self.verbose = verbose  # Whether to print debug information
        self.columns_for_priority = columns_for_priority if columns_for_priority else list(range(data.shape[1])) # The columns to use for priority computation
        self.clusters_nr_final = clusters_nr_final  # The desired number of clusters (K)
        self.acceptable_dissimilarity_percentile = acceptable_dissimilarity_percentile  # The percentile of dissimilarity to use as a threshold
        self.priorities = np.array([0]*len(data))  # Initialize with lowest priority
        self.dissimilarity_vector = list(range(len(data) - 1))  # A vector to store the dissimilarity to the next cluster for each cluster

    def aggregate(self):
        self.assign_priorities()
        self.initialize_clusters()
        if self.clusters_nr_final:
            while len(self.clusters) > self.clusters_nr_final:
                self.merge_clusters() # this will update self.clusters and self.dissimilarity_vector
                if self.verbose:
                    #print(f"Cluster sizes:", [len(self.clusters[i]["vectors"]) for i in range(len(self.clusters))])
                    #print(f"Clustered indices:", [self.clusters[i]["original_indices"] for i in range(len(self.clusters)) if self.clusters[i]["original_indices"][-1]<20])
                    pass
        elif self.acceptable_dissimilarity_percentile:
            while np.min(self.dissimilarity_vector) < self.threshold_dissimilarity:
                self.merge_clusters()
                
        return self.clusters
    
    def assign_priorities(self):
        """
        Assign priorities to each timestep based on columns specified in columns_for_priority. Priorities are used to determine how clusters are merged.
        When multiple columns are specified, the maximum priority is taken for each timestep.
        1: Low priority, 2: Medium priority, 3: High priority
        """
        if self.verbose:
            print(f"Assigning priorities based on columns: {self.columns_for_priority}")
        for column_index in self.columns_for_priority:
            # If the column_index is a list or tuple, average the columns in the lis. This represents the combining of regions or technologies.
            if type(column_index) in [list, tuple]:
                column = np.mean(self.data[:, column_index], axis=1)
            else:
                column = self.data[:, column_index]
            column_priorities = self.find_priorities_for_column(column)
            self.priorities = np.maximum(self.priorities, column_priorities)
        if self.clusters_nr_final < sum(self.priorities == 3):
            raise ValueError(f"The number of high-priority timesteps ({sum(self.priorities == 3)}) exceeds the number of final clusters ({self.clusters_nr_final})")
        if len(self.columns_for_priority) > 1 and self.verbose:
            print(f"Final nr of high / medium / low priority timesteps: {np.sum(self.priorities == 3)} / {np.sum(self.priorities == 2)} / {np.sum(self.priorities == 1)}")

    def find_priorities_for_column(self, column:np.ndarray, threshold=0):
        """
        Assign priorities for a single-dimensional array (column).
        - High priority: Global extrema
        - Medium priority: Local extrema that are sufficiently different from adjacent local extrema.
        - Low priority: Others
        """
        global_max = np.max(column)
        threshold = 0.10 * global_max

        priorities = np.array([1]*len(column))  # Initialize with lowest priority

        # Identify global extrema
        global_max_indices = np.argwhere(column == global_max).flatten()
        priorities = self.set_priority(global_max_indices, priorities, 3)  # Set high priority
        global_min_indices = np.argwhere(column == np.min(column)).flatten()
        priorities = self.set_priority(global_min_indices, priorities, 3)  # Set high priority

        # Identify local extrema
        local_extrema_indices = np.argwhere(self.find_local_extrema(column)).flatten()

        if threshold == 0:
            priorities = self.set_priority(local_extrema_indices, priorities, 2)  # Set medium priority for all local extrema
        else: # If there is a non-zero threshold, use it to filter out insignificantly different adjacent local extrema
            for i in range(1, len(local_extrema_indices) - 1):
                if abs(column[local_extrema_indices[i]] - column[local_extrema_indices[i - 1]]) > threshold \
                        and abs(column[local_extrema_indices[i]] - column[local_extrema_indices[i + 1]]) > threshold:
                    priorities = self.set_priority([local_extrema_indices[i]], priorities, 2)
        if self.verbose:
            print(f"Nr of high / medium / low priority timesteps: {np.sum(priorities == 3)} / {np.sum(priorities == 2)} / {np.sum(priorities == 1)}")
        return priorities

    @staticmethod # Static method since it does not depend on the instance (no self.# arguments) (could be a separate function outside the class as well)
    def find_local_extrema(column):
        # Shift the column to the right and left
        left_shifted = np.roll(column, 1)
        right_shifted = np.roll(column, -1)
        # Identify local maxima and minima
        local_maxima = (column > left_shifted) & (column > right_shifted)
        local_minima = (column < left_shifted) & (column < right_shifted)
        # Handle edge cases for the first and last element
        local_maxima[0] = local_maxima[-1] = False
        local_minima[0] = local_minima[-1] = False
        # return a single array of extrema bools
        return local_maxima | local_minima

    def set_priority(self, indices, priorities, priority_level):
        """
        Set the specified priority level to the given indices.
        """
        for index in indices:
            priorities[index] = np.max([priorities[index], priority_level])
        return priorities
    
    def is_local_extrema(self, column, index):
        """
        Check if the value at the given index is a local extrema.
        """
        return (column[index] > column[index - 1] and column[index] > column[index + 1]) \
            or (column[index] < column[index - 1] and column[index] < column[index + 1])

    def initialize_clusters(self):
        """
        Initialize clusters such that each cluster contains one vector from the dataset.
        Each cluster is represented as a dictionary with two keys: 'centroid' and 'vectors'.
        'centroid' is the centroid of the cluster (initially the vector itself),
        'vectors' is a list of vectors belonging to this cluster (initially just one vector).
        """
        self.clusters = [] # List since the size changes dynamically, unlike numpy arrays
        for i, vector in enumerate(self.data):
            cluster = {
                'centroid': vector,
                'vectors': [vector],
                'original_indices': [i], # Keep track of the original indices for debugging and visualization
                'priority': self.priorities[i]
            }
            self.clusters.append(cluster)
            if i>0:
                self.dissimilarity_vector[i-1] = self.compute_dissimilarity(self.clusters[i-1], self.clusters[i])
        if self.verbose:
            print(f"Initialized {len(self.clusters)} clusters and dissimilarity vector: {self.dissimilarity_vector[:10]} (10/{len(self.dissimilarity_vector)})")
        if self.acceptable_dissimilarity_percentile:
            self.threshold_dissimilarity = np.percentile(self.dissimilarity_vector, self.acceptable_dissimilarity_percentile)
            

    def merge_clusters(self):
        """
        Find the most similar adjacent clusters (pair_to_merge) and call apply_merging_rules to merge them.
        """
        if len(self.clusters) <= 2:
            return  # No clusters to merge

        # Step 4: Identify the pair of adjacent clusters with minimum dissimilarity
        ind_min_dissimilarity = np.argmin(self.dissimilarity_vector) # TODO: by only considering updated and previous near-minimums, the search can be made faster
        pair_to_merge = (ind_min_dissimilarity, ind_min_dissimilarity + 1)
        if self.verbose:
            #print(f"Pair to merge: {pair_to_merge} with dissimilarity {self.dissimilarity_vector[ind_min_dissimilarity]:.4f} ({len(self.clusters)} clusters left)")
            pass

        # Step 5: Merge the identified clusters following the priority rules
        self.apply_merging_rules(pair_to_merge)

    def apply_merging_rules(self, pair_to_merge):
        """
        Apply merging rules to the identified pair of clusters.
        Updates the clusters list and the dissimilarity vector.
        """
        i, j = pair_to_merge
        cluster_i, cluster_j = self.clusters[i], self.clusters[j]

        # Check for priority rules (a-e)
        # a) If both clusters contain a vector of subset ΩH, set D(I*, J*) to a large value
        if cluster_i['priority'] == 3 and cluster_j['priority'] == 3: # as is, top-priority timesteps can be merged as long as its not with another top-priority timestep
            self.dissimilarity_vector[i] = self.dissimilarity_vector[j] = np.inf
            return # Skip merging

        # b-e) Merge clusters and compute new centroid based on other rules
        new_cluster = self.merge_and_compute_centroid(cluster_i, cluster_j)

        # Replace the clusters in the list
        self.clusters[i] = new_cluster
        del self.clusters[j] # the cluster that got integrated into the other is removed
        del self.dissimilarity_vector[i] # technically, either i or i-1 can be removed 
        
        # Update the dissimilarity vector
        if i < len(self.clusters) - 1:
            self.dissimilarity_vector[i] = self.compute_dissimilarity(self.clusters[i], self.clusters[i + 1])
        if i > 0:
            self.dissimilarity_vector[i - 1] = self.compute_dissimilarity(self.clusters[i - 1], self.clusters[i])


    def merge_and_compute_centroid(self, cluster_i, cluster_j):
        """
        Merge two clusters and compute the new centroid based on priority rules.
        Returns the new cluster dictionary.
        """
        # Assuming centroids are numpy arrays for vectorized operations
        size_i = len(cluster_i['vectors'])
        size_j = len(cluster_j['vectors'])
        new_size = size_i + size_j

        # Rule (b-e)
        if cluster_i['priority'] == 3 or cluster_j['priority'] == 3:
            # Choose the centroid of the cluster with the high-priority vector
            new_centroid = cluster_i['centroid'] if cluster_i['priority'] == 3 else cluster_j['centroid']
        elif cluster_i['priority'] == cluster_j['priority']:
            # Compute the weighted average of the centroids
            new_centroid = (size_i * cluster_i['centroid'] + size_j * cluster_j['centroid']) / new_size
        else:
            # Choose the centroid of the cluster with the medium-priority vector
            new_centroid = cluster_i['centroid'] if cluster_i['priority'] == 2 else cluster_j['centroid']

        return {
            'centroid': new_centroid,
            'vectors': cluster_i['vectors'] + cluster_j['vectors'],
            'original_indices': cluster_i['original_indices'] + cluster_j['original_indices'],
            'priority': max(cluster_i['priority'], cluster_j['priority'])
        }

    def compute_dissimilarity(self, cluster_i, cluster_j):
        """
        Compute the dissimilarity between two clusters using Ward's method.

        Parameters:
        cluster_i (dict): The first cluster.
        cluster_j (dict): The second cluster.

        Returns:
        float: The computed dissimilarity.
        """
        centroid_i = self.calculate_centroid_for_dissimilarity(cluster_i['centroid'])
        centroid_j = self.calculate_centroid_for_dissimilarity(cluster_j['centroid'])

        size_i = len(cluster_i['vectors'])
        size_j = len(cluster_j['vectors'])

        # Compute the squared Euclidean distance between the centroids
        squared_distance = np.sum(((centroid_i - centroid_j)*self.weights_for_similarity) ** 2)
        # one way to account for nan values would be 
        # np.sum(np.nan_to_num((centroid_i-centroid_j)**2,nan=0))
        # where nans in centroid_i-centroid_j are replaced with 0 (aka no contribution to the dissimilarity for that dimension)

        # Calculate dissimilarity using Ward's method
        dissimilarity = 2.0 * size_i * size_j / (size_i + size_j) * squared_distance
        return dissimilarity

    def calculate_centroid_for_dissimilarity(self, centroid):
        """
        Calculate the modified centroid based on the columns_for_similarity.
        Only used for the dissimilarity computation to account for regions and technologies that can be combined.

        Parameters:
        centroid (array-like): The original centroid with full dimensions.

        Returns:
        array-like: A modified centroid containing only columns in columns_for_similarity.
        """
        modified_centroid = []
        for col in self.columns_for_similarity:
            #w = self.weights_for_similarity[i]
            if isinstance(col, (list, tuple)):
                # Average the values of the specified columns
                combined_value = np.mean([centroid[c] for c in col])
                modified_centroid.append(combined_value)
            else:
                # Use the value of the specified column
                modified_centroid.append(centroid[col])

        return np.array(modified_centroid)
    

def test_aggregator(timesteps=15, dimensions=3, final_clusters=6):
    data = np.random.rand(timesteps, dimensions)
    aggregator = PCTPCAggregator(data, clusters_nr_final=final_clusters, verbose=True, columns_for_priority=[0], columns_for_similarity=[0,(1,2)])
    clusters = aggregator.aggregate() 
    return clusters
    # clusters is a list of dictionaries, each dictionary contains "centroid", "vectors" and "priority"
    # concatenating all the vectors in the clusters recreates the 'data' from the previous step

if __name__ == "__main__":
    # Example usage
    clusters = test_aggregator()
    # clusters is a list of dictionaries, each dictionary contains "centroid", "vectors" and "priority"
    # concatenating all the vectors in the clusters recreates the 'data' from the previous step
    print(f"Successfully clustered indices:", [cluster["original_indices"] for cluster in clusters])