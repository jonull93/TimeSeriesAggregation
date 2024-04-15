# main.py
# This is the main entry point for the application. It should 
# - parse command-line arguments, 
# - load the data, 
# - process it if necessary, then apply the selected aggregation algorithm, and finally
# - export the results. 
# The main function should be called when the script is executed.
#from config_manager import parse_arguments
from TSA.data_importer import import_data
from TSA.data_processor import process_indata, process_outdata_clusters, process_outdata_array
from TSA.algorithms.pctpc import PCTPCAggregator
#from data_illustrator import create_plots
#from data_exporter import export_data


def main():
    # TODO: Parse command-line arguments
    profiles, weights, regions_to_aggregate, heat_load_to_electrify = parse_arguments()
    # TODO: Load data using data_importer
    ugly_data = import_data(profiles, weights)
    # TODO: Process data if necessary
    #data = process_data(ugly_data, heat_load_to_electrify, regions_to_aggregate)

    #combine import_data() and process_data() into black box?
    
    # Ready for testing: PCPTC Aggregation
    #aggregator = PCTPCAggregator(data, verbose=True)
    #clusters = aggregator.aggregate() 
    # clusters is a list of dictionaries, each dictionary contains "centroid", "vectors" and "priority"
    # concatenating all the vectors in the clusters recreates the 'data' from the previous step

    # TODO: Illustrate the data

    # TODO: Export the results
    # create new files in the same format as the input files, but with the aggregated data
    pass

def from_df(df, index_method='first', columns_for_priority=None, columns_for_similarity=None, clusters_nr_final=None, verbose=False, **kwargs):
    """
    Function for aggregating data from a pandas DataFrame.
    :param df: pandas DataFrame
    :param index: str, optional
        Index-method for the output DataFrame.
        'first' - the first index from the original data is used [default]
        'last' - the last index from the original data is used
        'all' - all indices from the original data are used
        'span' - if the original data has more than one index, the index will be "{first}-{last}"
    :param columns_for_priority: list of header indices to use when assigning priority, optional
    :param columns_for_similarity: list of header indices to use when calculating similarity, optional
        If a tuple is provided in the list, the values corresponding to the indices in the tuple are averaged,
            e.g. [0,(1,2)] will use calculate dissimilarity based on [v1, (v2 + v3)/2]
        If a dictionary is provided, the values corresponding to the key are the indices, and the value is the corresponding weight,
            e.g. {0:1,1:2} will use calculate dissimilarity based on [v1, 2*v2]
    :param clusters_nr_final: int, optional
        Number of clusters to generate. If None, the number of clusters is set to half.

    """
    data, header, index, scaling_factors = process_indata(df)
    aggregator = PCTPCAggregator(data, columns_for_priority=columns_for_priority, columns_for_similarity=columns_for_similarity, 
                                 clusters_nr_final=clusters_nr_final, verbose=verbose, **kwargs)
    clusters = aggregator.aggregate() 
    new_df = process_outdata_clusters(clusters, header, scaling_factors, index_method=index_method, ref_index=index)
    weights = [len(clusters[i]['vectors']) for i in range(len(clusters))]
    return new_df, weights


if __name__ == "__main__":
    main()
