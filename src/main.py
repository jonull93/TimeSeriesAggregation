# main.py
# This is the main entry point for the application. It should 
# - parse command-line arguments, 
# - load the data, 
# - process it if necessary, then apply the selected aggregation algorithm, and finally
# - export the results. 
# The main function should be called when the script is executed.
from config_manager import parse_arguments
from data_importer import import_data
from data_processor import process_data
from aggregation_algorithms.pctpc import PCTPCAggregator
from data_illustrator import create_plots
from data_exporter import export_data


def main():
    # TODO: Parse command-line arguments
    profiles, weights, regions_to_aggregate, heat_load_to_electrify = parse_arguments()
    # TODO: Load data using data_importer
    ugly_data = import_data(profiles, weights)
    # TODO: Process data if necessary
    data = process_data(ugly_data, heat_load_to_electrify, regions_to_aggregate)

    #combine import_data() and process_data() into black box?
    
    # Ready for testing: PCPTC Aggregation
    aggregator = PCTPCAggregator(data, verbose=True)
    clusters = aggregator.aggregate() 
    # clusters is a list of dictionaries, each dictionary contains "centroid", "vectors" and "priority"
    # concatenating all the vectors in the clusters recreates the 'data' from the previous step

    # TODO: Illustrate the data

    # TODO: Export the results
    # create new files in the same format as the input files, but with the aggregated data
    pass

if __name__ == "__main__":
    main()
