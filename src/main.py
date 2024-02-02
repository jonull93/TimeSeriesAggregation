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


def main():
    # TODO: Parse command-line arguments
    profiles, weights, regions_to_aggregate = parse_arguments()
    # TODO: Load data using data_importer
    ugly_data = import_data(profiles, weights, regions_to_aggregate)
    # TODO: Process data if necessary
    data = process_data(ugly_data)
    # TODO: Apply the selected aggregation algorithm
    # to execute the algorithm, call the aggregate method through "aggregator = PCTPCAggregator(data, verbose=True); clusters = aggregator.aggregate()"
    aggregator = PCTPCAggregator(data, verbose=True)
    clusters = aggregator.aggregate() # clusters is a list of dictionaries, each dictionary contains "centroid", "vectors" and "priority"
    # TODO: Export the results
    pass

if __name__ == "__main__":
    main()
