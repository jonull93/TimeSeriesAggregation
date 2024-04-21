import matplotlib.pyplot as plt
import pandas as pd
from TSA.data_processor import clusters_to_df, decompress_df

def create_plots(original_df, clusters, columns_to_plot=None, fig_path:str=None, show_fig=True, show_priority=True):
    """
    Function for creating plots from the aggregated data.
    :param original_df: pandas DataFrame 
    :param clusters: list of dictionaries
        Each dictionary contains "centroid", "vectors" and "priority"
    :param weights: list of integers
        Number of vectors in each cluster
    """
    if columns_to_plot is None:
        columns_to_plot = original_df.columns

    scaling_factors = [max(abs(original_df[column])) for column in original_df.columns]
    compressed_df = clusters_to_df(clusters, original_df.columns, scaling_factors, ref_index=original_df.index)
    weights = [len(clusters[i]['vectors']) for i in range(len(clusters))]
    priority = [clusters[i]['priority'] for i in range(len(clusters))]
    compressed_df['Priority'] = priority
    compressed_df['Weights'] = weights
    decompressed_df = decompress_df(compressed_df)

    # Plot the original and decompressed data. First column as subplot 1 and the rest as subplot 2 (vertically stacked subplots)
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].plot(original_df[columns_to_plot[0]], label=columns_to_plot[0])
    ax[0].plot(decompressed_df[columns_to_plot[0]], label='Agg. ' + columns_to_plot[0])
    if show_priority:
        # If priority is shown, the priority is shown as a vertical low-opacity colored box covering the consequtive values of the same priority
        priority = decompressed_df['Priority']
        priority_colors = {1: 'None', 2: 'yellow', 3: 'red', 4: 'black', 5: 'purple'}
        changes = priority.ne(priority.shift(1))
        boundaries = changes[changes].index.tolist()
        if boundaries[-1] != len(priority):
            boundaries.append(len(priority))
        x_values = [(boundaries[i], boundaries[i+1]-1) for i in range(len(boundaries)-1)]
        for i in range(len(x_values)):
            ax[0].axvspan(x_values[i][0], x_values[i][1], color=priority_colors[priority[x_values[i][0]]], alpha=0.2)
    ax[0].legend()
    ax[0].set_title(f'Column 1: {columns_to_plot[0]}')
    ax[0].set_xlabel('Time')

    for column in columns_to_plot[1:]:
        ax[1].plot(original_df[column], label=column)
        ax[1].plot(decompressed_df[column], label='Agg. ' + column)
    ax[1].legend()
    if len(columns_to_plot) > 2:
        ax[1].set_title(f'Columns 2-{len(columns_to_plot)}')
    else: 
        ax[1].set_title(f'Column 2: {columns_to_plot[1]}')
    ax[1].set_xlabel('Time')

    fig.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path)
    if show_fig:
        plt.show()
    
    return

if __name__ == "__main__":
    # Test the function
    import pickle
    from TSA.data_processor import process_indata
    from TSA.algorithms.pctpc import PCTPCAggregator
    main_dir_path = __file__.split('src')[0]
    #clusters = pickle.load(open(main_dir_path + 'tests/clusters.pickle', 'rb'))
    original_df = pd.read_pickle(main_dir_path + 'tests/original_df.pickle')
    original_df = original_df[["Load", "Temp"]]
    data, header, index, scaling_factors = process_indata(original_df)
    aggregator = PCTPCAggregator(data, columns_for_similarity={0:1, 1:0.15}, columns_for_priority=[0],
                                 global_prio_period=48, verbose=True)
    clusters = aggregator.aggregate() 
    new_df = clusters_to_df(clusters, header, scaling_factors, ref_index=index)
    weights = [len(clusters[i]['vectors']) for i in range(len(clusters))]
    columns_to_plot = ['Load', 'Temp']
    create_plots(original_df, clusters, show_priority=True, columns_to_plot=columns_to_plot)
    print("-- Plots created --")