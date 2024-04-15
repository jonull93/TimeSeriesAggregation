import pandas as pd
import numpy as np
from TSA.utils import print_red

def process_indata(data):
    def process_df(df):
        # take df, return numpy array, header and index (so that we can reconstruct the df later)
        arr = df.to_numpy()
        header = df.columns
        index = df.index
        scaling_factor = [max(abs(arr[:,i])) for i in range(arr.shape[1])]
        arr = arr / scaling_factor
        return arr, header, index, scaling_factor

    if isinstance(data, pd.DataFrame):
        return process_df(data)
    else:
        raise NotImplementedError('Data type not supported. Please provide a pandas DataFrame.')

def process_outdata_clusters(clusters:list, header, scaling_factors, ref_index=None, index_method:str='first'):
    if type(ref_index) not in [pd.Index, list, np.ndarray, range]:
        ref_index = range(len(clusters))
        print_red('Warning in process_outdata_clusters(): ref_index not provided. ref_index set to range(len(clusters)).')

    if index_method in ['first', 'last', 'all', 'span']:
        #build index from original_indices
        #if len(original_indices)>1, the make the index "{first_index}-{last_index}"
        built_index = []
        for cluster in clusters:
            i = cluster['original_indices']
            if index_method == 'first':
                ind = i[0] # index to use from ref_index
                built_index.append(ref_index[ind])
            elif index_method == 'last':
                built_index.append(i[-1])
                built_index.append(ref_index[ind])
            elif index_method == 'all':
                built_index.append(';'.join([str(ref_index[ind]) for ind in i]))
            elif index_method == 'span':
                if len(i)>1:
                    built_index.append(f'{ref_index[i[0]]}-{ref_index[i[-1]]}')
                else:
                    built_index.append(ref_index[i[0]])
    else:
        raise ValueError('Invalid index_method value. Please provide "first", "last", "all" or "span".')

    data = np.array([clusters[i]['centroid'] for i in range(len(clusters))])*scaling_factors
        
    """if len(data)>len(index_method):
        index_method = range(len(data))
        print_red('Warning in process_outdata(): Index length does not match data length. Index set to range(len(data)).')"""

    return pd.DataFrame(data, columns=header, index=built_index)


def process_outdata_array(data:np.ndarray, header, scaling_factor, index=None):
    if type(index) not in [pd.Index, list, np.ndarray, range]:
        index = range(len(data))
        print_red('Warning in process_outdata_array(): Index not provided. Index set to range(len(data)).')

    return pd.DataFrame(data*scaling_factor, columns=header, index=index)