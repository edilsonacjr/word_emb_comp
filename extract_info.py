

"""
    Script that gets a pajek file of a citation network and generates a file of all abstracts and a file of citations,
    both binary.
"""

import pandas as pd

from xnet import xnet2igraph


def main():
    file_path = '../data/wosAPSWithPACS_WithMAG_raw.xnet'

    g = xnet2igraph(file_path)

    data = {
        'abstract': g.vs['Title and Abstract'],
        'year': g.vs['Year Published'],
        'language': g.vs['Language'],
        'adj_list': g.get_adjlist()
    }

    df = pd.DataFrame.from_dict(data)
    df['adj_list'] = df['adj_list'].astype(str)

    df.to_feather('data/all_data.feather')
    #df.to_hdf('data/all_data.h5', 's', complib='blosc')


if __name__ == '__main__':
    main()
