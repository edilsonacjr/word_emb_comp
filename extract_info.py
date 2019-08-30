


"""
    Script that gets a pajek file of a citation network and generates a file of all abstracts and a file of citations,
    both binary.

    $ python

    pickle file of
"""

import pandas as pd

from collections import defaultdict

from xnet import xnet2igraph


def main():
    file_path = '../data/wosAPSWithPACS_WithMAG_raw.xnet'

    g = xnet2igraph(file_path)

    data = {
        'abstract': g.vs['Title and Abstract'],
        'year': g.vs['Year Published'],
        'language': g.vs['Language'],
    }

    df = pd.DataFrame.from_dict(data)

    df.to_feather('data/all_data.feather')


if __name__ == '__main__':
    main()


