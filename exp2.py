"""
    What happens to the distance between words in time?

"""

import pandas as pd
import numpy as np
import hypertools as hyp

from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity


def main():

    bins = [0, 1960, 1970, 1980, 1990, 2000, 2010, 2020]

    cuts = pd.cut(bins, bins=bins)[1:]

    cuts = cuts[-2:]

    for ind, cut in enumerate(cuts[:]):
        # load word2vecs
        wv_first = KeyedVectors.load("data/models/wordvectors_{}.kv".format(cut))
        wv_second = KeyedVectors.load("data/models/wordvectors_{}.kv".format(cuts[ind+1]))

        # get similar vocab
        similar_vocab = [word for word in wv_first.vocab if word in wv_second.vocab]

        # get matrices on similar vocab
        matrix_fir = np.array([wv_first.get_vector(word) for word in similar_vocab])
        matrix_sec = np.array([wv_second.get_vector(word) for word in similar_vocab])

        # align the second matrix to the first one
        aligned_sec = hyp.tools.procrustes(matrix_sec, matrix_fir)

        sim_fir = cosine_similarity(matrix_fir)
        knn_mat = np.array([sim_vec.argsort()[-20:] for sim_vec in sim_fir])

        del sim_fir


        # calculate knn
        # build a vector of similarity
        # get the top 10


if __name__ == '__main__':
    main()
