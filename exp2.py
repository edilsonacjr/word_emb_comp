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

        # calculate cosine similarity between each pair
        sim_fir = cosine_similarity(matrix_fir)

        # get the first 20 most similar words for each word
        knn_mat = sim_fir.argsort(axis=1)[:, -20:].copy()


        # build knn similarity matrix
        knn_fir_sim = np.take_along_axis(sim_fir, knn_mat, axis=1)

        del sim_fir

        sim_sec = cosine_similarity(aligned_sec)

        # build knn similarity matrix
        knn_sec_sim = np.take_along_axis(sim_sec, knn_mat, axis=1)

        del sim_sec

        # take second order vector similarity
        time_sim = cosine_similarity(knn_fir_sim, knn_sec_sim)

        time_sim.diagonal().argsort()

        np.array(similar_vocab).take(time_sim.diagonal().argsort())[:100]

        time_sim.take(time_sim.diagonal().argsort())[:100]

        """
        array(['continue', 'mld', 'ded', 'instable', 'fbr', 'nep', 'cts', 'cse',
       'nol', 'cpm', 'ssh', 'pme', 'gic', 'mds', 'gi', 'dune', 'idp',
       'mcs', 'pps', 'pcm', 'dir', 'snake', 'gdp', 'arrows', 'ret', 'pfs',
       'commensurable', 'shs', 'cda', 'als', 'hands', 'sos', 'ie', 'tfs',
       'diff', 'free', 'hss', 'mdm', 'tgc', 'gotten', 'nts', 'fa',
       'conclusion_holds', 'attenuations', 'sud', 'sgc', 'theses', 'hk',
       'course', 'ha', 'cps', 'pdm', 'cbe', 'rp', 'chopping', 'centrally',
       'mfl', 'na2v3o7', 'ws', 'fog', 'ccw', 'fractionally', 'defend',
       'mpd', 'sbr', 'attending', 'vf', 'departing', 'hereby', 'acf',
       'ers', 'undressing', 'tet', 'ssg', 'spare', 'ehs', 'vc', 'esd',
       'pll', 'dash', 'ipa', 'dar', 'interactive', 'shockwave',
       'commutation', 'mdi', 'cq', 'noncompensated', 'fis',
       'took_account', 'hmm', 's4', 'superclusters', 'sep', 'ptc', 'meta',
       'kuhn', 'lamb_dip', 'wood', '5g'], dtype='<U38')
        
        
        
        array([0.77950513, 0.8530327 , 0.8447501 , 0.87801754, 0.84060395,
       0.8790013 , 0.8814865 , 0.881638  , 0.8915266 , 0.874728  ,
       0.8857495 , 0.88604075, 0.8953358 , 0.8907889 , 0.87953454,
       0.8847911 , 0.9057933 , 0.9004386 , 0.8768155 , 0.88580585,
       0.89865583, 0.89725345, 0.9064925 , 0.91304237, 0.90392023,
       0.9086914 , 0.91933376, 0.90871376, 0.92898995, 0.9293774 ,
       0.91554284, 0.895836  , 0.9034303 , 0.9246509 , 0.92214733,
       0.89284503, 0.92192185, 0.91018903, 0.93590736, 0.9327579 ,
       0.9236227 , 0.9223017 , 0.9179727 , 0.92706764, 0.9242237 ,
       0.928583  , 0.9273435 , 0.9134901 , 0.9146059 , 0.91296226,
       0.92089045, 0.92603624, 0.93669224, 0.9113691 , 0.9315828 ,
       0.9273227 , 0.9295304 , 0.93453664, 0.91189563, 0.9317482 ,
       0.9369303 , 0.93624884, 0.94309866, 0.9514135 , 0.94366914,
       0.93989426, 0.9307546 , 0.93007696, 0.9254522 , 0.9320783 ,
       0.93590987, 0.9400435 , 0.94172764, 0.9392563 , 0.94138026,
       0.94825757, 0.9259351 , 0.93552524, 0.94638973, 0.9496513 ,
       0.94697887, 0.94828784, 0.9327371 , 0.94856995, 0.9423865 ,
       0.94011116, 0.93735397, 0.95065236, 0.9387233 , 0.9526518 ,
       0.94820553, 0.9375488 , 0.9491018 , 0.93550926, 0.9400458 ,
       0.9350416 , 0.94495714, 0.95591855, 0.9443899 , 0.9416945 ],
      dtype=float32)

        """

        # build a vector of similarity
        # get the top 10


if __name__ == '__main__':
    main()
