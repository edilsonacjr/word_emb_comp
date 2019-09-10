
"""
    What happens to the distance between words in time?
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict

from gensim.models import KeyedVectors

plt.rcParams["figure.figsize"] = [15, 40]
sns.set_style("dark")


def main():

    words_study = ['gene']
    words_reference = ['molecular', 'plant', 'compound', 'cell', 'inhibitor']

    bins = [0, 1960, 1970, 1980, 1990, 2000, 2010, 2020]

    cuts = pd.cut(bins, bins=bins)[1:]

    to_plot = {}

    for ind, cut in enumerate(cuts):
        wv = KeyedVectors.load("data/models/wordvectors_{}.kv".format(cut))
        for word in words_study:
            if word not in to_plot:
                to_plot[word] = defaultdict(list)
            for word_ref in words_reference:
                if word_ref in wv:
                    sim = wv.similarity(word, word_ref)
                    to_plot[word][cut].append(sim)
                else:
                    to_plot[word][cut].append(0)

    fig, ax = plt.subplots(7, 1)

    for word in words_study:
        for ind, cut in enumerate(cuts):
            for ind_ref, word_ref in enumerate(words_reference):
                ax[ind].barh(word_ref, to_plot[word][cut][ind_ref], 0.1)
            ax[ind].set_title(str(cut), fontsize=20)
            ax[ind].set_xlim(0, 1)
        fig.savefig('plots/word_{}.png'.format(word))


if __name__ == '__main__':
    main()
