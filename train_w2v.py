


"""
    Script for training a Word2Vec model on default parameters and saving a binary file.

    TODO:
     - Load extracted text
     -
"""

import nltk
import pandas as pd

from gensim.models import Word2Vec


def main():
    vec_dim = 100

    tokenize = nltk.tokenize.RegexpTokenizer('(?u)\\b\\w\\w+\\b')
    stopwords = nltk.corpus.stopwords.words('english')

    df = pd.read_feather('data/all_data.feather')

    df

    bins = [0, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
    df['binned_year'] = pd.cut(df['year'], bins=bins)

    cuts = df['binned_year'].unique()

    for cut in cuts:

        original_texts = df[df.binned_year==cut].abstract.values
        preprocessed_texts = []

        for text in original_texts:
            original_tokens = tokenize.tokenize(text)
            tokens = [w.lower() for w in original_tokens if w.lower() not in stopwords and
                      not w.isnumeric() and len(w) > 1]
            preprocessed_texts.append(tokens)

        # input format
        # preprocessed_texts = [['casa', 'maria', 'jose'], ['nlp', 'texto', 'agora']]

        model = Word2Vec(size=vec_dim, window=5, min_count=1, workers=4, sg=1)

        model.build_vocab(preprocessed_texts, progress_per=10000)

        model.train(preprocessed_texts, total_examples=len(preprocessed_texts), epochs=10)

        # Store for posterior training
        model.save("data/models/word2vec_{}.model".format(cut))

        # Save keyed vectors
        model.wv.save("data/models/wordvectors_{}.kv".format(cut))


        """
        from gensim.models import KeyedVectors
        wv = KeyedVectors.load("wordvectors_{}.kv".format(''))
        """

if __name__ == '__main__':
    main()
