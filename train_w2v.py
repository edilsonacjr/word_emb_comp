


"""
    Script for training a Word2Vec model on default parameters and saving a binary file.

    TODO:
     - Load extracted text
     -
"""

from gensim.models import Word2Vec


def main():

    all_texts = [['casa', 'maria', 'jose'], ['nlp', 'texto', 'agora']]

    model = Word2Vec(all_texts, size=100, window=5, min_count=1, workers=4)

    model.train([["hello", "world"]], total_examples=1, epochs=10)

    # Store for posterior training
    model.save("word2vec.model")

    # Save keyed vectors
    model.wv.save("wordvectors.kv")


if __name__ == '__main__':
    main()
