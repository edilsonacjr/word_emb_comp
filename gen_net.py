

"""
    Script that generates a .net file

    TODO
    - weights of the edges
    - simplify or not?

"""

import ast
import nltk
import pandas as pd
import igraph
import pickle


def main():

    tokenize = nltk.tokenize.RegexpTokenizer('(?u)\\b\\w\\w+\\b')
    stopwords = nltk.corpus.stopwords.words('english')

    df = pd.read_feather('data/all_data.feather')

    # full conversion if needed
    # df['adj_list'] = df['adj_list'].apply(lambda x: ast.literal_eval(x))

    bins = [0, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
    df['binned_year'] = pd.cut(df['year'], bins=bins)

    cuts = df['binned_year'].unique()

    for cut in cuts:
        df_new = df[df.binned_year==cut]
        vocab = set()

        original_texts = df_new.abstract.values
        preprocessed_texts = []

        for text in original_texts:
            original_tokens = tokenize.tokenize(text)
            tokens = [w.lower() for w in original_tokens if w.lower() not in stopwords and
                      not w.isnumeric() and len(w) > 1]
            preprocessed_texts.append(tokens)
            vocab.update(tokens)

        vocab_map = dict(zip(list(vocab), range(len(vocab))))
        rever_vocab_map = dict(zip(range(len(vocab)), list(vocab)))
        edges = []

        for tokens in preprocessed_texts:
            for ind, token in enumerate(tokens[:-1]):
                edges.append((vocab_map[token], vocab_map[tokens[ind+1]]))

        edges_citation = {}
        papers_edges = df_new.adj_list.values
        indices = df_new.index
        indices_map = dict(zip(list(indices), range(len(indices))))
        for ind_paper, ed_list, tokens in zip(indices, papers_edges, preprocessed_texts):
            adj_list = [ x for x in ast.literal_eval(ed_list) if x in indices]
            for token in tokens:
                for adj_paper in adj_list:
                    for conn_token in preprocessed_texts[indices_map[adj_paper]]:
                        if (vocab_map[token], vocab_map[conn_token]) not in edges_citation:
                            edges_citation[(vocab_map[token], vocab_map[conn_token])] = 1
                        else:
                            edges_citation[(vocab_map[token], vocab_map[conn_token])] += 1
                        #edges_citation.append((vocab_map[token], vocab_map[conn_token]))

        edges_cit = list(edges_citation.keys())
        weights_cit = list(edges_citation.values())

        # create an empty graph
        g = igraph.Graph()

        # add vertices to the graph
        g.add_vertices(len(vocab))

        # add edges to the graph
        g.add_edges(edges + edges_cit)

        # set the weight of every edge to 1
        g.es["weight"] = [1] * len(edges) + weights_cit

        # collapse multiple edges and sum their weights
        g.simplify(combine_edges={"weight": "sum"})

        g.save('data/nets/network_{}.ncol'.format(cut), format='ncol',  weights="weight")
        g.save('data/nets/network_{}.pickle'.format(cut), format='pickle')

        with open('data/vocabs/vocab_{}.pickle'.format(cut), 'wb') as pk_file:
            pickle.dump(vocab_map, pk_file, protocol=pickle.HIGHEST_PROTOCOL)

        with open('data/vocabs/rever_vocab_{}.pickle'.format(cut), 'wb') as pk_file:
            pickle.dump(rever_vocab_map, pk_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
