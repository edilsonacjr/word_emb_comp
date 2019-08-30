


"""
    Script that gets a pajek file of a citation network and generates a file of all abstracts and a file of citations,
    both binary.

    $ python

    pickle file of
"""

from collections import defaultdict

from xnet import xnet2igraph


def main():
    file_path = 'data/wosAPSWithPACS_WithMAG_raw.xnet'

    g = xnet2igraph(file_path)

    data = {
        'abstract': [],
        'year': [],
        'language': [],
        'hasabs': [],
        'doi': [],
    }

    info_pairs = [('abstract', '#v "Title and Abstract" s\n'),
                  ('year', '#v "Year Published" n\n'),
                  ('language', '#v "Language" s\n'),
                  ('hasabs', '#v "hasAbstract" s\n'),
                  ('doi', '#v "Digital Object Identifier (DOI)" s\n')]

    edges = defaultdict(list)

    for dic_key, info_name in info_pairs:
        with open(file_path, 'r') as origin_file:
            read_flag = False
            for line in origin_file:
                if read_flag and line[:2] != '#v':
                    data[dic_key].append(line.strip())

                if line == info_name:
                    read_flag = True
                elif read_flag and line[:2] == '#v':
                    break


if __name__ == '__main__':
    main()


