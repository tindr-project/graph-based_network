# Embedding + DL architecture: H. Gaspar 2018 hagax8@gmail.com
# node2vec functions were taken from the node2vec project

from ugtm import eGTM
from ugtm import landscape
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import numpy as np
from CNN_graph_embedding import proteinCNN
import argparse

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run Graph-GTM CNN.")

    parser.add_argument('--input', nargs='?', default='list_train',
                        help='Graph list')

    parser.add_argument('--output', nargs='?', default='output',
                        help='Output path')

    parser.add_argument('--labels', nargs='?', default='random_labels',
                        help='Output path')

    parser.add_argument('--colors', nargs='?', default='example_colors',
                        help='Output path')

    parser.add_argument('--grid_size', type=int, default=8,
                        help='Grid size for CNN input. Default is 8 (8x8 image).')

    return parser.parse_args()

def read_graph(inputgraph, weighted, directed):
    '''
    Reads edge list and create networkx graph
    '''
    if weighted:
        G = nx.read_edgelist(inputgraph,
                             nodetype=str, data=(('weight', float),),
                             create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(inputgraph,
                             nodetype=str,
                             create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    if not directed:
        G = G.to_undirected()
    return G


def learn_embeddings(walks, dimensions=128, window_size=5, workers=8, iter=5):
    '''
    Learns embeddings using node2vec random walk.
    '''
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=dimensions,
                     window=window_size,
                     min_count=0, sg=1, workers=workers, iter=iter)
    model.wv.save_word2vec_format('test')
#    print(embedding_matrix.shape)
    embedding_matrix = model.wv.syn0
    words = model.wv.index2word
    # print(words_colors)
    # print(words)
    return words, embedding_matrix


def doNode2Vec(inputgraph, weighted=False, directed=False,
               dimensions=128, window_size=5,
               workers=8, num_walks=10,
               walk_length=80, iter=5, p=1, q=1):
    '''
    node2vec pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph(inputgraph, weighted, directed)
    G = node2vec.Graph(nx_G, directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    words, embeddings = learn_embeddings(walks)
    return words, embeddings


def runTransform(embeddings, k=8, m=4, regul=0.1, s=0.3):
    '''
    Runs GTM dimensionality reduction.
    '''
    gtmodel = eGTM(k=k, m=m, regul=regul, s=s).fit(embeddings).optimizedModel
    return gtmodel


def graphEmbedding(inputgraph, weighted=False, directed=False,
                   dimensions=128, window_size=5,
                   workers=8, num_walks=10,
                   walk_length=80, iter=5, p=1, q=1,
                   k=8, m=4, regul=0.1, s=0.3):
    '''
    Runs node2vec and GTM dimensionality reduction.
    '''
    words, embeddings = doNode2Vec(inputgraph, weighted=weighted,
                                   directed=directed,
                                   dimensions=dimensions,
                                   window_size=window_size,
                                   workers=workers, num_walks=num_walks,
                                   walk_length=walk_length,
                                   iter=iter, p=p, q=q)
    transformed = runTransform(embeddings, k=k, m=m, regul=regul, s=s)
    return words, transformed


def colorGraph(transformed, words, colors):
    '''
    Creates coloured graph from GTM dimensionality reduction output.
    '''
    def checkcolor(x,words,colors):
        if x in colors:
            return colors[x]
        else:
            return 0.0
    words_colors = np.array([checkcolor(word,words,colors) for word in words]).astype(float)
    colored_graph = landscape(transformed, words_colors)
    return words_colors, colored_graph


def getImage(graph, k):
    '''
    Creates 2D matrix from meshgrid labels (graph) and meshgrid size (k)
    '''
    vec = []
    j = k
    for i in range(k):
        vec.append([graph[(x*j)-i-1] for x in range(1, k+1)])
    return(vec)


def mainworkflow(graph_list, color_file, labels, directed=True, weighted=True, output="out", k=8):
    '''
    Creation of embedding and network training
    '''
    with open(color_file, 'r') as file:
        inline = [line.strip().split(" ") for line in file]
    colors = {k: v for k, v in inline}

    count = 0
    with open(graph_list, 'r') as file:
        xnametrain = [line.strip() for line in file]

    with open(labels, 'r') as file:
        lines = file.readlines()
        for inputgraph in xnametrain:
            try:
                words, transformed = graphEmbedding(inputgraph,
                                                    directed=directed,
                                                    weighted=weighted,
                                                    k=k)
                density = getImage(np.sum(transformed.matR, axis=0), k)
                print(density)
                words_colors, colorgraph = colorGraph(transformed, words, colors)
                #transformed.plot_html(output="foo", labels=words_colors,ids=words)
                colorgraph = getImage(colorgraph, k)
                stacked = np.dstack((density, colorgraph))
                stacked = np.expand_dims(stacked, 0)
                line = lines[count]
                if count == 0:
                    X = stacked
                    Y = [line]
                else:
                    X = np.concatenate((X, stacked), axis=0)
                    Y.append(line)
                count += 1
            except:
                print("Error. Not processing "+str(inputgraph)+".")
#    with open(labels, 'r') as file:
#        Y = [line.strip() for line in file]
    Y = np.squeeze(Y)
    proteinCNN(X, Y, output)


args = parse_args()
mainworkflow(graph_list=args.input,
             output=args.output,
             k=args.grid_size,
             color_file=args.colors,
             labels=args.labels)
