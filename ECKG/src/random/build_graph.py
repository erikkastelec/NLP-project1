import time
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import pylab

from helper_functions import read_pickle, write_pickle

path_to_pickle = './analysis.pickle'
pickle_graph = './graph.pickle'


def get_nodes(article):
    nodes = []
    check = set()
    sum = 0
    sump = 0
    sumo = 0

    for x in article['org']:
        nodes.append([x[0], x[1], 1, 0, 0])
        sum += int(x[1])
        sumo += int(x[1])
        check.add(x[0])

    for x in article['per']:
        if x[0] not in check:
            nodes.append([x[0], x[1], 0, 0, 0])
            sum += int(x[1])
            sump += int(x[1])

    c = 0
    for n in nodes:
        x = int(n[1])
        if n[2] == 0:
            n[3] = x / sump
            n[4] = x / sum
        elif n[2] == 1:
            n[3] = x / sumo
            n[4] = x / sum
        c += 1

    return nodes, (sum, sump, sumo)


def get_edges(nodes):
    es = []
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if j > i:
                #### SUBJECT TO CHANGE
                # w = n1[3] * n1[4] + n2[3] * n2[4]
                ####
                es.append((n1[0], n2[0]))
    return es


def timex(articles):
    start = time.time()
    g, co = build_graph(articles)
    end = time.time()
    return g, co, end - start


def build_graph(articles):
    #### RESTRUCTURE
    ## ID as node name, keep dict of nodename->ID?
    g = nx.Graph()
    for a in articles:
        z = get_nodes(a)
        nodes, (sum, sump, sumo) = z
        for x in nodes:
            if g.has_node(x[0]):
                g.nodes[x[0]]['count'] += 1
            else:
                #### PROBLEM WITH SAME NAME DIF CATEGORY
                g.add_node(x[0], count=1, cat=x[2])
        edges = get_edges(nodes)
        for e in edges:
            if g.has_edge(*e):
                g[e[0]][e[1]]['count'] += 1
            else:
                g.add_edge(e[0], e[1], count=1, occ=0)

    #### Everything linked start to set weights

    co_occ = []
    max_count = 0;
    for e1, e2, d in g.edges(data=True):
        n1 = g.nodes[e1]['count']
        n2 = g.nodes[e2]['count']
        if d['count'] > max_count:
            max_count = d['count']
        ### SUBJECT TO CHANGE
        term = (d['count'] / n1) + (d['count'] / n2)
        term /= 2
        ###
        g[e1][e2]['occ'] = term
        t = (e1, e2, term)
        co_occ.append(t)

    tmp_dict = {}
    for e1, e2, d in g.edges(data=True):
        tmp_dict[(e1, e2)] = abs(d['count'] - max_count + 1) / max_count
    nx.set_edge_attributes(g, tmp_dict, "weight")
    # Normalize weight num_connections/
    return g, co_occ


def filter_graph(g, deg_cutoff, count_cutoff):
    to_remove = []
    for n in g.nodes:
        if g.degree[n] <= deg_cutoff:
            to_remove.append(n)
        elif g.nodes[n]['count'] <= count_cutoff:
            to_remove.append(n)
    g.remove_nodes_from(to_remove)


def deg_dist(degrees):
    deg = [x[1] for x in degrees]
    N = len(deg)
    seq = deg
    degCount = Counter(seq)
    pk = []
    pkx = []
    for d in degCount:
        pk.append(degCount[d] / N)
        pkx.append(d)
    plot([pk, pkx])


def plot(d):
    fig, ax = plt.subplots()
    plt.plot(d[1], d[0], 'b.', label='Deg. dist.')
    ax.set_ylabel('pk')
    ax.set_xlabel('k')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    pylab.show()


if __name__ == '__main__':
    try:
        g = read_pickle(pickle_graph)
    except FileNotFoundError:
        articles = read_pickle(path_to_pickle)
        g, co = build_graph(articles)
        write_pickle(g, pickle_graph)
    temp = sorted(g.degree, key=lambda x: x[1], reverse=True)
    deg_dist(g.degree)
    print(len(g))
    print(max([x[2]['occ'] for x in list(g.edges.data())]))
    print(max([x[2] for x in co]))
    filter_graph(g, 10, 1)
    print(len(g))
    deg_dist(g.degree)

    print(max([x[2]['occ'] for x in list(g.edges.data())]))
    print(max([x[2] for x in co]))

    # g2 = pickle.load(open(path_to_pickle, 'rb'))
    # nx.draw_networkx(g, with_labels=False)
    # plt.show()
