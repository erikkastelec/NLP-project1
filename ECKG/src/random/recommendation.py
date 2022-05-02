import networkx as nx
from networkx.exception import NetworkXNoPath, NodeNotFound

from helper_functions import read_pickle

path_to_pickle = 'analysis.pickle'
pickle_graph = 'graph.pickle'


def entities_from_article(a):
    nodes = []
    check = set()
    sum = 0
    sump = 0
    sumo = 0

    for x in a['org']:
        nodes.append([x[0], x[1], 1, 0, 0])
        sum += int(x[1])
        sumo += int(x[1])
        check.add(x[0])

    for x in a['per']:
        if x[0] not in check:
            nodes.append([x[0], x[1], 0, 0, 0])
            sum += int(x[1])
            sump += int(x[1])
            check.add(x[0])

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

    return check, nodes, (sum, sump, sumo)


def get_similarity(id, articles, g):
    a = articles[id]
    entities, vals, (sum, sump, sumo) = entities_from_article(a)

    subgraph = g.subgraph(list(entities))
    sim = []
    for i, a in enumerate(articles):
        if i != id:
            e, v, (s, sp, so) = entities_from_article(a)
            s = g.subgraph(list(e))
            # subgraph expansion

            score = 0
            sim.append(id, score)


def add_neighbors(a, graph):
    res = a.copy()
    for node in list(a):
        try:
            # Sort neighbors by number of co-occurrences
            neighbors = sorted(graph[node].items(), key=lambda edge: edge[1]['weight'])
            for neighbor in neighbors[:5]:
                res.add(neighbor[0])
        except KeyError:
            continue
    return res


def relational_weighting_scheme(graph, nodes):
    res_graph = nx.Graph()

    # print(max([x[1] for x in graph.degree]))
    for i in range(0, len(nodes)):
        # Construct a set from neighbor list
        a = set(nx.neighbors(graph, nodes[i]))

        # for a1 in a.copy():
        #     for neig in nx.neighbors(graph,a1):
        #         a.add(neig)
        count_sum_a = sum([graph[nodes[i]][x]["count"] for x in a])
        occ_count_a = sum([graph[nodes[i]][x]["occ"] for x in a])
        for j in range(i + 1, len(nodes)):
            if nodes[i] == nodes[j]:
                continue
            b = set(nx.neighbors(graph, nodes[j]))
            count_sum_b = sum([graph[nodes[j]][x]["count"] for x in b])
            occ_count_b = sum([graph[nodes[j]][x]["occ"] for x in b])
            try:
                # Combined weights
                intersection = a.intersection(b)
                tmp = 0
                for inter in intersection:
                    tmp += graph[nodes[i]][inter]["occ"] / occ_count_a
                    tmp += graph[nodes[j]][inter]["occ"] / occ_count_b
                cond_prob = max(1 - (tmp / 2.0), 1e-16)
                res_graph.add_edge(nodes[i], nodes[j], weight=cond_prob)
                # Weighted our
                # intersection = a.intersection(b)
                # tmp = 0
                # for inter in intersection:
                #     tmp += graph[nodes[i]][inter]["occ"] / count_sum_b
                #     tmp += graph[nodes[j]][inter]["occ"] / count_sum_b
                #
                # cond_prob = max(1 - tmp / 2.0, 1e-16)
                # res_graph.add_edge(nodes[i], nodes[j], weight=cond_prob)

                # Weighted
                # intersection = a.intersection(b)
                # tmp = 0
                # for inter in intersection:
                #     tmp += graph[nodes[i]][inter]["count"] / count_sum_a
                #     tmp += graph[nodes[j]][inter]["count"] / count_sum_b
                #
                # cond_prob = max(1 - tmp / 2.0, 1e-16)
                # res_graph.add_edge(nodes[i], nodes[j], weight=cond_prob)
                # Simple
                # intersection = a.intersection(b)
                # cond_prob = 1 - (len(intersection) / len(a) + len(intersection) / len(b)) / 2
                # res_graph.add_edge(nodes[i], nodes[j], weight=cond_prob)
            except ZeroDivisionError:
                continue
    return res_graph


def get_article_similarity(a1, a2, graph):
    # Get nodes for article
    a1 = a1[0]
    a2 = a2[0]
    if len(a1) == 0 or len(a2) == 0:
        return 0, []
    # Add a few neighbors
    # a1_plus = add_neighbors(a1, graph)
    # a2_plus = add_neighbors(a2, graph)
    subgraph = relational_weighting_scheme(graph, list(a1) + list(a2))
    distances1 = {}
    distances2 = {}
    score1 = 0
    score2 = 0
    for node1 in a1:
        distances1[node1] = []
        for node2 in a2:
            if node1 == node2:
                distances1[node1].append(0)
                continue
            try:
                distances1[node1].append(nx.shortest_path_length(subgraph, source=node1, target=node2, weight='weight'))
            except (NetworkXNoPath, NodeNotFound):
                distances1[node1].append(1)
        score1 = score1 + min(distances1[node1])
        distances1[node1] = min(distances1[node1])
    for node1 in a2:
        distances2[node1] = []
        for node2 in a1:
            if node1 == node2:
                distances2[node1].append(0)
                continue
            try:
                distances2[node1].append(nx.shortest_path_length(subgraph, source=node1, target=node2, weight='weight'))
            except (NetworkXNoPath, NodeNotFound):
                distances2[node1].append(1)
        score2 = score2 + min(distances2[node1])
        distances2[node1] = min(distances2[node1])

    score = min(1, max(1 - ((score1 / len(a1)) + (score2 / len(a2))) / 2, 0))
    return score, distances1, distances2


if __name__ == '__main__':
    CNREC_PATH = './data/CNRec'
    pickle_graph = CNREC_PATH + '/graph.pickle'
    path_to_pickle = CNREC_PATH + '/analysis_cnrec.pickle'
    g = read_pickle(pickle_graph)
    articles = read_pickle(path_to_pickle)
    a1 = entities_from_article(articles[25])
    a2 = entities_from_article(articles[96])
    score = get_article_similarity(a1, a2, g)
    print("25,96 ", score)
    print("Proper score: 0.08333")
    a1 = entities_from_article(articles[25])
    a2 = entities_from_article(articles[227])
    score = get_article_similarity(a1, a2, g)
    print("25,227 ", score)
    print("Proper score: 0.00000")
