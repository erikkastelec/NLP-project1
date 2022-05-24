import classla
import numpy as np
import stanza
import pickle
import os
from tqdm import tqdm
import sys

from ECKG.src.eventify import Eventify
from ECKG.src.helper_functions import fix_ner, deduplicate_named_entities, get_relations_from_sentences, \
    create_graph_from_pairs, graph_entity_importance_evaluation, get_entities_from_svo_triplets, find_similar, \
    get_relations_from_sentences_sentiment, get_relations_from_sentences_sentiment_verb, group_relations_filter, \
    create_graph_from_pairs_sentiment, group_relations, \
    get_triads, evaluate_triads, get_entities_from_svo_triplets_sentiment, EnglishCorefPipeline, \
    get_relations_from_sentences_coref_sentence_sent
from books.get_data import get_data, Book
from ECKG.src.character import Character
from sentiment.sentiment_analysis import *


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import networkx as nx
import matplotlib.pyplot as plt





def train_classifiers(X_train, X_test, Y_train, Y_test):
    clf1 = DecisionTreeClassifier()
    clf1.fit(X_train, Y_train)

    clf2 = KNeighborsClassifier(n_neighbors=3)
    clf2.fit(X_train, Y_train)

    clf3 = SVC(kernel='linear', C=3)
    clf3.fit(X_train, Y_train)

    clf4 = RandomForestClassifier(n_estimators=100)
    clf4.fit(X_train, Y_train)

    print(f'SVC: {clf3.score(X_test, Y_test)}')
    print(f'KNN: {clf2.score(X_test, Y_test)}')
    print(f'Decision Tree: {clf1.score(X_test, Y_test)}')
    print(f'Random Forest: {clf4.score(X_test, Y_test)}')

    return clf1, clf2, clf3, clf4

def evaluate_prediction_book(book, characters, predictions_p, predictions_p_proba, predictions_a, predictions_a_proba):
    p_true = []
    p_ranks = []
    a_true = []
    a_ranks = []
    p_names = []
    a_names = []

    for protagonist in book.protagonists_names:
        try:
            i = list(characters).index(protagonist)
        except ValueError:
            print(f'P:{protagonist} not found in characters for book {book.title}')
            continue
        p = predictions_p[i]
        p_true.append(p)
        p_ranks.append(predictions_p_proba[i][1])
        p_names.append(protagonist)

    for antagonist in book.antagonists_names:
        try:
            i = list(characters).index(antagonist)
        except ValueError:
            print(f'A:{antagonist} not found in characters for book {book.title}')
            continue
        a = predictions_a[i]
        a_true.append(a)
        a_ranks.append(predictions_a_proba[i][1])
        a_names.append(antagonist)

    return p_true, p_ranks, p_names, a_true, a_ranks, a_names, book

def get_predictions(classifier, X):
    return classifier.predict(X), classifier.predict_proba(X)


def get_characters(sentiment_pairs, names, text):
    relations_sentiment = group_relations(sentiment_pairs)
    relationship_graph = create_graph_from_pairs_sentiment(relations_sentiment)
    triads = get_triads(relationship_graph)
    triad_classes, ts = evaluate_triads(triads)

    characters = []
    for n in names:
        if n not in relationship_graph.nodes:
            continue
        c = Character(n)
        c.degree = relationship_graph.degree(n)
        c.times_appeared = text.upper().count(n.upper())  # TODO: fix when coreference is implemented
        c.negative_t1 = ts[1][n]
        c.positive_t2 = ts[2][n]
        c.t3 = ts[3][n]

        w = []
        wp = []
        wn = []
        for link in list(relationship_graph[n]):
            weight = relationship_graph.get_edge_data(n, link)[0]['weight']
            w.append(weight)
            if weight < 0:
                wn.append(weight)
            else:
                wp.append(weight)
        c.weight_sum = sum(w)
        c.weight_pos_sum = sum(wp)
        c.weight_neg_sum = sum(wn)
        c.pos_neg_diff = sum(wp) - sum(wn)
        characters.append(c)
    return characters, relationship_graph


def evaluate_book(book: Book, pipeline, sa, coref_pipeline=None, verb=True):
    # Run text through pipeline
    data = pipeline(book.text)
    # Fix NER anomalies
    data = fix_ner(data)

    deduplication_mapper, count = deduplicate_named_entities(data, count_entities=True)

    if coref_pipeline:
        coref_pipeline.process_text(book.text)
        coref_chains, new_chains = coref_pipeline.unify_naming_coreferee(deduplication_mapper)

    if verb:
        sentence_relations = get_relations_from_sentences_sentiment_verb(data, deduplication_mapper, sa)
    else:
        #sentence_relations = get_relations_from_sentences_sentiment(data, deduplication_mapper, sa)
        sentence_relations = get_relations_from_sentences_coref_sentence_sent(data, deduplication_mapper, sa, coref_pipeline)
    graph = create_graph_from_pairs(sentence_relations)

    if len(graph.nodes) == 0:
        return [], [], []

    names, values = graph_entity_importance_evaluation(graph)
    characters, relationship_graph = get_characters(sentence_relations, names, book.text)

    return characters, relationship_graph, deduplication_mapper


def evaluate_book_svo(book: Book, pipeline, svo_extractor, sa, coref_pipeline=None):
    # Run text through pipeline
    data = pipeline(book.text)
    # Fix NER anomalies
    data = fix_ner(data)

    deduplication_mapper, count = deduplicate_named_entities(data, count_entities=True)

    if coref_pipeline:
        coref_pipeline.process_text(book.text)
        coref_chains, new_chains = coref_pipeline.unify_naming_coreferee(deduplication_mapper)

    entities_from_svo_triplets, svo_sentiment = get_entities_from_svo_triplets_sentiment(book, svo_extractor,
                                                                                         deduplication_mapper, sa,
                                                                                         doc=data,
                                                                                         coref_pipeline=coref_pipeline)
    svo_triplet_graph = create_graph_from_pairs(entities_from_svo_triplets)
    try:
        names, values = graph_entity_importance_evaluation(svo_triplet_graph)
        characters, relationship_graph = get_characters(svo_sentiment, names, book.text)
    except ValueError:
        characters = []
        relationship_graph = nx.MultiGraph()


    return characters, relationship_graph, deduplication_mapper


def make_dataset(corpus, path, svo, verb):
    # Initialize Slovene classla pipeline
    classla.download('sl')
    sl_pipeline = classla.Pipeline("sl", processors='tokenize, ner, lemma, pos, depparse', use_gpu=True)
    sl_e = Eventify(language="sl")

    # Initialize English stanza pipeline
    stanza.download("en")
    en_pipeline = stanza.Pipeline("en", processors='tokenize, ner, lemma, pos, mwt, depparse', use_gpu=True)
    en_e = Eventify(language="en")
    en_coref_pipeline = EnglishCorefPipeline()

    job = read_lexicon_job(
        "../../sentiment/lexicons/Slovene sentiment lexicon JOB 1.0/Slovene_sentiment_lexicon_JOB.txt")
    kss = convert_list_to_dict(
        read_lexicon_kss('../../sentiment/lexicons/Slovene sentiment lexicon KSS 1.1/')['Slolex'])
    afinn_en = read_lexicon_afinn_en("../../sentiment/lexicons/AFINN_en/AFINN-111.txt")
    sentiwordnet_en = read_lexicon_sentiwordnet("../../sentiment/lexicons/SentiWordNet.txt")

    # SL
    sa_kss = SentimentAnalysis(kss)
    sa_job = SentimentAnalysis(job)

    # EN
    sa_afinn = SentimentAnalysis(afinn_en)
    sa_swn = SentimentAnalysis(sentiwordnet_en)


    sl = []
    en = []
    both = []
    graphs = []
    n_mapper = []

    forbidden = ['The Lord Of The Rings: The Fellowship of the Ring']

    for book in tqdm(corpus, total=len(corpus), file=sys.stdout):
        print(book.title)
        if book.title in forbidden:
            both.append([])
            graphs.append([])
            n_mapper.append([])
            continue

        if book.language == "slovenian":
            if svo:
                characters, graph, ner_mapper = evaluate_book_svo(book, sl_pipeline, sl_e, sa_kss, coref_pipeline=None) # TODO sl coref pipeline
            else:
                characters, graph, ner_mapper = evaluate_book(book, sl_pipeline, sa_kss, coref_pipeline=None, verb=verb)
            sl.append(characters)
            both.append(characters)
            graphs.append(graph)
            n_mapper.append(ner_mapper)
        else:
            if svo:
                characters, graph, ner_mapper = evaluate_book_svo(book, en_pipeline, en_e, sa_swn, coref_pipeline=en_coref_pipeline)
            else:
                characters, graph, ner_mapper = evaluate_book(book, en_pipeline, sa_swn, coref_pipeline=en_coref_pipeline, verb=verb)
            en.append(characters)
            both.append(characters)
            graphs.append(graph)
            n_mapper.append(ner_mapper)


    #with open('characters_sl.pickle', 'wb') as f:
    #    pickle.dump(sl, f, pickle.HIGHEST_PROTOCOL)

    #with open('characters_en.pickle', 'wb') as f:
    #    pickle.dump(en, f, pickle.HIGHEST_PROTOCOL)

    with open(path + 'characters_all.pickle', 'wb') as f:
        pickle.dump(both, f, pickle.HIGHEST_PROTOCOL)

    with open(path + 'graphs.pickle', 'wb') as f:
        pickle.dump(graphs, f, pickle.HIGHEST_PROTOCOL)

    with open(path + 'ner.pickle', 'wb') as f:
        pickle.dump(n_mapper, f, pickle.HIGHEST_PROTOCOL)

    return sl, en, both

def get_features(character):
    features = [character.degree, character.negative_t1, character.positive_t2, character.t3, character.times_appeared,
                character.weight_neg_sum, character.weight_pos_sum]
    return features



def shape_features(characters_all):
    features_all = []
    for characters in characters_all:
        n_char = []
        for character in characters:
            features = get_features(character)
            n_char.append(features)
        features_all.append(n_char)
    return features_all


def shape_labels(corpus, characters_all):

    labels_antagonist = []
    labels_protagonist = []
    labels_both = []
    books = []
    chars = []

    for characters, book in zip(characters_all, corpus):
        for character in characters:
            antagonist = 0
            protagonist = 0
            both = 0
            chars.append(character.name)
            books.append(book.title)

            ant_name_sin = [item for sublist in [[character.name] + character.sinonims for character in book.antagonists] for item in sublist]
            prot_name_sin = [item for sublist in [[character.name] + character.sinonims for character in book.protagonists] for item in sublist]

            if character.name in ant_name_sin:
                antagonist = 1
                both = 1

            if character.name in prot_name_sin:
                protagonist = 1
                both = 2

            labels_both.append(both)
            labels_protagonist.append(protagonist)
            labels_antagonist.append(antagonist)

    labels = np.array([labels_antagonist, labels_protagonist, labels_both, chars, books])

    return labels

def prune_books(corpus, characters_all, graphs, ners):
    corpus_new = []
    characters_new = []
    graphs_new = []
    ner_new = []
    for book, characters, graph, ner in zip(corpus, characters_all, graphs, ners):
        if len(characters) > 0:
            corpus_new.append(book)
            characters_new.append(characters)
            graphs_new.append(graph)
            ner_new.append(ner)

    return corpus_new, characters_new, graphs_new, ner_new

def prune_characters(corpus, characters_all, graphs, ners):
    characters_new = []
    corpus_new = []
    graphs_new = []
    ner_new = []
    for book, characters, graph, ner in zip(corpus, characters_all, graphs, ners):
        characters_with_sinonims = [[character.name] + character.sinonims for character in book.characters]
        characters_with_sinonims = [item for sublist in characters_with_sinonims for item in sublist]
        n_char = []
        for character in characters:
            if character.name in characters_with_sinonims:
                n_char.append(character)
        if len(n_char) > 0:
            corpus_new.append(book)
            characters_new.append(n_char)
            graphs_new.append(graph)
            ner_new.append(ner)
    return corpus_new, characters_new, graphs_new, ner_new

def eval_all_books(corpus, characters, p_p, pp_p, p_a, pp_a):
    evals = []
    for book in corpus:
        book_eval = evaluate_prediction_book(book, characters, p_p, pp_p, p_a, pp_a)
        evals.append(book_eval)
    return evals


def draw_network(edgelist, save=False):
    g = nx.Graph()
    g.add_weighted_edges_from(edgelist)

    pos = nx.circular_layout(g)
    nodes = g.nodes()
    edges = g.edges()
    weights = [g[u][v]['weight'] for u, v in edges]
    colors = []
    for w in weights:
        if w < 0:
            colors.append('r')
        elif w == 0:
            colors.append('b')
        else:
            colors.append('g')

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos, edgelist=edges, edge_color=colors)
    nx.draw_networkx_labels(g, pos)
    if save:
        plt.savefig('plots/network.png')
    plt.draw()
    plt.show()



def draw_network_comparison(our, gold, save=False, path=None):
    g = nx.Graph()
    g.add_weighted_edges_from(our)
    gg = nx.Graph()
    gg.add_weighted_edges_from(gold)


    nodes_g = gg.nodes()

    g = nx.induced_subgraph(g, nodes_g)

    pos_g = nx.circular_layout(gg)
    pos = pos_g
    # pos = nx.circular_layout(g)


    edges = g.edges()
    weights = [g[u][v]['weight'] for u, v in edges]
    colors = []



    edges_g = gg.edges()
    edges_g = [edge for edge in edges_g if edge[0] != edge[1]]
    weights_g = [gg[u][v]['weight'] for u, v in edges_g]
    colors_g = []
    for w in weights:
        if w < 0:
            colors.append('r')
        elif w == 0:
            colors.append('b')
        else:
            colors.append('g')

    for w in weights_g:
        if w == 'negative':
            colors_g.append('r')
        elif w == 'neutral':
            colors_g.append('b')
        else:
            colors_g.append('g')

    fig, (ax1, ax2) = plt.subplots(1, 2)
    try:
        nx.draw_networkx_nodes(g, pos, ax=ax1)
    except:
        pos = nx.circular_layout(g)
        nx.draw_networkx_nodes(g, pos, ax=ax1)
    nx.draw_networkx_edges(g, pos, edgelist=edges, edge_color=colors, ax=ax1)
    nx.draw_networkx_labels(g, pos, ax=ax1)
    nx.draw_networkx_nodes(gg, pos_g, ax=ax2)
    nx.draw_networkx_edges(gg, pos_g, edgelist=edges_g, edge_color=colors_g, ax=ax2)
    nx.draw_networkx_labels(gg, pos_g, ax=ax2)
    if save:
        plt.savefig(path)
    plt.draw()
    plt.show()



def draw_network_comparison_all(graphs, version, save=False):
    if save:
        try:
            os.mkdir('plots/' + version)
        except:
            print('Couldnt make dir')
            pass
    for g, gg, book in graphs:
        draw_network_comparison(g, gg, save=save, path='plots/'+version+'/'+book+'.png')


def flatten(t):
    return [item for sublist in t for item in sublist]


def evaluate_relationships(relationships_gold, relationships, chars):
    results = {}
    rl = []
    for r in relationships:
        if r[2] > 0:
            rl.append((r[0], r[1], 'positive'))
        elif r[2] == 0:
            rl.append((r[0], r[1], 'neutral'))
        else:
            rl.append((r[0], r[1], 'negative'))

    len_gold = len(relationships_gold)
    len_pred = 0
    correct = 0
    false = 0
    not_found = 0
    c_p = 0
    c_n = 0

    lgp = sum([1 if r[2] == 'positive' else 0 for r in relationships_gold])
    lgn = sum([1 if r[2] == 'negative' else 0 for r in relationships_gold])

    for r in rl:
        if r[0] in chars and r[1] in chars:
            len_pred += 1
            for g in relationships_gold:
                if r[0] in g and r[1] in g:
                    if r[2] == g[2]:
                        correct += 1
                        if r[2] == 'positive':
                            c_p += 1
                        elif r[2] == 'negative':
                            c_n += 1
                    else:
                        false += 1
                else:
                    not_found += 1

    if len_pred == 0 or len_gold == 0 or correct == 0:
        results['precision'] = 0
        results['recall'] = 0
        results['f1'] = 0
        results['accuracy_fonly'] = f'0/{false}'
        results['negative_relationships'] = f'0/{lgn}'
        results['positive_relationships'] = f'0/{lgp}'
        results['negative_relationships'] = f'0/{lgn}'
        results['correct'] = 0
        results['false'] = false
        results['not_found_in_gold'] = not_found
        results['num_predicted'] = len_pred
        results['num_gold'] = len_gold
    else:
        precision = correct / len_pred
        recall = correct / len_gold
        results['precision'] = precision
        results['recall'] = recall
        results['f1'] = 2 * precision * recall / (precision + recall)
        results['accuracy_fonly'] = f'{correct}/{correct+false}'
        results['positive_relationships'] = f'{c_p}/{lgp}'
        results['negative_relationships'] = f'{c_n}/{lgn}'
        results['correct'] = correct
        results['false'] = false
        results['not_found_in_gold'] = not_found
        results['num_predicted'] = len_pred
        results['num_gold'] = len_gold

    return results

def get_edgelist(graph):
    edges = []
    for edge in graph.edges():
        edges.append((edge[0], edge[1], graph.get_edge_data(edge[0], edge[1])['weight']))
    return edges

def find_char_by_id(book, id):
    for character in book.characters:
        if character.id == id:
            return character.name
    return None

def map_ner(relationships, ner_mapper):
    res = []
    for r in relationships:
        if isinstance(r, str):
            try:
                x = ner_mapper[r]
            except:
                x = r
            res.append(x)
        else:
            try:
                x = ner_mapper[r[0]]
            except:
                x = r[0]

            try:
                y = ner_mapper[r[1]]
            except:
                y = r[1]
            res.append((x, y, r[2]))
    return res

def eval_book_prediction(book):
    results = {}

    prot_gold = book.protagonists_names
    ant_gold = book.antagonists_names

    predicted_prot = []
    predicted_ant = []

    len_p_gold = len(prot_gold)
    len_p_pred = len(predicted_prot)
    p_c = 0

    for p in prot_gold:
        if p in predicted_prot:
            p_c += 1
    p_f = len_p_pred - p_c
    p_fn = len_p_gold - p_c

    len_a_gold = len(ant_gold)
    len_a_pred = len(predicted_ant)
    a_c = 0

    for a in ant_gold:
        if a in predicted_ant:
            a_c += 1
    a_f = len_a_pred - a_c
    a_fn = len_a_gold - a_c

    results['both_correct'] = p_c + a_c
    results['both_false'] = p_f + a_f
    results['both_fn'] = p_fn + a_fn

    results['protagonist_correct'] = p_c
    results['protagonist_false'] = p_f
    results['protagonist_fn'] = p_fn
    results['protagonists'] = prot_gold
    results['protagonists_predicted'] = predicted_prot

    results['antagonist_correct'] = a_c
    results['antagonist_false'] = a_f
    results['antagonist_fn'] = a_fn
    results['antagonists'] = ant_gold
    results['antagonists_predicted'] = predicted_ant

    return results

def evaluate_corpus(books):
    results = {'relationships': {}, 'prot/ant': {}}

    pr = sum([int(book['relation_eval']['positive_relationships'].split('/')[0]) for book in books])
    nr = sum([int(book['relation_eval']['negative_relationships'].split('/')[0]) for book in books])
    pr_a = sum([int(book['relation_eval']['positive_relationships'].split('/')[1]) for book in books])
    nr_a = sum([int(book['relation_eval']['negative_relationships'].split('/')[1]) for book in books])

    pred_all = sum([book['relation_eval']['num_predicted'] for book in books])
    gold_all = sum([book['relation_eval']['num_gold'] for book in books])

    correct = sum([book['relation_eval']['correct'] for book in books])
    false = sum([book['relation_eval']['false'] for book in books])

    if pred_all == 0 or gold_all == 0 or (pr + nr) == 0 or correct+false == 0:
        precision = 0
        recall = 0
        f1 = 0
        acc_fonly = 0
    else:
        precision = (pr + nr) / pred_all
        recall = (pr + nr) / gold_all
        f1 = 2 * precision * recall / (precision + recall)
        acc_fonly = correct / (correct + false)

    results['relationships']['accuracy_fonly'] = acc_fonly
    results['relationships']['positive_relationships'] = f'{pr}/{pr_a}'
    results['relationships']['negative_relationships'] = f'{nr}/{nr_a}'
    results['relationships']['accuracy'] = (pr + nr) / float(pr_a + nr_a)
    results['relationships']['precision'] = precision
    results['relationships']['recall'] = recall
    results['relationships']['f1'] = f1

    return results

if __name__ == '__main__':
    FULL = True
    path = 'pickles/svo_v1_'

    svo = True
    verb = False  # sent only, True sentiment on verb only, False sentiment on whole sentence

    MAKE_DATASET = False
    MAKE_BOOKS = True
    SHOW_PLOTS = False
    SAVE_PLOTS = False
    SAVE_CORPUS_EVAL = True

    if FULL:
        MAKE_DATASET = True
        MAKE_BOOKS = True
        SHOW_PLOTS = True
        SAVE_PLOTS = True
        SAVE_CORPUS_EVAL = True


    corpus_full = get_data("../../books/corpus.tsv", get_text=True)
    corpus_slo = [book for book in corpus_full if book.language == "slovenian"]
    corpus_eng = [book for book in corpus_full if book.language != "slovenian"]
    if MAKE_DATASET:
        make_dataset(corpus_full, path, svo, verb)

    with open(path + 'characters_all.pickle', 'rb') as f:
        data_set = pickle.load(f)

    with open(path + 'graphs.pickle', 'rb') as f:
        graphs = pickle.load(f)

    with open(path + 'ner.pickle', 'rb') as f:
        ners = pickle.load(f)

    corpus = corpus_full

    corpus, characters_x, graphs, ners = prune_books(corpus, data_set, graphs, ners)
    corpus, characters, graphs, ners = prune_characters(corpus, characters_x, graphs, ners)
    labels = shape_labels(corpus, characters).T
    features = shape_features(characters)
    features_flat = np.array([item for sublist in features for item in sublist])

    if MAKE_BOOKS:
        mapped_rels = []
        for book in corpus:
            relationships_mapped = [(find_char_by_id(book, relation[0]), find_char_by_id(book, relation[1]), relation[2]) for relation in book.relations]
            mapped_rels.append(relationships_mapped)

        cc = [[x.name for x in c] for c in characters]
        books = []
        count = 0
        i = 0
        for ff, book, graph, mr, ner in zip(features, corpus, graphs, mapped_rels, ners):
            x = {}
            x['title'] = book.title
            x['protagonists'] = book.protagonists_names
            x['antagonists'] = book.antagonists_names
            x['language'] = book.language
            temp_char = map_ner([c.name for c in book.characters], ner)
            x['relation_eval'] = evaluate_relationships(map_ner(mr, ner), map_ner(get_edgelist(nx.Graph(graph)), ner), temp_char)
            x['gold_relations'] = mr
            x['graph'] = get_edgelist(nx.Graph(graph))
            x['features'] = ff
            x['true_labels'] = labels[count:count+len(ff)]
            t = []
            for j, f in enumerate(ff):
                t.append(list(f)+list(labels[count+j]))
            x['both'] = t
            x['ner_mapper'] = ner

            count += len(ff)
            i += 1
            books.append(x)

        with open(path + 'books.pickle', 'wb') as f:
            pickle.dump(books, f)
    else:
        with open(path + 'books.pickle', 'rb') as f:
            books = pickle.load(f)

    #X_train, X_test, Y_train, Y_test = train_test_split(features_flat, labels, test_size=0.5, random_state=42) # split among all chars

    ffs = [book['features'] for book in books]
    lbs = [book['true_labels'] for book in books]

    X_train, X_test, Y_train, Y_test = train_test_split(ffs, lbs, test_size=0.5, random_state=42) # split by books

    books_test = [y[0][-1]for y in Y_test]
    books_train = [y[0][-1] for y in Y_train]

    X_train = np.array(flatten(X_train))
    X_test = np.array(flatten(X_test))
    Y_train = np.array(flatten(Y_train))
    Y_test = np.array(flatten(Y_test))

    cf_a_dt, cf_a_rf, cf_a_knn, cf_a_svc = train_classifiers(X_train, X_test, Y_train[:, 0], Y_test[:, 0])
    cf_p_dt, cf_p_rf, cf_p_knn, cf_p_svc = train_classifiers(X_train, X_test, Y_train[:, 1], Y_test[:, 1])
    cf_dt, cf_rf, cf_knn, cf_svc = train_classifiers(X_train, X_test, Y_train[:, 2], Y_test[:, 2])

    #p_a, pp_a = get_predictions(cf_a_svc, X_test)
    #p_p, pp_p = get_predictions(cf_p_svc, X_test)
    #p, pp = get_predictions(cf_svc, X_test)

    #evals_test = eval_all_books(corpus_slo, Y_test[:, 3], p_p, pp_p, p_a, pp_a)

    #print('TEST:')
    #print(evals_test)

    p_a, pp_a = get_predictions(cf_a_svc, features_flat)
    p_p, pp_p = get_predictions(cf_p_svc, features_flat)
    p, pp = get_predictions(cf_svc, features_flat)

    evals_all = eval_all_books(corpus_slo, labels[:, 3], p_p, pp_p, p_a, pp_a)

    print('ALL:')
    print(evals_all)

    temp = np.concatenate((labels[:, 2:], np.reshape(p_p, (len(p_p), 1)), np.reshape(p_a, (len(p_a), 1))), axis=1)

    if SHOW_PLOTS:
        gs = []
        for book in books:
            gs.append((book['graph'], book['gold_relations'], book['title']))
        draw_network_comparison_all(gs, path.split('/')[1], save=SAVE_PLOTS)

    eval_corpus = evaluate_corpus(books)

    if SAVE_CORPUS_EVAL:
        with open(path + 'corpus_eval.pickle', 'wb') as f:
            pickle.dump(eval_corpus, f)

    pass