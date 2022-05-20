import classla
import numpy as np
import stanza
from networkx.algorithms.triads import triads_by_type

from ECKG.src.eventify import Eventify
from ECKG.src.helper_functions import fix_ner, deduplicate_named_entities, get_relations_from_sentences, \
    create_graph_from_pairs, graph_entity_importance_evaluation, get_entities_from_svo_triplets, find_similar, \
    get_relations_from_sentences_sentiment, group_relations_filter, create_graph_from_pairs_sentiment, group_relations, \
    get_triads, evaluate_triads, get_entities_from_svo_triplets_sentiment
from books.get_data import get_data, Book
from ECKG.src.character import Character
from sentiment.sentiment_analysis import *

def precision_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result


def ranking_metric(book: Book, predictions):
    actual = []
    for character in book.characters:
        predicted = False
        for sinonim in character.sinonims:
            og_sinonim = sinonim
            sinonim = find_similar(sinonim, predictions, similarity=80)
            if sinonim is not None:
                predicted = True
                actual.append(sinonim)
                break
        if not predicted:
            actual.append(og_sinonim)
    res = 0
    print(actual)
    print(predictions)
    for k in range(1, len(actual) + 1):
        res += precision_at_k(actual, predictions, k)
    return res / len(actual)


def calculate_metrics(book: Book, predictions):
    predictions = list(map(lambda x: x.lower(), predictions))
    tp = 0
    fp = 0
    fn = 0
    actual = []

    for character in book.characters:
        predicted = False
        for sinonim in character.sinonims:
            sinonim = find_similar(sinonim, predictions, similarity=80)
            if sinonim is not None:
                predicted = True
                tp += 1
                actual.append(sinonim)
                break
        if not predicted:
            fn += 1
    for prediction in predictions:
        if prediction not in actual:
            fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = tp / (tp + ((fp + fn) / 2))

    # mAP
    return precision, recall, f1


def keep_top(names: list, values: list, cutoff=0.9):
    values = np.array(values)
    values = values[values > np.quantile(values, cutoff)].tolist()
    names = names[0:len(values)]
    return names, values


def print_metrics(metrics):
    print("Precision: ", str(metrics[0]))
    print("Recall: ", str(metrics[1]))
    print("F1: ", str(metrics[2]))


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


def evaluate_book(book: Book, pipeline, svo_extractor, sa, cutoff=0.9, verbose=False):
    # Run text through pipeline
    data = pipeline(book.text)
    # Fix NER anomalies
    data = fix_ner(data)

    # Run named entity deduplication/resolution
    p = []
    r = []
    f = []
    # for cutoff in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     print(cutoff)
    deduplication_mapper, count = deduplicate_named_entities(data, count_entities=True)

    # 1. IMPORTANCE BASED ON NUMBER OF OCCURRENCES IN THE TEXT
    count = dict(sorted(count.items(), key=lambda x: -x[1]))
    # keep top 10%
    names, values = keep_top(list(count.keys()), list(count.values()), cutoff=cutoff)
    # for i in range(len(cutoff)):
    #     print(names[i] + ": " + str(cutoff[i]))
    metrics1 = calculate_metrics(book, names)
    p.append(metrics1[0])
    r.append(metrics1[1])
    f.append(metrics1[2])
    metrics1 = (metrics1[0], metrics1[1], metrics1[2], ranking_metric(book, names))
    if verbose:
        print("Results for importance ranking based on number of occurences in the text")
        print_metrics(metrics1)
    # plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], p, label="precision")
    # plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], r, label="recall")
    # plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], f, label="F1 score")
    # plt.title('Effect of cutoff argument on precision, recall and F1 score')
    # plt.legend()
    # plt.savefig("plot.png")
    # 2. IMPORTANCE BASED ON GRAPH CENTRALITIES, WHERE GRAPH IS CONSTRUCTED FROM ENTITY CO-OCCURRENCES IN THE SAME SENTENCE
    sentence_relations = get_relations_from_sentences_sentiment(data, deduplication_mapper, sa)
    graph = create_graph_from_pairs(sentence_relations)
    names, values = graph_entity_importance_evaluation(graph)
    names, values = keep_top(names, values, cutoff=cutoff)


    characters1, relationship_graph1 = get_characters(sentence_relations, names, book.text)


    # [print((x, y)) for x, y in zip(names, values)]
    metrics2 = calculate_metrics(book, names)
    metrics2 = (metrics2[0], metrics2[1], metrics2[2], ranking_metric(book, names))

    if verbose:
        print("Results for importance ranking based on network centralities, where graph is constructed from entity "
              "co-occurrences in the same sentence")
        print_metrics(metrics2)
    # [print((x, y)) for y, x in sorted(zip(values, graph.nodes), reverse=True)]

    names_keep = names

    # 3. IMPORTANCE BASED ON GRAPH CENTRALITIES, WHERE GRAPH IS CONSTRUCTED FROM ENTITY CO-OCCURRENCES IN THE SVO TRIPLET
    entities_from_svo_triplets, svo_sentiment = get_entities_from_svo_triplets_sentiment(book, svo_extractor,
                                                                               deduplication_mapper, sa)
    svo_triplet_graph = create_graph_from_pairs(entities_from_svo_triplets)
    names, values = graph_entity_importance_evaluation(svo_triplet_graph)
    # [print((x, y)) for x, y in zip(names, values)]
    names, values = keep_top(names, values, cutoff=cutoff)
    # print(names)
    metrics3 = calculate_metrics(book, names)
    metrics3 = (metrics3[0], metrics3[1], metrics3[2], ranking_metric(book, names))

    characters2, relationship_graph2 = get_characters(svo_sentiment, names_keep, book.text)


    if verbose:
        print("Results for importance ranking based on network centralities, where graph is constructed from entity "
              "co-occurrences in the same SVO triplet")
        print_metrics(metrics3)
    # [print((x, y)) for y, x in sorted(zip(important_entities_from_svo_triplets, svo_triplet_graph.nodes), reverse=True)]
    return metrics1, metrics2, metrics3


def evaluate_all(corpus_path="../../books/corpus.tsv"):
    corpus = get_data(corpus_path, get_text=True)
    # Initialize Slovene classla pipeline
    classla.download('sl')
    sl_pipeline = classla.Pipeline("sl", processors='tokenize,ner, lemma, pos, depparse', use_gpu=True)
    sl_e = Eventify(language="sl")

    # Initialize English stanza pipeline
    stanza.download("en")
    en_pipeline = stanza.Pipeline("en", processors='tokenize,ner, lemma, pos, depparse', use_gpu=True)
    en_e = Eventify(language="en")
    for book in corpus:
        if book.language == "slovenian":
            evaluate_book(book, sl_pipeline, sl_e, cutoff=0.9)
        else:
            evaluate_book(book, en_pipeline, en_e, cutoff=0.9)


if __name__ == "__main__":
    corpus = get_data("../../books/corpus.tsv", get_text=True)

    classla.download('sl')
    sl_pipeline = classla.Pipeline("sl", processors='tokenize,ner, lemma, pos, depparse', use_gpu=True)
    sl_e = Eventify(language="sl")

    # Initialize English stanza pipeline
    stanza.download("en")
    en_pipeline = stanza.Pipeline("en", processors='tokenize,ner, lemma, pos, depparse', use_gpu=True)
    en_e = Eventify(language="en")


    # job = read_lexicon_job("../sentiment/lexicons/Slovene sentiment lexicon JOB 1.0/Slovene_sentiment_lexicon_JOB.txt")
    kss = convert_list_to_dict(read_lexicon_kss('../../sentiment/lexicons/Slovene sentiment lexicon KSS 1.1/')['Slolex'])
    # afinn_en = read_lexicon_afinn_en("../sentiment/lexicons/AFINN_en/AFINN-111.txt")
    # sentiwordnet_en = read_lexicon_sentiwordnet("../sentiment/lexicons/SentiWordNet.txt")

    # SL
    sa_kss = SentimentAnalysis(kss)
    # sa_job = SentimentAnalysis(job)

    # EN
    # sa_afinn = SentimentAnalysis(afinn_en)
    # sa_swn = SentimentAnalysis(sentiwordnet_en)


    book = corpus[9]
    evaluate_book(book, sl_pipeline, sl_e, sa_kss, cutoff=0.9, verbose=True)

    pass
