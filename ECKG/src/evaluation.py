import classla
import numpy as np
import stanza

from ECKG.src.eventify import Eventify
from ECKG.src.helper_functions import fix_ner, deduplicate_named_entities, get_relations_from_sentences, \
    create_graph_from_pairs, graph_entity_importance_evaluation, get_entities_from_svo_triplets, find_similar
from books.get_data import get_data, Book


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


def evaluate_book(book: Book, pipeline, svo_extractor, cutoff=0.9, verbose=False):
    # Run text through pipeline
    data = pipeline(book.text)
    # Fix NER anomalies
    data = fix_ner(data)

    # Run named entity deduplication/resolution
    deduplication_mapper, count = deduplicate_named_entities(data, count_entities=True)
    p = []
    r = []
    f = []
    # for cutoff in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     print(cutoff)

    # 1. IMPORTANCE BASED ON NUMBER OF OCCURRENCES IN THE TEXT
    count = dict(sorted(count.items(), key=lambda x: -x[1]))
    names = list(count.keys())
    values = list(count.values())
    cutoff_names, cutoff_values = keep_top(names, values, cutoff=cutoff)
    print(cutoff_names)
    # for i in range(len(cutoff)):
    #     print(names[i] + ": " + str(cutoff[i]))
    metrics11 = calculate_metrics(book, cutoff_names)
    metrics12 = calculate_metrics(book, names[0:len(book.characters)])
    # p.append(metrics1[0])
    # r.append(metrics1[1])
    # f.append(metrics1[2])
    metrics1 = (
    metrics11[0], metrics11[1], metrics11[2], metrics12[0], metrics12[1], metrics12[2], ranking_metric(book, names))
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
    sentence_relations = get_relations_from_sentences(data, deduplication_mapper)
    graph = create_graph_from_pairs(sentence_relations)
    names, values = graph_entity_importance_evaluation(graph)

    cutoff_names, cutoff_values = keep_top(names, values, cutoff=cutoff)
    print(cutoff_names)
    # values = np.array(v2, dtype=np.float32)
    # values = values[values > 0]
    # names = n2[0:len(values)]
    # [print((x, y)) for x, y in zip(names, values)]
    metrics21 = calculate_metrics(book, cutoff_names)
    metrics22 = calculate_metrics(book, names[0:len(book.characters)])
    metrics2 = (
    metrics21[0], metrics21[1], metrics21[2], metrics22[0], metrics22[1], metrics22[2], ranking_metric(book, names))

    if verbose:
        print("Results for importance ranking based on network centralities, where graph is constructed from entity "
              "co-occurrences in the same sentence")
        print_metrics(metrics2)
    # [print((x, y)) for y, x in sorted(zip(values, graph.nodes), reverse=True)]

    # 3. IMPORTANCE BASED ON GRAPH CENTRALITIES, WHERE GRAPH IS CONSTRUCTED FROM ENTITY CO-OCCURRENCES IN THE SVO TRIPLET
    entities_from_svo_triplets = get_entities_from_svo_triplets(book, svo_extractor, deduplication_mapper)
    svo_triplet_graph = create_graph_from_pairs(entities_from_svo_triplets)
    names, values = graph_entity_importance_evaluation(svo_triplet_graph)
    # [print((x, y)) for x, y in zip(names, values)]
    n3 = names
    v3 = values
    cutoff_names, cutoff_values = keep_top(names, values, cutoff=cutoff)
    print(cutoff_names)
    # values = np.array(v3, dtype=np.float32)
    # values = values[values > 0]
    # names = n3[0:len(values)]
    # print(names)
    print([[character.sinonims for character in book.characters]])
    metrics31 = calculate_metrics(book, cutoff_names)
    metrics32 = calculate_metrics(book, names[0:len(book.characters)])
    metrics3 = (
    metrics31[0], metrics31[1], metrics31[2], metrics32[0], metrics32[1], metrics32[2], ranking_metric(book, names))

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
    print(precision_at_k(["a", "b", "c"], ["b", "d", "c"], 2))

    corpus = get_data("../../books/corpus.tsv", get_text=True)
    sl_pipeline = classla.Pipeline("sl", processors='tokenize,ner, lemma, pos, depparse', use_gpu=True)
    # en_pipeline = stanza.Pipeline("en", processors='tokenize,ner, lemma, pos, depparse', use_gpu=True)
    sl_e = Eventify(language="sl")
    # en_e = Eventify(language="en")
    slo_books = []

    m1 = [[] for _ in range(7)]
    m2 = [[] for _ in range(7)]
    m3 = [[] for _ in range(7)]
    count = 0
    for book in corpus:
        if count == 5:
            break
        if book.language == "slovenian":
            try:
                res = evaluate_book(book, sl_pipeline, sl_e, cutoff=0.8, verbose=False)
                for (x, y) in zip(res, [m1, m2, m3]):
                    for i, z in enumerate(x):
                        y[i].append(z)

                count += 1
            except Exception as e:
                print(e)
        elif book.language == "english":
            continue
            try:
                res = evaluate_book(book, en_pipeline, en_e, cutoff=0.8, verbose=False)
                for (x, y) in zip(res, [m1, m2, m3]):
                    for i, z in enumerate(x):
                        y[i].append(z)
                count += 1
            except Exception as e:
                print(e)

    for i, x in enumerate([m1, m2, m3]):
        print("Method ", str(i))
        print("Precision 1: ", str(sum(x[0]) / len(x[0])))
        print("Recall 1: ", str(sum(x[1]) / len(x[1])))
        print("F1 score 1: ", str(sum(x[2]) / len(x[2])))
        print("Precision 2: ", str(sum(x[3]) / len(x[3])))
        print("Recall 2: ", str(sum(x[4]) / len(x[4])))
        print("F1 score 2: ", str(sum(x[5]) / len(x[5])))
        print("Mean average precision: ", str(sum(x[6]) / len(x[6])))
