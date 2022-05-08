import classla
import numpy as np
import stanza

from ECKG.src.eventify import Eventify
from ECKG.src.helper_functions import fix_ner, deduplicate_named_entities, get_relations_from_sentences, \
    create_graph_from_pairs, graph_entity_importance_evaluation, get_entities_from_svo_triplets
from books.get_data import get_data, Book


def calculate_metrics(book: Book, predictions):
    predictions = list(map(lambda x: x.lower(), predictions))
    tp = 0
    fp = 0
    fn = 0
    true = []

    for character in book.characters:
        predicted = False
        for sinonim in character.sinonims:
            sinonim = sinonim.lower()
            if sinonim in predictions:
                predicted = True
                tp += 1
                true.append(sinonim)
        if not predicted:
            fn += 1
    for prediction in predictions:
        if prediction not in true:
            fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = tp / (tp + ((fp + fn) / 2))

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

    # 1. IMPORTANCE BASED ON NUMBER OF OCCURRENCES IN THE TEXT
    count = dict(sorted(count.items(), key=lambda x: -x[1]))
    # keep top 10%
    names, values = keep_top(list(count.keys()), list(count.values()))
    # for i in range(len(cutoff)):
    #     print(names[i] + ": " + str(cutoff[i]))
    metrics1 = calculate_metrics(book, names)
    if verbose:
        print("Results for importance ranking based on number of occurences in the text")
        print_metrics(metrics1)

    # 2. IMPORTANCE BASED ON GRAPH CENTRALITIES, WHERE GRAPH IS CONSTRUCTED FROM ENTITY CO-OCCURRENCES IN THE SAME SENTENCE
    sentence_relations = get_relations_from_sentences(data, deduplication_mapper)
    graph = create_graph_from_pairs(sentence_relations)
    names, values = graph_entity_importance_evaluation(graph)
    names, values = keep_top(names, values, 0.5)

    [print((x, y)) for x, y in zip(names, values)]
    metrics2 = calculate_metrics(book, names)
    if verbose:
        print("Results for importance ranking based on network centralities, where graph is constructed from entity "
              "co-occurrences in the same sentence")
        print_metrics(metrics2)
    # [print((x, y)) for y, x in sorted(zip(values, graph.nodes), reverse=True)]

    # 3. IMPORTANCE BASED ON GRAPH CENTRALITIES, WHERE GRAPH IS CONSTRUCTED FROM ENTITY CO-OCCURRENCES IN THE SVO TRIPLET
    entities_from_svo_triplets = get_entities_from_svo_triplets(book, svo_extractor, deduplication_mapper)
    svo_triplet_graph = create_graph_from_pairs(entities_from_svo_triplets)
    names, values = graph_entity_importance_evaluation(svo_triplet_graph)
    [print((x, y)) for x, y in zip(names, values)]
    names, values = keep_top(names, values, 0.5)
    print(names)
    metrics3 = calculate_metrics(book, names)
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
    sl_pipeline = classla.Pipeline("sl", processors='tokenize,ner, lemma, pos, depparse', use_gpu=True)
    sl_e = Eventify(language="sl")
    book = corpus[9]
    evaluate_book(book, sl_pipeline, sl_e, cutoff=0.8, verbose=True)
