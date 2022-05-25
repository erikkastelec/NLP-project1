import classla
import numpy as np
import stanza

from ECKG.src.eventify import Eventify
from ECKG.src.helper_functions import fix_ner, deduplicate_named_entities, get_relations_from_sentences, \
    create_graph_from_pairs, graph_entity_importance_evaluation, get_entities_from_svo_triplets, find_similar, \
    EnglishCorefPipeline, get_relations_from_sentences_coref_sentence_sent, SloveneCorefPipeline
from books.get_data import get_data, Book
import json

from sentiment.sentiment_analysis import SentimentAnalysis, read_lexicon_sentiwordnet


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
            sinonim = find_similar(sinonim, predictions, similarity=90)
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
    # for prediction in predictions:
    #     if prediction not in actual:
    #         fp += 1
    try:
        precision = tp / len(predictions)
        recall = tp / len(book.characters)
        f1 = 2 * (precision * recall / (precision + recall))
    except ZeroDivisionError:
        precision = 0
        recall = 0
        f1 = 0
    # mAP
    return precision, recall, f1


def keep_top(names: list, values: list, cutoff=0.9):
    if len(names) < 10:
        return names, values
    values = np.array(values)
    values = values[values > np.quantile(values, cutoff)].tolist()
    names = names[0:len(values)]
    return names, values


def print_metrics(metrics):
    print("Precision: ", str(metrics[0]))
    print("Recall: ", str(metrics[1]))
    print("F1: ", str(metrics[2]))


def evaluate_book(book: Book, pipeline, coref_pipeline, svo_extractor, cutoff=0.9, verbose=False):
    # Run text through pipeline
    data = pipeline(book.text)
    data = fix_ner(data)

    # Create coref_pipeline if Slo
    # if book.language == "slovenian":
    #     coref_pipeline = SloveneCorefPipeline(data)
    #
    # x = coref_pipeline.predict()
    #
    # for coreference in x["coreferences"]:
    #     print(x["mentions"][coreference["id1"]-1])
    #     print(x["mentions"][coreference["id2"]-1])
    # print(x)
    # exit()
    # Run named entity deduplication/resolution
    deduplication_mapper, count = deduplicate_named_entities(data, count_entities=True, add_missed=False, book=book)
    # data = add_missed(book, data, deduplication_mapper)
    # deduplication_mapper, count = deduplicate_named_entities(data, count_entities=True)
    if coref_pipeline:
        # Coreference resolution (new_chains = (word: scapy object, mentions)
        coref_pipeline.process_text(book.text)
        coref_chains, new_chains = coref_pipeline.unify_naming_coreferee(deduplication_mapper)

        # Set count of named entities to number of coreference mentions
        for key, value in coref_chains.items():
            count[key] = len(value)
        # Add entities from new_chains to named ent1ity count dict
        for key, value in new_chains.items():
            count[value[0].text] = len(value[1])

    count = dict(sorted(count.items(), key=lambda x: -x[1]))
    p = []
    r = []
    f = []
    # for cutoff in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     print(cutoff)

    # 1. IMPORTANCE BASED ON NUMBER OF OCCURRENCES IN THE TEXT
    names = list(count.keys())
    values = list(count.values())
    cutoff_names = names
    cutoff_values = values
    cutoff_names, cutoff_values = keep_top(names, values, cutoff=cutoff)
    # print(cutoff_names)
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
    try:
        sentence_relations = get_relations_from_sentences(data, deduplication_mapper, coref_pipeline=coref_pipeline)
        graph = create_graph_from_pairs(sentence_relations)
        names, values = graph_entity_importance_evaluation(graph)
        cutoff_names = names
        cutoff_values = values
        cutoff_names, cutoff_values = keep_top(names, values, cutoff=cutoff)
        # print(cutoff_names)
        # values = np.array(v2, dtype=np.float32)
        # values = values[values > 0]
        # names = n2[0:len(values)]
        # [print((x, y)) for x, y in zip(names, values)]
        metrics21 = calculate_metrics(book, cutoff_names)
        metrics22 = calculate_metrics(book, names[0:len(book.characters)])
        metrics2 = (
            metrics21[0], metrics21[1], metrics21[2], metrics22[0], metrics22[1], metrics22[2],
            ranking_metric(book, names))

        if verbose:
            print(
                "Results for importance ranking based on network centralities, where graph is constructed from entity "
                "co-occurrences in the same sentence")
            print_metrics(metrics2)
    except ValueError:
        metrics2 = (0, 0, 0, 0, 0, 0, 0)
    # [print((x, y)) for y, x in sorted(zip(values, graph.nodes), reverse=True)]

    # 3. IMPORTANCE BASED ON GRAPH CENTRALITIES, WHERE GRAPH IS CONSTRUCTED FROM ENTITY CO-OCCURRENCES IN THE SVO TRIPLET
    try:
        entities_from_svo_triplets = get_entities_from_svo_triplets(book, svo_extractor, deduplication_mapper, doc=data,
                                                                    coref_pipeline=coref_pipeline)
        svo_triplet_graph = create_graph_from_pairs(entities_from_svo_triplets)

        names, values = graph_entity_importance_evaluation(svo_triplet_graph)
        # [print((x, y)) for x, y in zip(names, values)]
        n3 = names
        v3 = values
        cutoff_names = names
        cutoff_values = values
        cutoff_names, cutoff_values = keep_top(names, values, cutoff=cutoff)
        # print(cutoff_names)
        # values = np.array(v3, dtype=np.float32)
        # values = values[values > 0]
        # names = n3[0:len(values)]
        # print(names)
        # print([[character.sinonims for character in book.characters]])
        metrics31 = calculate_metrics(book, cutoff_names)
        metrics32 = calculate_metrics(book, names[0:len(book.characters)])
        metrics3 = (
            metrics31[0], metrics31[1], metrics31[2], metrics32[0], metrics32[1], metrics32[2],
            ranking_metric(book, names))
    except ValueError:
        metrics3 = (0, 0, 0, 0, 0, 0, 0)
    if verbose:
        print("Results for importance ranking based on network centralities, where graph is constructed from entity "
              "co-occurrences in the same SVO triplet")
        print_metrics(metrics3)
    # [print((x, y)) for y, x in sorted(zip(important_entities_from_svo_triplets, svo_triplet_graph.nodes), reverse=True)]
    return metrics1, metrics2, metrics3


def print_metric_final(m1, m2, m3, titles):
    method_titles = ["EOIS", "ECS", "ECSVO"]
    for i in range(0, len(titles)):
        print("Title: ", titles[i])
        for j, x in enumerate([m1, m2, m3]):
            precision = x[3][i]
            recall = x[4][i]
            f_score = x[5][i]
            mAP = x[6][i]
            print("Method: ", method_titles[j])
            print("Precision: ", str(precision))
            print("Recall: ", str(recall))
            print("F1 score: ", str(f_score))
            print("Mean average precision: ", str(mAP))
            print("*************************")
    print("AVERAGE")
    for i, x in enumerate([m1, m2, m3]):
        print("Method ", str(i))
        print("Precision: ", str(sum(x[3]) / len(x[3])))
        print("Recall: ", str(sum(x[4]) / len(x[4])))
        print("F1 score: ", str(sum(x[5]) / len(x[5])))
        print("Mean average precision: ", str(sum(x[6]) / len(x[6])))


def evaluate_all(corpus_path="../../books/corpus.tsv"):
    corpus = get_data(corpus_path, get_text=True)
    # Initialize Slovene classla pipeline

    m1_slo = [[] for _ in range(7)]
    m2_slo = [[] for _ in range(7)]
    m3_slo = [[] for _ in range(7)]
    title_slo = []
    m1_en = [[] for _ in range(7)]
    m2_en = [[] for _ in range(7)]
    m3_en = [[] for _ in range(7)]
    title_en = []
    m1_comb = [[] for _ in range(7)]
    m2_comb = [[] for _ in range(7)]
    m3_comb = [[] for _ in range(7)]
    title_comb = []
    classla.download('sl')
    sl_pipeline = classla.Pipeline("sl", processors='tokenize,ner, lemma, pos, depparse', use_gpu=True)
    sl_e = Eventify(language="sl")
    # Initialize English stanza pipeline
    stanza.download("en")
    en_pipeline = stanza.Pipeline("en", processors='tokenize,ner, lemma, pos, mwt, depparse', use_gpu=True)
    en_e = Eventify(language="en")
    en_coref_pipeline = EnglishCorefPipeline()
    count = 0
    disregard_list = ['The Lord Of The Rings: The Fellowship of the Ring', 'The Great Gatsby',]
    for book in corpus:
        # if count == 3:
        #     continue
        count += 1
        if book.title in disregard_list:
            continue
        if book.language == "slovenian":
            try:
                res = evaluate_book(book, sl_pipeline, None, sl_e, cutoff=0.8)
                for (x, y) in zip(res, [m1_slo, m2_slo, m3_slo]):
                    for i, z in enumerate(x):
                        y[i].append(z)
                for (x, y) in zip(res, [m1_comb, m2_comb, m3_comb]):
                    for i, z in enumerate(x):
                        y[i].append(z)
                title_slo.append(book.title)
                title_comb.append(book.title)
                print("Analized book: ", str(count), " with title: ", book.title)
            except Exception:
                print("Failed to analyze: ", book.title)
        else:
            try:
                res = evaluate_book(book, en_pipeline, en_coref_pipeline, en_e, cutoff=0.8)
                for (x, y) in zip(res, [m1_en, m2_en, m3_en]):
                    for i, z in enumerate(x):
                        y[i].append(z)
                for (x, y) in zip(res, [m1_comb, m2_comb, m3_comb]):
                    for i, z in enumerate(x):
                        y[i].append(z)
                title_en.append(book.title)
                title_comb.append(book.title)
                print("Analized book: ", str(count), " with title: ", book.title)
            except Exception:
                print("Failed to analyze: ", book.title)

    if len(m1_slo[0]) > 0:
        print("SLO books statistics: ")
        print_metric_final(m1_slo, m2_slo, m3_slo, title_slo)

    if len(m1_en[0]) > 0:
        print("ENG books statistics: ")
        print_metric_final(m1_en, m2_en, m3_en, title_en)
    print("COMBINED statistics: ")

    print_metric_final(m1_comb, m2_comb, m3_comb, title_comb)


if __name__ == "__main__":
    evaluate_all()
    # corpus = get_data("../../books/corpus.tsv", get_text=True)
    # sl_pipeline = classla.Pipeline("sl", processors='tokenize,ner, lemma, pos, depparse', use_gpu=True)
    # sl_e = Eventify(language="sl")
    # # en_pipeline = stanza.Pipeline("en", processors='tokenize,ner, lemma, pos,mwt,  depparse', use_gpu=True)
    # # en_coref_pipeline = EnglishCorefPipeline()
    # # en_e = Eventify(language="en")
    # slo_books = []
    #
    # m1 = [[] for _ in range(7)]
    # m2 = [[] for _ in range(7)]
    # m3 = [[] for _ in range(7)]
    # count = 0
    # for book in corpus:
    #     if count == 5:
    #         break
    #     if book.language == "slovenian":
    #         # continue
    #         # try:
    #         if book.title != "Deseti brat":
    #             continue
    #         res = evaluate_book(book, sl_pipeline, None, sl_e, cutoff=0.8, verbose=False)
    #         for (x, y) in zip(res, [m1, m2, m3]):
    #             for i, z in enumerate(x):
    #                 y[i].append(z)
    #
    #         count += 1
    #         # except Exception as e:
    #         #     print(e)
    #     elif book.language == "english":
    #         continue
    #         # try:
    #         if book.title == "Little Red Cap":
    #             res = evaluate_book(book, en_pipeline, en_coref_pipeline, en_e, cutoff=0.8, verbose=False)
    #             for (x, y) in zip(res, [m1, m2, m3]):
    #                 for i, z in enumerate(x):
    #                     y[i].append(z)
    #             count += 1
    #         # except Exception as e:
    #         #     print(e)
    #
    # for i, x in enumerate([m1, m2, m3]):
    #     print("Method ", str(i))
    #     print("Precision 1: ", str(sum(x[0]) / len(x[0])))
    #     print("Recall 1: ", str(sum(x[1]) / len(x[1])))
    #     print("F1 score 1: ", str(sum(x[2]) / len(x[2])))
    #     print("Precision 2: ", str(sum(x[3]) / len(x[3])))
    #     print("Recall 2: ", str(sum(x[4]) / len(x[4])))
    #     print("F1 score 2: ", str(sum(x[5]) / len(x[5])))
    #     print("Mean average precision: ", str(sum(x[6]) / len(x[6])))
