import classla
import numpy as np
import stanza
from networkx.algorithms.triads import triads_by_type
import pickle

from ECKG.src.eventify import Eventify
from ECKG.src.helper_functions import fix_ner, deduplicate_named_entities, get_relations_from_sentences, \
    create_graph_from_pairs, graph_entity_importance_evaluation, get_entities_from_svo_triplets, find_similar, \
    get_relations_from_sentences_sentiment, group_relations_filter, create_graph_from_pairs_sentiment, group_relations, \
    get_triads, evaluate_triads, get_entities_from_svo_triplets_sentiment
from books.get_data import get_data, Book
from ECKG.src.character import Character
from sentiment.sentiment_analysis import *


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_breast_cancer


def train_classifiers(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

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

    return


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

def evaluate_book(book: Book, pipeline, sa):
    # Run text through pipeline
    data = pipeline(book.text)
    # Fix NER anomalies
    data = fix_ner(data)

    deduplication_mapper, count = deduplicate_named_entities(data, count_entities=True)

    sentence_relations = get_relations_from_sentences_sentiment(data, deduplication_mapper, sa)
    graph = create_graph_from_pairs(sentence_relations)

    if len(graph.nodes) == 0:
        return [], []

    names, values = graph_entity_importance_evaluation(graph)
    characters, relationship_graph = get_characters(sentence_relations, names, book.text)

    return characters, relationship_graph


def evaluate_book_svo(book: Book, pipeline, svo_extractor, sa):
    # Run text through pipeline
    data = pipeline(book.text)
    # Fix NER anomalies
    data = fix_ner(data)

    deduplication_mapper, count = deduplicate_named_entities(data, count_entities=True)

    entities_from_svo_triplets, svo_sentiment = get_entities_from_svo_triplets_sentiment(book, svo_extractor,
                                                                               deduplication_mapper, sa)
    svo_triplet_graph = create_graph_from_pairs(entities_from_svo_triplets)
    names, values = graph_entity_importance_evaluation(svo_triplet_graph)
    characters, relationship_graph = get_characters(svo_sentiment, names, book.text)

    return characters, relationship_graph


def make_dataset(corpus_path="../../books/corpus.tsv"):
    corpus = get_data(corpus_path, get_text=True)
    # Initialize Slovene classla pipeline
    classla.download('sl')
    sl_pipeline = classla.Pipeline("sl", processors='tokenize,ner, lemma, pos, depparse', use_gpu=True)
    sl_e = Eventify(language="sl")

    # Initialize English stanza pipeline
    stanza.download("en")
    en_pipeline = stanza.Pipeline("en", processors='tokenize,ner, lemma, pos, depparse', use_gpu=True)
    en_e = Eventify(language="en")

    job = read_lexicon_job("../../sentiment/lexicons/Slovene sentiment lexicon JOB 1.0/Slovene_sentiment_lexicon_JOB.txt")
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

    for book in corpus:
        if book.language == "slovenian":
            characters, graph = evaluate_book(book, sl_pipeline, sa_kss)
            sl.append(characters)
            both.append(characters)
        else:
            characters, graph = evaluate_book(book, en_pipeline, sa_swn)
            en.append(characters)
            both.append(characters)




    #with open('characters_all', 'wb') as f:
    #    pickle.dump(both, f, pickle.HIGHEST_PROTOCOL)


    return sl, en, both


if __name__ == '__main__':
    data = load_breast_cancer()
    X = data.data
    Y = data.target

    #with open('data/characters.pickle', 'rb') as f:
    #    data_set = pickle.load(f)

    make_dataset()

    pass