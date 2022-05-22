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
            print(f'{protagonist} not found in characters for book {book.title}')
            continue
        p = predictions_p[i]
        p_true.append(p)
        p_ranks.append(predictions_p_proba[i][1])
        p_names.append(protagonist)

    for antagonist in book.antagonists_names:
        try:
            i = list(characters).index(antagonist)
        except ValueError:
            print(f'{antagonist} not found in characters for book {book.title}')
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


def make_dataset(corpus):
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




    with open('characters_sl.pickle', 'wb') as f:
        pickle.dump(sl, f, pickle.HIGHEST_PROTOCOL)

    with open('characters_en.pickle', 'wb') as f:
        pickle.dump(en, f, pickle.HIGHEST_PROTOCOL)

    with open('characters_all.pickle', 'wb') as f:
        pickle.dump(both, f, pickle.HIGHEST_PROTOCOL)


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

def prune_books(corpus, characters_all):
    corpus_new = []
    characters_new = []

    for book, characters in zip(corpus, characters_all):
        if len(characters) > 0:
            corpus_new.append(book)
            characters_new.append(characters)

    return corpus_new, characters_new

def prune_characters(corpus, characters_all):
    characters_new = []
    corpus_new = []
    for book, characters in zip(corpus, characters_all):
        characters_with_sinonims = [[character.name] + character.sinonims for character in book.characters]
        characters_with_sinonims = [item for sublist in characters_with_sinonims for item in sublist]
        n_char = []
        for character in characters:
            if character.name in characters_with_sinonims:
                n_char.append(character)
        if len(n_char) > 0:
            corpus_new.append(book)
            characters_new.append(n_char)
    return corpus_new, characters_new

def eval_all_books(corpus, characters, p_p, pp_p, p_a, pp_a):
    evals = []
    for book in corpus:
        book_eval = evaluate_prediction_book(book, characters, p_p, pp_p, p_a, pp_a)
        evals.append(book_eval)
    return evals


if __name__ == '__main__':
    data = load_breast_cancer()
    X = data.data
    Y = data.target

    corpus_full = get_data("../../books/corpus.tsv", get_text=True)
    corpus_slo = [book for book in corpus_full if book.language == "slovenian"]
    corpus_eng = [book for book in corpus_full if book.language != "slovenian"]

    #make_dataset(corpus_full)


    with open('characters_all.pickle', 'rb') as f:
        data_set = pickle.load(f)

    corpus = corpus_slo

    corpus, characters = prune_books(corpus, data_set)
    corpus, characters = prune_characters(corpus, characters)
    labels = shape_labels(corpus, characters).T
    features = shape_features(characters)
    features_flat = np.array([item for sublist in features for item in sublist])

    X_train, X_test, Y_train, Y_test = train_test_split(features_flat, labels, test_size=0.5, random_state=42)

    cf_a_dt, cf_a_rf, cf_a_knn, cf_a_svc = train_classifiers(X_train, X_test, Y_train[:, 0], Y_test[:, 0])
    cf_p_dt, cf_p_rf, cf_p_knn, cf_p_svc = train_classifiers(X_train, X_test, Y_train[:, 1], Y_test[:, 1])
    cf_dt, cf_rf, cf_knn, cf_svc = train_classifiers(X_train, X_test, Y_train[:, 2], Y_test[:, 2])

    p_a, pp_a = get_predictions(cf_a_svc, X_test)
    p_p, pp_p = get_predictions(cf_p_svc, X_test)
    p, pp = get_predictions(cf_svc, X_test)

    evals_test = eval_all_books(corpus_slo, Y_test[:, 3], p_p, pp_p, p_a, pp_a)

    p_a, pp_a = get_predictions(cf_a_svc, features_flat)
    p_p, pp_p = get_predictions(cf_p_svc, features_flat)
    p, pp = get_predictions(cf_svc, features_flat)

    evals_all = eval_all_books(corpus_slo, labels[:, 3], p_p, pp_p, p_a, pp_a)

    #make_dataset()

    pass