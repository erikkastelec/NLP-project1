import errno
import os
import pickle
from collections import Counter
from itertools import combinations

import classla
import networkx as nx
from classla import Document
from fuzzywuzzy import fuzz
from fuzzywuzzy.process import extract
from simstring.database.dict import DictDatabase
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher

from ECKG.src.eventify import Eventify
from books.get_data import get_data
from sentiment.sentiment_analysis import SentimentAnalysis


def map_text(text, mapper):
    distinct_entities = set()

    for key, value in mapper.items():
        text = text.replace(key, value)
        distinct_entities.add(value)

    return text, distinct_entities


def graph_entity_importance_evaluation(G: nx.Graph, centrality_weights=None):
    if centrality_weights is None:
        centrality_weights = [0.33, 0.33, 0.33]
        centrality_weights = [1]
    eigenvector_centrality = list(nx.eigenvector_centrality(G, weight="weight").values())
    degree_centrality = list(nx.degree_centrality(G).values())
    closeness_centrality = list(nx.closeness_centrality(G, distance="weight").values())
    betweenness_centrality = list(nx.betweenness_centrality(G, weight="weight").values())
    centralities = [eigenvector_centrality, degree_centrality, closeness_centrality]
    centralities = [betweenness_centrality]

    assert len(centralities) == len(centrality_weights)
    res = [0 for _ in range(len(centralities[0]))]
    for centrality, weight in zip(centralities, centrality_weights):
        for i in range(len(centrality)):
            res[i] += centrality[i] * weight
    res = zip(*sorted(zip(list(G.nodes), res), key=lambda x: -x[1]))
    return res


def is_similar_string(a, b, similarity=70):
    return fuzz.token_set_ratio(a, b) > similarity


def find_similar(a, l, similarity=70):
    res = extract(a, l, limit=1, scorer=fuzz.token_set_ratio)
    if res[0][1] > 70:
        return res[0][0]

    return None


def create_graph_from_pairs(pairs):
    MG = nx.MultiGraph()
    MG.add_weighted_edges_from([(x, y, 1) for x, _, y in pairs])
    # MG.degree(weight='weight')

    # Convert MultiGraph to Graph
    GG = nx.Graph()
    for n, nbrs in MG.adjacency():
        for nbr, edict in nbrs.items():
            value = len(edict.values())
            GG.add_edge(n, nbr, weight=1 / value)

    return GG

def create_graph_from_pairs_sentiment(pairs):
    MG = nx.MultiGraph()
    MG.add_weighted_edges_from([(x, y, s) for x, _, y, s in pairs])
    # MG.degree(weight='weight')

    # Convert MultiGraph to Graph
    GG = nx.Graph()
    for n, nbrs in MG.adjacency():
        for nbr, edict in nbrs.items():
            value = len(edict.values())
            GG.add_edge(n, nbr, weight=value)

    return GG



def get_relations_from_sentences(data: Document, ner_mapper: dict):
    """
    Find pairs of entities, which co-occur in the same sentence.
    Returns:
        list of entity verb entity pairs
        TODO: verb extraction -> None for now
    """
    pairs = []
    for i, sentence in enumerate(data.sentences):
        if len(sentence.entities) > 1:
            curr_words = []
            for j, entity in enumerate(sentence.entities):
                curr_words.append(" ".join(x.text for x in entity.words))
            for x, y in combinations(curr_words, 2):
                try:
                    pairs.append((ner_mapper[x], None, ner_mapper[y]))
                except KeyError:
                    print("WARNING")
    return pairs


def get_relations_from_sentences_sentiment(data: Document, ner_mapper: dict, sa: SentimentAnalysis):
    """
    Find pairs of entities, which co-occur in the same sentence and attach sentiment.
    Returns:
        list of entity verb entity sentiment pairs
        TODO: verb extraction -> None for now
    """
    pairs = []
    for i, sentence in enumerate(data.sentences):
        if len(sentence.entities) > 1:
            curr_words = []
            for j, entity in enumerate(sentence.entities):
                curr_words.append(" ".join(x.text for x in entity.words))
            for x, y in combinations(curr_words, 2):
                try:
                    ner_x = ner_mapper[x]
                    ner_y = ner_mapper[y]
                    word_list = sentence.strip().split(' ')
                    mask_list = []
                    try:
                        index_x = word_list.index(x)
                        index_y = word_list.index(y)
                        mask_list.append(index_x)
                        mask_list.append(index_y)
                    except ValueError:
                        print("word not found")
                    sentiment = sa.get_sentiment_sentence(word_list, mask_list)
                    pairs.append((ner_x, None, ner_y, sentiment))

                except KeyError:
                    print("WARNING")
    return pairs


def fix_ner(data):
    """
    Fixes common problems with NER (detecting only ADJ and not noun after it)
    """
    data.entities = []
    for i, sentence in enumerate(data.sentences):
        if len(sentence.entities) != 0:
            for j, entity in enumerate(sentence.entities):
                if len(entity.words) == 1:
                    if entity.words[0].upos == "ADJ" and data.sentences[i].words[
                        entity.words[0].head - 1].upos == "NOUN":
                        data.sentences[i].entities[j].tokens.append(
                            data.sentences[i].words[entity.words[0].head - 1].parent)
                        data.sentences[i].entities[j].words.append(data.sentences[i].words[entity.words[0].head - 1])
                data.entities.append(data.sentences[i].entities[j])
    return data


def deduplicate_named_entities(data, map=True, count_entities=True):
    """
    Cleans data by removing duplicates from extracted named entities
    Args:
        data: output of Classla pipeline (Classla Document)
    Returns:
        eg:
            input: "Slovenija", "Sloveniji", "Slovenije", "Slovenija"
            if map:
                # dictionary map to unified name
                {
                "Sloveniji": "Slovenija",
                "Sloveniji": "Slovenija",
                "Slovenija": "Slovenija",
                }
            else:
                ["Slovenija"]
    """
    db_helper = {}
    # Put in list if single Classla Document is provided
    entities = {}
    entities_alias = {}
    db = DictDatabase(CharacterNgramFeatureExtractor(2))
    searcher = Searcher(db, CosineMeasure())
    try:
        data = data.entities
    except AttributeError:
        data = data
    for j, entity in enumerate(data):
        added = False
        # combine tokens (eg: "Novo", "mesto" -> "Novo mesto")

        try:
            name = " ".join([x.text for x in entity.tokens])
        except Exception:
            name = entity

        # Try finding it in aliases
        try:
            e = entities_alias[name]
            try:
                entities[e].append(name)
            except KeyError:
                entities[e] = [name]
            added = True
        except KeyError:
            # Reduce complexity by performing fuzzy matching if searcher score is at least 0.6

            results = searcher.ranked_search(name.lower(), 0.6)
            for i, (_, e_name) in enumerate(results):
                if fuzz.token_set_ratio(name, e_name) > 70:
                    added = True
                    # print(e_name)
                    e_name = db_helper[e_name]
                    # if key != entities[e][e][0][0][1]:
                    #
                    #     del ner[key][i]
                    try:
                        entities[e_name].append(name)
                        entities_alias[name] = e_name
                    except KeyError:
                        entities[e_name] = [name]
                        entities_alias[name] = e_name
                    break

            if not added:
                entities[name] = [name]
                entities_alias[name] = name
                db.add(name.lower())
                db_helper[name.lower()] = name
    if map:
        res = {}
    else:
        res = []
    count_dict = {}
    for key, values in entities.items():
        most_frequent = most_frequent_item(values)
        if count_entities:
            count_dict[most_frequent] = len(values)
        if map:
            for value in values:
                res[value] = most_frequent
        else:
            map.append(most_frequent)

    if count_entities:
        res = (res, count_dict)

    return res


def most_frequent_item(l):
    counter = 0
    num = l[0]
    for i in l:
        curr_frequency = l.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num


def get_entities_from_svo_triplets(book, e: Eventify, deduplication_mapper):
    dedup_keys = deduplication_mapper.keys()
    events = e.eventify(book.text)
    NER_containing_events = []
    event_entity_count = {}
    event_entities = []
    event_tmp = []
    for s, v, o in events:
        if isinstance(s, tuple) or isinstance(o, tuple):
            continue
        s_sim = find_similar(s, dedup_keys)
        o_sim = find_similar(o, dedup_keys)
        if s_sim is not None and o_sim is not None:
            s = deduplication_mapper[s_sim]
            o = deduplication_mapper[o_sim]
            NER_containing_events.append((s, v, o))

        elif s != '<|EMPTY|>' and o != '<|EMPTY|>' and len(s.split()) < 3 and len(o.split()) < 3 and (
                s_sim is not None or o_sim is not None):
            event_entities.append(o)
            event_entities.append(s)
            try:
                event_entity_count[s] += 1
            except KeyError:
                event_entity_count[s] = 1
            try:
                event_entity_count[o] += 1
            except KeyError:
                event_entity_count[o] = 1
            # event_tmp.append((s, v, o))

    deduplication_mapper, count = deduplicate_named_entities(event_entities, count_entities=True)
    for s, v, o in event_tmp:
        NER_containing_events.append((deduplication_mapper[s], v, deduplication_mapper[o]))

    return NER_containing_events

def remove_from_list(list, unwanted_indexes):
    """
    Removes unwanted elements from list
    Args:
        list: Original list
        unwanted_indexes: List of indexes to remove

    Returns:
        list: List with removed element
    """
    for el in sorted(unwanted_indexes, reverse=True):
        del list[el]
    return list


def num_files_in_dir(path):
    """
    Counts files in a directory recursively

        Parameters:
            path (string): path to directory

        Returns:
            num_files (int): number of files in the directory
    """
    num_files = 0

    for base, dirs, files in os.walk(path):
        num_files += len(files)
    return num_files


def fuzzywuzzy_custom_dedupe(contains_dupes, threshold=70, scorer=fuzz.token_set_ratio):
    extractor = []

    # iterate over items in *contains_dupes*
    for item in contains_dupes:
        # return all duplicate matches found
        matches = extract(item, contains_dupes, limit=None, scorer=scorer)
        # filter matches based on the threshold
        filtered = [x for x in matches if x[1] > threshold]
        # if there is only 1 item in *filtered*, no duplicates were found so append to *extracted*
        if len(filtered) == 1:
            extractor.append((filtered[0][0], 1))

        else:
            # alpha sort
            filtered = sorted(filtered, key=lambda x: x[0])
            # number of occurrences sort
            l_sorted = Counter(filtered).most_common()
            filter_sort = []
            if l_sorted[0][1] != l_sorted[-1][1]:
                tmp = []
                first_in = False
                for t in l_sorted:
                    if not first_in:
                        if " " in t[0][0]:
                            first_in = True
                            filter_sort = [t[0]] + tmp
                        else:
                            tmp.append(t[0])
                    else:
                        filter_sort.append(t[0])
                if len(filter_sort) == 0:
                    filter_sort = tmp
            else:
                # length sort
                filter_sort = sorted(filtered, key=lambda x: len(x[0]), reverse=True)
            # take first item as our 'canonical example'
            extractor.append((filter_sort[0][0], len(filtered)))

    # uniquify *extractor* list
    keys = {}
    for e, i in extractor:
        keys[e] = i
    extractor = list(keys.keys())
    for i, a in enumerate(extractor):
        extractor[i] = (a, keys[a])

    return extractor


def fuzzywuzzy_custom_bestmatch(contains_dupes, threshold=70, scorer=fuzz.token_set_ratio):
    extractor = []

    # iterate over items in *contains_dupes*
    for item in contains_dupes:
        # return all duplicate matches found
        matches = extract(item, contains_dupes, limit=None, scorer=scorer)
        # filter matches based on the threshold
        filtered = [x for x in matches if x[1] > threshold]
        # if there is only 1 item in *filtered*, no duplicates were found so append to *extracted*
        if len(filtered) == 1:
            extractor.append((filtered[0][0], 1))

        else:
            # alpha sort
            filtered = sorted(filtered, key=lambda x: x[0])
            # number of occurrences sort
            l_sorted = Counter(filtered).most_common()
            filter_sort = []
            if l_sorted[0][1] != l_sorted[-1][1]:
                tmp = []
                first_in = False
                for t in l_sorted:
                    if not first_in:
                        if " " in t[0]:
                            first_in = True
                            filter_sort.append(t[0])
                            filter_sort = filter_sort + tmp
                        else:
                            tmp.append(t[0])
                    else:
                        filter_sort.append(t[0])
            else:
                # length sort
                filter_sort = sorted(filtered, key=lambda x: len(x[0]), reverse=True)
            # take first item as our 'canonical example'
            extractor.append((filter_sort[0][0], len(filtered)))

    # uniquify *extractor* list
    keys = {}
    for e, i in extractor:
        keys[e] = i
    extractor = list(keys.keys())
    for i, a in enumerate(extractor):
        extractor[i] = (a, keys[a])

    return extractor


def write_pickle(data, path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def slo_month_to_num(month):
    d = {
        "januar": 1,
        "februar": 2,
        "marec": 3,
        "april": 4,
        "maj": 5,
        "junij": 6,
        "julij": 7,
        "avgust": 8,
        "september": 9,
        "oktober": 10,
        "november": 11,
        "december": 12
    }
    try:
        res = d[month]
    except KeyError:
        return -1
    return res


if __name__ == "__main__":
    corpus = get_data("../../books/corpus.tsv", get_text=True)
    # Initialize Slovene classla pipeline
    classla.download('sl')
    sl_pipeline = classla.Pipeline("sl", processors='tokenize,ner, lemma, pos, depparse', use_gpu=True)
    # Analyze using classla/stanza to get Document format
    sl_text = corpus[6].text
    data = sl_pipeline(sl_text)
    print("extracted named entities".upper())
    data = fix_ner(data)

    print([" ".join([y.text for y in x.tokens]) for x in data.entities])

    # Run named entity deduplication/resolution
    deduplication_mapper, count = deduplicate_named_entities(data, count_entities=True)

    print("deduplication mapper".upper())
    print(deduplication_mapper)
    dedup_keys = deduplication_mapper.keys()

    # 1. IMPORTANCE BASED ON NUMBER OF OCCURRENCES IN THE TEXT
    print("1. IMPORTANCE BASED ON NUMBER OF OCCURRENCES IN THE TEXT")
    count = dict(sorted(count.items(), key=lambda x: -x[1]))
    t = 0
    for key, value in count.items():
        print(key + ": " + str(value))
    # # print("hello")
    # # Initialize CLASSLA pipeline
    # # classla.download('sl')
    # # nlp = classla.Pipeline("sl", processors='tokenize,ner')
    # # ner_res = ner_extraction("Strokovna svetovalna skupina za covid-19 je že minuli teden predlagala podaljšanje epidemije, pogoji za to pa so še vedno izpolnjeni, je izpostavila vodja skupine in infektologinja Mateja Logar. Se pa, če bodo tudi v sredo epidemiološki podatki ugodni, po njenih navedbah sproščajo pogoji za rumeno fazo in s tem povezano sproščanje ukrepov.\
    # #                Strokovna svetovalna skupina je že minuli teden predlagala podaljšanje epidemije, pogoji za to pa so še vedno izpolnjeni, je izpostavila vodja skupine in infektologinja Mateja. Se pa, če bodo tudi v sredo epidemiološki podatki ugodni, po njenih navedbah sproščajo pogoji za rumeno fazo in s tem povezano sproščanje ukrepov.\
    # #                Strokovna svetovalna skupina je. NSI je stranka, ki je v minulem letu. Vodstvo LMS-ja je lani opravilo.", nlp)
    # # for key, value in ner_res.items():
    # #     print("Values for ", key, ": ", value)
    # article = {
    #     "name": "mursic-govoriti-da-so-ljudje-krivi-ker-so-poredni-je-omalovazujoce-549022.txt",
    #     "category": "slovenija"
    # }
    # # print(get_link_from_article("RTVSLO", article))
    # # print(fuzzywuzzy_custom_dedupe(["Joze Anton", "Joze Antonov", "Miha Novak", "Novak Miha"]))
    # v = "Minister Gantar"
    # v = v[9:]
    # print(v)
