import errno
import os
import pickle
from collections import Counter
from collections import defaultdict
from itertools import combinations
from statistics import mean

import classla
import networkx as nx
import numpy as np
import spacy
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


def list_to_string(l):
    s = ""
    try:
        s = l[0]
    except IndexError:
        return ""
    for x in range(1, len(l)):
        try:
            s += " " + l[x]
        except TypeError:
            print("hello")
    return s


class English_coref_pipeline:
    def __init__(self):
        nlp = spacy.load("en_core_web_trf")
        nlp.add_pipe('coreferee')
        self.pipeline = nlp
        self.rules_analyzer = nlp.get_pipe('coreferee').annotator.rules_analyzer

    def unify_naming_coreferee(self, doc, deduplication_mapper):
        dedup_keys = deduplication_mapper.keys()
        disregard_list = ["man", "woman", "name", ]
        chains = doc._.coref_chains
        named_entity_chains = {}
        new_chains = {}
        # (spacy word, count)
        new_chains_words = {}

        for chain in chains:
            if len(chain.mentions[chain.most_specific_mention_index]) == 1:
                word = doc[chain.mentions[chain.most_specific_mention_index].root_index]
                name = list_to_string([x.text for x in self.rules_analyzer.get_propn_subtree(word)])
                if isinstance(name, str) and name == "":
                    name = list_to_string(
                        [doc[x].text for x in chain.mentions[chain.most_specific_mention_index].token_indexes])

                similar = find_similar(name, dedup_keys, similarity=80)
                if similar:
                    try:
                        named_entity_chains[similar] = named_entity_chains[similar] + flatten_list(chain.mentions)
                    except KeyError:
                        named_entity_chains[similar] = flatten_list(chain.mentions)
                else:

                    if word.tag_ == "NN" and word.text not in disregard_list:
                        try:
                            new_chains_words[name] = (new_chains_words[name][0], new_chains_words[name][1] + 1)
                            new_chains[name] = new_chains[name] + flatten_list(chain.mentions)
                        except KeyError:
                            print(name)
                            new_chains_words[name] = (word, 1)
                            new_chains[name] = flatten_list(chain.mentions)

        # sort dict
        new_chains_words = dict(sorted(new_chains_words.items(), key=lambda x: -x[1][1]))
        # Keep only top 10% new ones
        new_chains_words = keep_top_dict(new_chains_words)
        for key, value in new_chains_words.items():
            new_chains_words[key] = (value[0], new_chains[key])
        return named_entity_chains, new_chains_words


def keep_top_dict(d, cutoff=0.9):
    names = list(d.keys())
    values_og = list(d.values())
    values = [x[1] for x in values_og]
    values = np.array(values)
    values = values[values > np.quantile(values, cutoff)].tolist()
    names = names[0:len(values)]
    values = values_og[0:len(values)]
    return {k: v for k, v in zip(names, values)}


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
    # eigenvector_centrality = list(nx.eigenvector_centrality(G, weight="weight").values())
    eigenvector_centrality = []
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
    try:
        if res[0][1] > similarity:
            return res[0][0]
    except IndexError:
        pass
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
    MG.add_weighted_edges_from([(x, y, s) for (x, y), s in pairs])
    return MG


def get_triads(g):
    triads = []
    for nodes in combinations(g.nodes, 3):
        sub = g.subgraph(nodes)
        n_edges = sub.number_of_edges()
        if n_edges != 3:
            continue
        triads.append(sub)

    return triads


def evaluate_triads(triads):
    triad_classes = {'t0': [], 't1': [], 't2': [], 't3': []}
    t0s = defaultdict(int)
    t1s = defaultdict(int)
    t2s = defaultdict(int)
    t3s = defaultdict(int)
    for triad in triads:
        pos = 0
        neg = 0
        p = defaultdict(int)
        n = defaultdict(int)

        for z in list(triad.edges):
            x = z[0]
            y = z[1]

            try:
                weight = triad[x][y][0]['weight']
            except KeyError:
                weight = triad.adj[x][y]['weight']

            if weight > 0:
                pos += 1
                p[x] += 1
                p[y] += 1
            else:
                neg += 1
                n[x] += 1
                n[y] += 1

        if pos == 3:
            triad_classes['t3'].append(triad)
            for n in triad.nodes():
                t3s[n] += 1
        elif neg == 3:
            triad_classes['t0'].append(triad)
            for n in triad.nodes():
                t0s[n] += 1
        elif pos == 2 and neg == 1:
            triad_classes['t2'].append(triad)
            x = list(p.keys())[list(p.values()).index(2)]
            t2s[x] += 1
        elif pos == 1 and neg == 2:
            triad_classes['t1'].append(triad)
            x = list(n.keys())[list(n.values()).index(2)]
            t1s[x] += 1
        else:
            print('Unknown triad')

    return triad_classes, (t0s, t1s, t2s, t3s)


def entity_text_string(a):
    return " ".join(x.text for x in a.words)


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
            verbs = []
            for j, entity in enumerate(sentence.entities):
                curr_words.append(entity)
            for w in sentence.words:
                if w.upos == "VERB":
                    verbs.append((w.text, w.id))
            for x, y in combinations(curr_words, 2):
                try:
                    appended = False
                    # for verb in verbs:
                    #     if min(x[1], y[1]) < verb[1] < max(x[1], y[1]):
                    #         appended = True
                    #         pairs.append((ner_mapper[x[0]], verb[0], ner_mapper[y[0]]))
                    #         break
                    x_heads = []
                    for w in x.words:
                        if sentence.words[w.head - 1].upos == "VERB":
                            x_heads.append(w.head)
                    for w in y.words:
                        if w.head in x_heads:
                            appended = True
                            pairs.append((ner_mapper[entity_text_string(x)], sentence.words[w.head - 1].text,
                                          ner_mapper[entity_text_string(y)]))
                    if not appended:
                        pairs.append((ner_mapper[entity_text_string(x)], None, ner_mapper[entity_text_string(y)]))
                except KeyError:
                    # entity is not type PER
                    pass
                    # print("WARNING")

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
                    word_list = [word.lemma for word in sentence.words]
                    mask_list = []
                    try:
                        index_x = word_list.index(x)
                        index_y = word_list.index(y)
                        mask_list.append(index_x)
                        mask_list.append(index_y)
                    except ValueError:
                        print("word not found")
                    sentiment = sa.get_sentiment_sentence(word_list, mask_list)
                    pairs.append((ner_x, sentiment, ner_y))

                except KeyError:
                    pass
    return pairs

def get_relations_from_sentences_sentiment_verb(data: Document, ner_mapper: dict, sa: SentimentAnalysis):
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
            verbs = []
            for j, entity in enumerate(sentence.entities):
                curr_words.append(entity)
            for w in sentence.words:
                if w.upos == "VERB":
                    verbs.append((w.text, w.id))
            for x, y in combinations(curr_words, 2):
                try:
                    appended = False
                    x_heads = []
                    for w in x.words:
                        if sentence.words[w.head - 1].upos == "VERB":
                            x_heads.append(w.head)
                    for w in y.words:
                        if w.head in x_heads:
                            appended = True
                            verb = sentence.words[w.head - 1].text
                            sentiment = sa.get_sentiment_word(verb)
                            pairs.append((ner_mapper[entity_text_string(x)], sentiment,
                                          ner_mapper[entity_text_string(y)]))
                    if not appended:
                        word_list = [word.lemma for word in sentence.words]
                        mask_list = []
                        try:
                            index_x = word_list.index(x)
                            index_y = word_list.index(y)
                            mask_list.append(index_x)
                            mask_list.append(index_y)
                        except ValueError:
                            #print("word not found")
                            pass
                        sentiment = sa.get_sentiment_sentence(word_list, mask_list)
                        pairs.append((ner_mapper[entity_text_string(x)], sentiment, ner_mapper[entity_text_string(y)]))
                except KeyError:
                    # entity is not type PER
                    pass
                    # print("WARNING")

    return pairs


def group_relations_filter(pairs, keep):
    relations = defaultdict(list)
    for x, s, y in pairs:
        if x in keep and y in keep and s != 0 and x != y:
            if x < y:
                key = (x, y)
            else:
                key = (y, x)

            relations[key].append(s)

    final_rel = [(k, mean(v)) for k, v in relations.items()]
    return final_rel


def group_relations(pairs):
    relations = defaultdict(list)
    for x, s, y in pairs:
        if s != 0 and x != y:
            if x < y:
                key = (x, y)
            else:
                key = (y, x)

            relations[key].append(s)
    final_rel = [(k, mean(v)) for k, v in relations.items()]
    return final_rel


def flatten_list(l):
    return [j for i in l for j in i]



def fix_ner(data):
    """
    Fixes common problems with NER (detecting only ADJ and not noun after it)
    Keeps only PER entities
    """
    data.entities = []
    noun_criterium = lambda word: word.xpos in ["NNP"] and word.text not in flatten_list(
        [x.split(" ") for x in ents]) and len(
        word.text) > 2

    for i, sentence in enumerate(data.sentences):
        delete_list = []
        #     start_char = None
        #     word_text = None
        #     sentence_len = len(sentence.words)
        #     ents = [ent.text for ent in sentence.entities]
        #     for j, word in enumerate(sentence.words):
        #         # if word.text == "house" or word.text == "father":
        #         #     print("hello")
        #         if start_char:
        #             if sentence_len <= j + 1 or not noun_criterium(sentence.words[j + 1]):
        #                 data.entities.append({
        #                     "text": word_text + " " + word.text,
        #                     "type": "PERSON",
        #                     "start_char": start_char,
        #                     "end_char": word.end_char
        #                 })
        #                 word_text = None
        #                 start_char = None
        #         else:
        #             if noun_criterium(word):
        #                 try:
        #                     if sentence_len > (j + 1) and noun_criterium(sentence.words[j + 1]):
        #                         start_char = word.start_char
        #                         word_text = word.text
        #                     else:
        #                         start_char = None
        #                         word_text = None
        #                         data.entities.append({
        #                             "text": word.text,
        #                             "type": "PERSON",
        #                             "start_char": word.start_char,
        #                             "end_char": word.end_char
        #                         })
        #                 except IndexError:
        #                     print("hello")

        if len(sentence.entities) != 0:
            for j, entity in enumerate(sentence.entities):
                # Keep only PER entities
                if (entity.type == "PER" or entity.type == "PERSON") and "VERB" not in [x.upos for x in entity.words]:
                    if len(entity.words) == 1:
                        if (entity.words[0].upos == "ADJ" or entity.words[0].upos == "VERB") and \
                                data.sentences[i].words[
                                    entity.words[0].head - 1].upos == "NOUN":
                            data.sentences[i].entities[j].tokens.append(
                                data.sentences[i].words[entity.words[0].head - 1].parent)
                            data.sentences[i].entities[j].words.append(
                                data.sentences[i].words[entity.words[0].head - 1])
                    data.entities.append(data.sentences[i].entities[j])
                else:
                    delete_list.append(j)
        if not len(delete_list) == 0:
            for e in reversed(delete_list):
                del data.sentences[i].entities[e]

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
            if isinstance(entity, dict):
                name = entity["text"]

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


def get_entities_from_svo_triplets_sentiment(book, e: Eventify, deduplication_mapper, sa: SentimentAnalysis):
    dedup_keys = deduplication_mapper.keys()
    events = e.eventify(book.text)
    NER_containing_events = []
    relations = []
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
            sentiment = sa.get_sentiment_word(v)
            relations.append((deduplication_mapper[s], sentiment, deduplication_mapper[o]))

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
        sentiment = sa.get_sentiment_word(v)
        relations.append((deduplication_mapper[s], sentiment, deduplication_mapper[o]))

    return NER_containing_events, relations


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
