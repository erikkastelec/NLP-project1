import copy
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
from classla.models.common.doc import Word
from fuzzywuzzy import fuzz
from fuzzywuzzy.process import extract
from simstring.database.dict import DictDatabase
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher

from ECKG.src.SloCOREF.contextual_model_bert import ContextualControllerBERT
from ECKG.src.SloCOREF.data import Token, Mention
from ECKG.src.eventify import Eventify
from books.get_data import get_data
from sentiment.sentiment_analysis import SentimentAnalysis

import coreferee

OBJECT_DEPS = {"dobj", "dative", "attr", "oprd"}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}


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


"""
Coreference pipeline class. Modified from: https://github.com/RSDO-DS3/SloCOREF
"""


class SloveneCorefPipeline:
    def __init__(self, doc: Document, pipeline, coref_model_path="../SloCOREF_contextual_model_bert/bert_model"):
        self.doc = doc
        self.pipeline = pipeline
        self.coref_model = ContextualControllerBERT.from_pretrained(coref_model_path)
        self.coref_model.eval_mode()

    def classla_output_to_coref_input(self):
        # Transforms CLASSLA's output into a form that can be fed into coref model.
        output_tokens = {}
        output_sentences = []
        output_mentions = {}
        output_clusters = []

        str_document = self.doc.text
        start_char = 0
        MENTION_MSD = {"N", "V", "R", "P"}  # noun, verb, adverb, pronoun

        current_mention_id = 1
        token_index_in_document = 0
        for sentence_index, input_sentence in enumerate(self.doc.sentences):
            output_sentence = []
            mention_tokens = []
            for token_index_in_sentence, input_token in enumerate(input_sentence.tokens):
                input_word = input_token.words[0]
                output_token = Token(str(sentence_index) + "-" + str(token_index_in_sentence),
                                     input_word.text,
                                     input_word.lemma,
                                     input_word.xpos,
                                     sentence_index,
                                     token_index_in_sentence,
                                     token_index_in_document)

                # FIXME: This is a possibly inefficient way of finding start_char of a word. Stanza has this functionality
                #  implemented, Classla unfortunately does not, so we resort to a hack
                new_start_char = str_document.find(input_word.text, start_char)
                output_token.start_char = new_start_char
                if new_start_char != -1:
                    start_char = new_start_char

                if len(mention_tokens) > 0 and mention_tokens[0].msd[0] != output_token.msd[0]:
                    output_mentions[current_mention_id] = Mention(current_mention_id, mention_tokens)
                    output_clusters.append([current_mention_id])
                    mention_tokens = []
                    current_mention_id += 1

                # Simplistic mention detection: consider nouns, verbs, adverbs and pronouns as mentions
                if output_token.msd[0] in MENTION_MSD:
                    mention_tokens.append(output_token)

                output_tokens[output_token.token_id] = output_token
                output_sentence.append(output_token.token_id)
                token_index_in_document += 1

            # Handle possible leftover mention tokens at end of sentence
            if len(mention_tokens) > 0:
                output_mentions[current_mention_id] = Mention(current_mention_id, mention_tokens)
                output_clusters.append([current_mention_id])
                mention_tokens = []
                current_mention_id += 1

            output_sentences.append(output_sentence)

        return Document(1, output_tokens, output_sentences, output_mentions, output_clusters)

    def predict(self, threshold, return_singletons):
        # 1. re-format classla_output into coref_input (incl. mention detection)
        coref_input = self.classla_output_to_coref_input()

        # 2. process prepared input with coref
        coref_output = self.coref_model.evaluate_single(coref_input)

        # 4. prepare response (mentions + coreferences)
        coreferences = []
        coreferenced_mentions = set()
        for id2, id1s in coref_output["predictions"].items():
            if id2 is not None:
                for id1 in id1s:
                    mention_score = coref_output["scores"][id1]

                    if threshold is not None and mention_score < threshold:
                        continue

                    coreferenced_mentions.add(id1)
                    coreferenced_mentions.add(id2)

                    coreferences.append({
                        "id1": int(id1),
                        "id2": int(id2),
                        "score": mention_score
                    })

        mentions = []
        for mention in coref_input.mentions.values():
            [sentence_id, token_id] = [int(idx) for idx in mention.tokens[0].token_id.split("-")]
            mention_score = coref_output["scores"][mention.mention_id]

            if return_singletons is False and mention.mention_id not in coreferenced_mentions:
                continue

            # while this is technically already filtered with coreferenced_mentions, singleton mentions aren't, but they
            # have some score too that can be thresholded.
            # if req_body.threshold is not None and mention_score < req_body.threshold:
            #    continue

            mention_raw_text = " ".join([t.raw_text for t in mention.tokens])
            mentions.append(
                {
                    "id": mention.mention_id,
                    "start_idx": mention.tokens[0].start_char,
                    "length": len(mention_raw_text),
                    "ner_type": self.doc.sentences[sentence_id].tokens[token_id].ner.replace("B-", "").replace(
                        "I-", ""),
                    "msd": mention.tokens[0].msd,
                    "text": mention_raw_text
                }
            )

        return {
            "mentions": mentions,
            "coreferences": sorted(coreferences, key=lambda x: x["id1"])
        }


class EnglishCorefPipeline:
    def __init__(self):
        "python -m spacy download en_core_web_trf "
        nlp = spacy.load("en_core_web_trf")
        nlp.add_pipe('coreferee')
        self.pipeline = nlp
        self.rules_analyzer = nlp.get_pipe('coreferee').annotator.rules_analyzer
        self.doc = None
        self.coref_chains = None
        self.mapping_dict = None
        self.new_entity_names = None

    def process_text(self, text):
        self.doc = self.pipeline(text)

    def resolve_coref(self, id):
        # t = find_similar(self.doc[id].text, self.new_entity_names)
        # if t:
        #     print(t)
        #     print(self.doc[id].text)
        # if not t:
        try:
            t = self.mapping_dict[id]
        except KeyError:
            t = None
        return t
        # return self.coref_chains.resolve(self.doc[id])

    def get_proper_coref_name(self, id):
        return self.rules_analyzer.get_propn_subtree(self.doc[id])

    def unify_naming_coreferee(self, deduplication_mapper):
        if not self.doc:
            raise Exception("No text has been processed. (call process_text(text))")
        dedup_keys = deduplication_mapper.keys()
        disregard_list = ["man", "woman", "name", ]
        chains = self.doc._.coref_chains
        self.coref_chains = chains
        named_entity_chains = {}
        new_chains = {}
        # (spacy word, count)
        new_chains_words = {}

        for chain in chains:
            if len(chain.mentions[chain.most_specific_mention_index]) == 1:
                word = self.doc[chain.mentions[chain.most_specific_mention_index].root_index]
                name = list_to_string([x.text for x in self.rules_analyzer.get_propn_subtree(word)])
                if isinstance(name, str) and name == "":
                    name = list_to_string(
                        [self.doc[x].text for x in chain.mentions[chain.most_specific_mention_index].token_indexes])

                similar = find_similar(name, dedup_keys, similarity=80)
                if similar:
                    try:
                        named_entity_chains[similar] = named_entity_chains[similar] + flatten_list(chain.mentions)
                    except KeyError:
                        named_entity_chains[similar] = flatten_list(chain.mentions)
                        deduplication_mapper[name] = similar
                else:

                    if word.tag_ == "NN" and word.text not in disregard_list:
                        try:
                            new_chains_words[name] = (new_chains_words[name][0], new_chains_words[name][1] + 1)
                            new_chains[name] = new_chains[name] + flatten_list(chain.mentions)
                        except KeyError:
                            new_chains_words[name] = (word, 1)
                            new_chains[name] = flatten_list(chain.mentions)

        # sort dict
        new_chains_words = dict(sorted(new_chains_words.items(), key=lambda x: -x[1][1]))
        # Keep only top 10% new ones
        new_chains_words = keep_top_dict(new_chains_words)
        mapping_dict = {}

        for key, value in new_chains_words.items():
            new_chains_words[key] = (value[0], new_chains[key])
            for x in new_chains[key]:
                mapping_dict[x] = key
        self.new_entity_names = new_chains_words.keys()
        for key, value in named_entity_chains.items():
            for x in value:
                mapping_dict[x] = key
            deduplication_mapper[key] = key
        self.mapping_dict = mapping_dict

        return named_entity_chains, new_chains_words


def get_relations_from_sentences(data: Document, ner_mapper: dict, coref_pipeline=None):
    """
    Find pairs of entities, which co-occur in the same sentence.
    Returns:
        list of entity verb entity pairs
    """

    class TempWord:
        def __init__(self, word: Word, id):
            self.words = [word]
            self.id = id

    pairs = []
    dedup_keys = ner_mapper.keys()
    id_count = 0
    # data = copy.deepcopy(data)
    for i, sentence in enumerate(data.sentences):
        og_len = len(sentence.entities)
        curr_entities = sentence.entities
        named_entity_word_ids = [x.id - 1 for x in flatten_list([y.words for y in sentence.entities])]
        if coref_pipeline:
            for c, id in enumerate(range(id_count, id_count + len(sentence.words))):
                if c not in named_entity_word_ids:
                    try:
                        ssss = coref_pipeline.coref_chains.resolve(coref_pipeline.doc[id])

                        print(coref_pipeline.mapping_dict[id])
                        print(ssss)
                        tmp = coref_pipeline.mapping_dict[id]
                        sentence.words[c].text = tmp
                        # curr_entities.append(TempWord(sentence.words[c], c+1))
                        if ssss:
                            for ss in ssss:
                                curr_entities.append(ss)
                    except KeyError:
                        pass

            id_count += len(sentence.words)
        if len(curr_entities) > 1:
            curr_words = []
            verbs = []
            for j, entity in enumerate(curr_entities):
                curr_words.append(entity)
            for w in sentence.words:
                if w.upos == "VERB":
                    verbs.append((w.text, w.id))
            for x, y in combinations(curr_words, 2):
                try:
                    appended = False
                    # for verb in verbs:
                    #     if min(x.words[0].id, y.words[0].id) < verb[1] < max(x.words[-1].id, y.words[-1].id):
                    #         appended = True
                    #         pairs.append((ner_mapper[list_to_string([a.text for a in x.words])], verb[0], ner_mapper[list_to_string([a.text for a in y.words])]))
                    if entity_text_string(x) != entity_text_string(y):

                        try:
                            x_heads = []
                            for w in x.words:
                                if sentence.words[w.head - 1].upos == "VERB":
                                    x_heads.append(w.head)
                            for w in y.words:
                                if w.head in x_heads:
                                    appended = True
                                    pairs.append((ner_mapper[entity_text_string(x)], sentence.words[w.head - 1].text,
                                                  ner_mapper[entity_text_string(y)]))
                        except Exception:
                            pass
                        if not appended:
                            tmp1 = entity_text_string(x)
                            tmp2 = entity_text_string(y)
                            try:
                                tmp1 = ner_mapper[tmp1]
                            except KeyError:
                                tmp1 = find_similar(tmp1, dedup_keys, similarity=80)
                            try:
                                tmp2 = ner_mapper[tmp2]
                            except KeyError:
                                tmp2 = find_similar(tmp2, dedup_keys, similarity=80)
                            if tmp1 and tmp2:
                                pairs.append((tmp1, None, tmp2))
                except KeyError:
                    print('what')
                    # entity is not type PER
                    pass
                    # print("WARNING")
    print(pairs)
    return pairs


def get_relations_from_sentences_coref_sentence_sent(data: Document, ner_mapper: dict, coref_pipeline=None):
    """
    Find pairs of entities, which co-occur in the same sentence.
    Returns:
        list of entity verb entity pairs
    """

    class TempWord:
        def __init__(self, word: Word, id):
            self.words = [word]
            self.id = id

    pairs = []
    dedup_keys = ner_mapper.keys()
    id_count = 0
    # data = copy.deepcopy(data)

    for i, sentence in enumerate(data.sentences):
        appended_words = 0
        og_len = len(sentence.entities)
        curr_entities = sentence.entities
        named_entity_word_ids = [x.id - 1 for x in flatten_list([y.words for y in sentence.entities])]
        if coref_pipeline:
            for c, id in enumerate(range(id_count, id_count + len(sentence.words))):
                if c not in named_entity_word_ids:
                    try:
                        ssss = coref_pipeline.coref_chains.resolve(coref_pipeline.doc[id])

                        print(coref_pipeline.mapping_dict[id])
                        print(ssss)
                        tmp = coref_pipeline.mapping_dict[id]
                        sentence.words[c].text = tmp
                        # curr_entities.append(TempWord(sentence.words[c], c+1))
                        if ssss:
                            for ss in ssss:
                                curr_entities.append(ss)
                    except KeyError:
                        pass

            id_count += len(sentence.words)
        if len(curr_entities) > 1:

            for x, y in combinations(curr_entities, 2):
                word_list = [word.lemma for word in sentence.words]
                mask_list = []
                try:
                    tmp1 = entity_text_string(x)
                    tmp2 = entity_text_string(y)
                    try:
                        tmp1 = ner_mapper[tmp1]
                    except KeyError:
                        tmp1 = find_similar(tmp1, dedup_keys, similarity=80)
                    try:
                        tmp2 = ner_mapper[tmp2]
                    except KeyError:
                        tmp2 = find_similar(tmp2, dedup_keys, similarity=80)
                    if tmp1 and tmp2:
                        pairs.append((tmp1, None, tmp2))

                except KeyError:
                    # entity is not type PER
                    pass
                    # print("WARNING")
    print(pairs)
    return pairs

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
    if isinstance(a, spacy.tokens.token.Token):
        return a.text
    return " ".join(x.text for x in a.words)



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
                            # print("word not found")
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

    token_count = 0
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
        for tc, token in enumerate(sentence.tokens):
            data.sentences[i].tokens[tc].global_id = token_count
            data.sentences[i].words[tc].global_id = token_count
            token_count += 1
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


def get_entities_from_svo_triplets(book, e: Eventify, deduplication_mapper, doc=None, coref_pipeline=None):
    dedup_keys = deduplication_mapper.keys()
    if doc:
        events = e.eventify(book.text, data=doc)
    else:
        events = e.eventify(book.text)
    NER_containing_events = []
    event_entities = []
    event_tmp = []
    for s, v, o in events:
        if isinstance(s, tuple) or isinstance(o, tuple) or s == '<|EMPTY|>' or o == '<|EMPTY|>':
            continue

        s_sim = find_similar(list_to_string([x.text for x in s]), dedup_keys)
        o_sim = find_similar(list_to_string([x.text for x in o]), dedup_keys)
        if s_sim is not None and o_sim is not None:
            # s.text = deduplication_mapper[s_sim]
            # o.text = deduplication_mapper[o_sim]
            NER_containing_events.append((s_sim, list_to_string([x.text for x in v]), o_sim))

        elif len(s) < 3 and len(o) < 3 and coref_pipeline:
            s_ok = None
            o_ok = None
            if not s_sim:
                cor_s = coref_pipeline.resolve_coref(s[0].global_id)
                if cor_s:
                    s_ok = cor_s
            else:
                s_ok = s_sim

            if not o_sim:
                cor_o = coref_pipeline.resolve_coref(o[0].global_id)
                if cor_o:
                    o_ok = cor_o
            else:
                o_ok = o_sim

            # if cor_o:
            #     print("og: " + list_to_string([x.text for x in o]))
            #     print(cor_o)
            # if cor_s:
            #     print("og: " + list_to_string([x.text for x in s]))
            #     print(cor_s)
            if o_ok and s_ok and o_ok != s_ok:
                event_entities.append(o_ok)
                event_entities.append(s_ok)
                event_tmp.append((s_ok, [x.text for x in v], o_ok))

    # print(NER_containing_events)
    deduplication_mapper, count = deduplicate_named_entities(event_entities, count_entities=True)
    for s, v, o in event_tmp:
        NER_containing_events.append((deduplication_mapper[s], v, deduplication_mapper[o]))
    # print(NER_containing_events)
    return NER_containing_events


def get_entities_from_svo_triplets_sentiment(book, e: Eventify, deduplication_mapper, sa: SentimentAnalysis, doc=None,
                                             coref_pipeline=None):
    NER_containing_events = get_entities_from_svo_triplets(book, e, deduplication_mapper, doc=doc,
                                                           coref_pipeline=coref_pipeline)
    relations = []
    for x in range(0, len(NER_containing_events)):
        relations.append((NER_containing_events[x][0], sa.get_sentiment_word(NER_containing_events[x][1]),
                          NER_containing_events[x][2]))
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
