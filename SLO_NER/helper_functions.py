import errno
import os
import pickle
from collections import Counter

import classla
import progressbar
import stanza as stanza
from fuzzywuzzy import fuzz
from fuzzywuzzy.process import extract
from simstring.database.dict import DictDatabase
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher


def ner_extract_all(path="./data", gpu=True, website="RTVSLO", language="sl"):
    """
    NER for all files in the path (recursive walk through directories)
    Args:
        website: From which website articles originate (for naming purposes)
        gpu: Should GPU be used
        path: Path to search files in
    Returns:
        data: list of length num_files
    """
    stanza.download('en')
    classla.download('sl')
    if gpu and language == "sl":
        nlp = classla.Pipeline("sl", processors='tokenize,ner')
    elif not gpu and language == "sl":
        nlp = classla.Pipeline("sl", processors='tokenize,ner', use_gpu=False)
    elif gpu and language == "en":
        nlp = stanza.Pipeline(lang="en", processors='tokenize, ner')
    elif not gpu and language == "en":
        nlp = stanza.Pipeline(lang="en", processors='tokenize, ner', use_gpu=False)
    data = []
    count = 0
    if website == "RTVSLO":
        with progressbar.ProgressBar(max_value=num_files_in_dir(path)) as bar:
            for website in os.listdir(path):
                for year in os.listdir(path + '/' + website):
                    for file in os.listdir(path + '/' + website + '/' + year):
                        filename = file.split('_')
                        name = filename.pop()
                        if website == 'RTVSLO':
                            name = filename.pop() + '-' + name
                        date = filename[0] + '.' + year
                        del filename[0]
                        category = " ".join(filename)
                        try:
                            with open('./data/' + website + '/' + year + '/' + file, 'r') as f:
                                ner = ner_extraction(f.read(), nlp)
                                data.append({
                                    "name": name,
                                    "date": date,
                                    "category": category,
                                    "per": ner["per"],
                                    "org": ner["org"],
                                    "website": website
                                })
                        except:
                            continue
                        count += 1
                        bar.update(count)
    else:
        with progressbar.ProgressBar(max_value=num_files_in_dir(path)) as bar:
            for file in os.listdir(path):
                try:
                    with open(path + '/' + file, 'r') as f:
                        ner = ner_extraction(f.read(), nlp)
                        data.append({
                            "name": file,
                            "per": ner["per"],
                            "org": ner["org"],
                            "website": "cnrec"
                        })
                except:
                    continue
                count += 1
                bar.update(count)
    return data


def clean_all_ner(data):
    """
    Cleans data by removing duplicates across all the ner extraction
    Args:
        data: list of NER extractions (Output of ner_extract_all)

    Returns:
        data: cleaned list of NER extractions
    """
    entities = {}
    entities_alias = {}
    db = DictDatabase(CharacterNgramFeatureExtractor(2))
    count = 0
    searcher = Searcher(db, CosineMeasure())
    with progressbar.ProgressBar(max_value=len(data)) as bar:
        for file in data:
            for key in ["per", "org"]:
                for j, entity in enumerate(file[key]):
                    added = False
                    # Try finding it in aliases
                    try:
                        e = entities_alias[entity[0]]
                        try:
                            entities[e][entity[0]][0] += [(count, key, j)]
                            entities[e][entity[0]][1] += entity[1]
                        except KeyError:
                            entities[e][entity[0]] = [[(count, key, j)], entity[1]]
                        added = True
                    except KeyError:
                        # Reduce complexity
                        results = searcher.ranked_search(entity[0], 0.6)
                        for i, e in enumerate(results):
                            if fuzz.token_set_ratio(entity[0], e[1]) > 70:
                                added = True
                                # if key != entities[e][e][0][0][1]:
                                #
                                #     del ner[key][i]
                                try:
                                    entities[e[1]][entity[0]][0] += [(count, key, j)]
                                    entities[e[1]][entity[0]][1] += entity[1]
                                    entities_alias[entity[0]] = e[1]
                                except KeyError:
                                    entities[e[1]][entity[0]] = [[(count, key, j)], entity[1]]
                                    entities_alias[entity[0]] = e[1]
                                break

                        if not added:
                            entities[entity[0]] = {entity[0]: [[(count, key, j)], entity[1]]}
                            entities_alias[entity[0]] = entity[0]
                            db.add(entity[0])
            count += 1
            bar.update(count)
        for key, value in entities.items():
            maxi = 0
            maxi_key = ""
            for k, val in value.items():
                if val[1] > maxi:
                    maxi_key = k
                    maxi = val[1]
            for k, val in value.items():
                for en in val[0]:
                    data[en[0]][en[1]][en[2]] = (maxi_key, data[en[0]][en[1]][en[2]][1])
    return data


def ner_extraction(text, classla_pipeline):
    """
    Performs NER analysis on the text,
    removes duplicates and groups entries referring to the same entity.

        Parameters:
            text (string): Text to be analyzed
            classla_pipeline: Pipeline that does processing

        Returns:
            entities (dict): Dictionary with keys PER, ORG, LOC that contain list of entities (strings)
    """
    ner_data = classla_pipeline(text)
    entities = {
        "per": [],
        "org": []
    }
    # Add found entities to list
    for i, entity in enumerate(ner_data.entities):
        # Skips entity types that we do not intend to keep
        try:
            if entity.type == "PERSON":
                entity.type = "per"
            entities[entity.type.lower()].append("")
        except KeyError:
            continue
        for token in entity.tokens:
            entities[entity.type.lower()][-1] = entities[entity.type.lower()][-1] + token.text + " "

        # Remove last space
        entities[entity.type.lower()][-1] = entities[entity.type.lower()][-1][:-1]
    # Clean data
    for key, value in entities.items():
        # Remove "-*" eg. SD-ja, SD-jev -> SD
        for i, v in enumerate(value):
            tmp = v.split('-')
            if len(tmp) != 1:
                entities[key][i] = tmp[0]
            if key == "per" and "Minister" in v:
                v = v[9:]

        # Remove duplicates
        entities[key] = fuzzywuzzy_custom_dedupe(value, threshold=70)
        unwanted = []
        if key == "per":
            for i, v in enumerate(entities[key]):
                # Remove if only first or last name
                if not " " in v[0]:
                    # Check if entity type was missed
                    if v[0] in entities["org"]:
                        for x in range(0, v[1]):
                            entities["org"].append(v[0])
                    unwanted.append(i)
        entities[key] = remove_from_list(entities[key], unwanted)
    # for key, value in entities.items():
    #     print("Values for ", key, ": ", value)
    return entities


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
    dict = {
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
        res = dict[month]
    except KeyError:
        return -1
    return res


def get_link_from_article(website, article):
    """
    Constructs a link to the article from article dict

        Parameters:
            website (string): One of [24ur, RTVSLO]
            article (object): article object

        Returns:
            url (string): url to the file
    """
    if website == "RTVSLO":
        article_name = article["name"].split(".")[0]
        article_id = article_name.split("-")[-1]
        article_name = "-".join(article_name.split("-")[:-1])
        url = "https://www.rtvslo.si/" + "/".join(
            article["category"].split(" ")) + "/" + article_name + "/" + article_id

    return url


if __name__ == "__main__":
    # print("hello")
    # Initialize CLASSLA pipeline
    # classla.download('sl')
    # nlp = classla.Pipeline("sl", processors='tokenize,ner')
    # ner_res = ner_extraction("Strokovna svetovalna skupina za covid-19 je že minuli teden predlagala podaljšanje epidemije, pogoji za to pa so še vedno izpolnjeni, je izpostavila vodja skupine in infektologinja Mateja Logar. Se pa, če bodo tudi v sredo epidemiološki podatki ugodni, po njenih navedbah sproščajo pogoji za rumeno fazo in s tem povezano sproščanje ukrepov.\
    #                Strokovna svetovalna skupina je že minuli teden predlagala podaljšanje epidemije, pogoji za to pa so še vedno izpolnjeni, je izpostavila vodja skupine in infektologinja Mateja. Se pa, če bodo tudi v sredo epidemiološki podatki ugodni, po njenih navedbah sproščajo pogoji za rumeno fazo in s tem povezano sproščanje ukrepov.\
    #                Strokovna svetovalna skupina je. NSI je stranka, ki je v minulem letu. Vodstvo LMS-ja je lani opravilo.", nlp)
    # for key, value in ner_res.items():
    #     print("Values for ", key, ": ", value)
    article = {
        "name": "mursic-govoriti-da-so-ljudje-krivi-ker-so-poredni-je-omalovazujoce-549022.txt",
        "category": "slovenija"
    }
    # print(get_link_from_article("RTVSLO", article))
    # print(fuzzywuzzy_custom_dedupe(["Joze Anton", "Joze Antonov", "Miha Novak", "Novak Miha"]))
    v = "Minister Gantar"
    v = v[9:]
    print(v)
