import argparse
import logging
from time import sleep

import progressbar
from Spider_24ur import Spider_24ur
from Spider_RTVSLO import Spider_RTVSLO
from fuzzywuzzy import fuzz
from scrapyscript import Processor, Job
from simstring.database.dict import DictDatabase
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher

from helper_functions import num_files_in_dir, read_pickle, write_pickle, ner_extract_all, clean_all_ner


def main(website, mode, gpu):
    logger = logging.getLogger("NewsScraper")
    data = {
        "24ur": {
            "processed_files": num_files_in_dir('./data/24ur')
        },
        "RTVSLO": {
            "categories": ["slovenija", 'svet', 'sport', 'kultura', 'zabava-in-slog'],
            "processed_files": num_files_in_dir('./data/RTVSLO')
        }
    }
    processor = Processor(settings=None)
    if mode == 'scrape':
        if website == '24ur':
            try:
                page = read_pickle('./progress/24ur.pickle')
            except FileNotFoundError:
                page = 0
            while True:
                page = page + 1
                print("Scraping page: ", page)
                job = Job(Spider_24ur, page, 'https://www.24ur.com/arhiv?stran=' + str(page))
                processor_out = processor.run([job])
                try:
                    if not processor_out[0]["continue"]:
                        break
                except Exception as e:
                    print(e)
                    logger.error(e)

        elif website == 'RTVSLO':
            try:
                (category_done, page) = read_pickle('./progress/RTVSLO.pickle')
                category_done = data[website]["categories"].index(category_done)
            except FileNotFoundError:
                category_done = 0
                page = 0
            for i in range(category_done, len(data[website]["categories"])):
                print("Scraping category: ", data[website]["categories"][i])
                while True:
                    page = page + 1
                    print("Scraping page: ", page)
                    job = Job(Spider_RTVSLO, data[website]["categories"][i], page,
                              "https://www.rtvslo.si/" + data[website]["categories"][i] + '/arhiv/?&page=' + str(page))
                    processor_out = processor.run([job])
                    try:
                        if not processor_out[0]["continue"]:
                            break
                    except Exception as e:
                        logger.error(e)
                page = 0
    elif mode == 'process':
        # Read from file if already processed
        try:
            data = read_pickle('./analysis.pickle')
        except FileNotFoundError:
            try:
                ner_data = read_pickle('./ner_data.pickle')
            except FileNotFoundError:
                ner_data = ner_extract_all(path="./data", gpu=True, website="RTVSLO")
                write_pickle(ner_data, './ner_data.pickle')
            data = clean_all_ner(ner_data)
            # Write data to file
            write_pickle(data, './analysis.pickle')
    elif mode == "ner":
        try:
            ner_data = read_pickle('./ner_data.pickle')
        except FileNotFoundError:
            ner_data = ner_extract_all(path="./data", gpu=True, website="RTVSLO")
            write_pickle(ner_data, './ner_data.pickle')
    else:
        # FOR TESTING
        try:
            data = read_pickle('./ner_data.pickle')
            print("hello")
        except FileNotFoundError:
            with progressbar.ProgressBar(max_value=10) as bar:
                for i in range(10):
                    sleep(1)
                    bar.update(i)
        list = []
        db = DictDatabase(CharacterNgramFeatureExtractor(2))
        for dat in data:
            for key, value in dat.items():
                for v in value:
                    db.add(v[0])
                    list.append(v[0])

        searcher = Searcher(db, CosineMeasure())
        search_string = 'Janezu Janšiju'
        results = searcher.ranked_search(search_string, 0.6)
        for e in list:
            score = fuzz.token_set_ratio(search_string, e)
            if score > 70:
                print(e, score)

        print(results)
        # df = pd.DataFrame(data=per, columns=['name'])
        # deduplipy_per = Deduplicator(col_names=["name"])
        # deduplipy_per.fit(df)
        # write_pickle(deduplipy_per, './dedup.pickle')


if __name__ == "__main__":
    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument('--website', choices=['24ur', 'RTVSLO'], help='Website to be scraped', default="RTVSLO")
    argumentParser.add_argument('--mode', choices=['process', 'scrape', 'ner'], help='Operation mode',
                                default='process')
    argumentParser.add_argument('--gpu', type=bool, help='Operation mode', default=True)

    args = vars(argumentParser.parse_args())
    # main(args["website"], args['mode'], args["gpu"])
    main('RTVSLO', "process", True)
