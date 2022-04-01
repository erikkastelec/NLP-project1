import collections
import random
import signal

import pandas
import pandas as pd
import progressbar

from helper_functions import read_pickle, get_link_from_article, write_pickle
from recommendation import get_article_similarity

if __name__ == '__main__':

    # HALF NEGATIVE, HALF POSITIVE
    NUM_EXAMPLES = 50

    try:
        evaluation = pd.read_csv("eval.csv")
    except FileNotFoundError:
        try:
            articles = read_pickle('./analysis.pickle')
            graph = read_pickle('./graph.pickle')
            try:
                node_to_article_map = read_pickle('./node_to_article_map.pickle')
            except FileNotFoundError:
                node_to_article_map = {}
                for i in range(0, len(articles)):
                    for per in articles[i]["per"]:
                        try:
                            node_to_article_map[per[0]].append(i)
                        except KeyError:
                            node_to_article_map[per[0]] = [i]
                    for org in articles[i]["org"]:
                        try:
                            node_to_article_map[org[0]].append(i)
                        except KeyError:
                            node_to_article_map[org[0]] = [i]
                write_pickle(node_to_article_map, './node_to_article_map.pickle')
        except FileNotFoundError:
            print("err")
            exit()

        res_dict = {
            "article1": [],
            "article2": [],
            "scoreAlgo": [],
            "recommendPerson": []
        }
        count_negative = 0
        count_positive = 0
        i = 0
        c = 0


        class TimeoutException(Exception):  # Custom exception class
            pass


        def timeout_handler(signum, frame):  # Custom signal handler
            raise TimeoutException


        # Change the behavior of SIGALRM
        signal.signal(signal.SIGALRM, timeout_handler)
        stuck = 0
        queue = collections.deque([], 100)
        with progressbar.ProgressBar(max_value=NUM_EXAMPLES) as bar:
            while count_negative < NUM_EXAMPLES / 2 or count_positive < NUM_EXAMPLES / 2:
                if len(queue) < 2 or stuck == 100:
                    stuck = 0
                    queue = collections.deque([], 100)
                    tmp = random.choices(articles, k=2)
                    queue.append(tmp[0])
                    queue.append(tmp[1])
                art1 = queue.pop()
                art2 = queue.pop()
                while (art1 == art2):
                    if (len(queue) != 0):
                        art2 = queue.pop()
                    else:
                        tmp = random.choices(articles, k=2)
                        queue.append(tmp[0])
                        queue.append(tmp[1])
                        art1 = queue.pop()
                        art2 = queue.pop()
                        break
                link1 = get_link_from_article("RTVSLO", art1)
                link2 = get_link_from_article("RTVSLO", art2)
                x = art1
                y = art2
                max = 0
                max_n = ""
                for per in x["per"]:
                    if max < per[1]:
                        max_n = per[0]
                        max = per[1]
                for m in random.choices(node_to_article_map[max_n], k=4):
                    queue.append(articles[m])
                x = ([f[0] for f in x["per"]] + [f[0] for f in x["org"]], x["name"])
                y = ([f[0] for f in y["per"]] + [f[0] for f in y["org"]], y["name"])
                signal.alarm(3)
                try:
                    score = get_article_similarity(x, y, graph)[0]
                    if c == 100:
                        print("Not stuck, just working")
                        c = 0
                    if count_negative < NUM_EXAMPLES / 2 and score < 0.5:
                        res_dict["article1"].append(link1)
                        res_dict["article2"].append(link2)
                        res_dict["scoreAlgo"].append(score)
                        count_negative += 1
                        i += 1
                        stuck = 0
                        bar.update(i)
                    if count_positive < NUM_EXAMPLES / 2 and score >= 0.5:
                        res_dict["article1"].append(link1)
                        res_dict["article2"].append(link2)
                        res_dict["scoreAlgo"].append(score)
                        count_positive += 1
                        i += 1
                        bar.update(i)
                        stuck = 0
                    c += 1
                    stuck += 1
                except TimeoutException:
                    continue
                else:
                    signal.alarm(0)

        for i in range(0, NUM_EXAMPLES):
            score = res_dict["scoreAlgo"][i]
            link1 = res_dict["article1"][i]
            link2 = res_dict["article2"][i]
            print(str(i), ": ", link1, " ", link2)
            res_dict["recommendPerson"].append(int(input("Input score: 0 or 1: ")))

        eval = pandas.DataFrame(res_dict)
        eval.to_csv("eval.csv")
