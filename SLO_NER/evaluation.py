import csv

import pandas as pd

from build_graph import build_graph
from helper_functions import write_pickle, clean_all_ner, read_pickle, ner_extract_all
from recommendation import get_article_similarity

# Read from file if already processed

if __name__ == '__main__':
    CNREC_PATH = './data/CNRec'
    try:
        articles = read_pickle(CNREC_PATH + '/analysis_cnrec.pickle')
    except FileNotFoundError:
        try:
            ner_data = read_pickle(CNREC_PATH + '/ner_data_cnrec.pickle')
        except FileNotFoundError:
            ner_data = ner_extract_all(path=CNREC_PATH + "/CNRec_RawText", gpu=True, website="other", language="en")
            write_pickle(ner_data, CNREC_PATH + '/ner_data_cnrec.pickle')
        articles = clean_all_ner(ner_data)
        # Write data to file
        write_pickle(articles, CNREC_PATH + '/analysis_cnrec.pickle')

    try:
        g = read_pickle(CNREC_PATH + '/graph.pickle')
    except FileNotFoundError:
        g, co = build_graph(articles)
        write_pickle(g, CNREC_PATH + '/graph.pickle')

    # load article id dictionary
    article_mapper = {}
    with open(CNREC_PATH + '/articleToID.csv') as file:
        reader = csv.reader(file)
        for k, v in reader:
            article_mapper[v] = k
    # list articles to dict
    article_dict = {}
    for article in articles:
        article_dict[article_mapper[article["name"]]] = article
    try:
        ground_truth = pd.read_pickle(CNREC_PATH + '/ground_truth.pickle')
    except FileNotFoundError:

        ground_truth = pd.read_csv(CNREC_PATH + '/CNRec_groundTruth.csv')
        ground_truth["ourSim"] = 0.0

        for i, row in ground_truth.iterrows():
            x = article_dict[str(round(row["art1"]))]
            x = ([f[0] for f in x["per"]] + [f[0] for f in x["org"]], x["name"])
            y = article_dict[str(round(row["art2"]))]
            y = ([f[0] for f in y["per"]] + [f[0] for f in y["org"]], y["name"])
            ground_truth.loc[i, "ourSim"] = get_article_similarity(x, y, g)[0]
        ground_truth.to_pickle(CNREC_PATH + "/ground_truth.pickle")

    col1 = ground_truth["ourSim"]
    ground_truth["meanSimRating"] = ground_truth["meanSimRating"] / 2
    col2 = ground_truth["meanSimRating"]
    view = ground_truth[["ourSim", "meanSimRating", "art1", "art2"]]
    art1 = 0
    art2 = 1
    ground_truth_num = 1
    print("Ground truth: ", ground_truth["meanSimRating"][ground_truth_num])
    x = article_dict[str(art1)]
    x = ([f[0] for f in x["per"]] + [f[0] for f in x["org"]], x["name"])
    y = article_dict[str(art2)]
    y = ([f[0] for f in y["per"]] + [f[0] for f in y["org"]], y["name"])
    print("Entities art1: ", x[0])
    print("Entities art2: ", y[0])
    set_x = set(x[0])
    set_y = set(y[0])
    intersection = set_x.intersection(set_y)
    print("Entities in both: ", intersection)
    print("Result: ", get_article_similarity(x, y, g))
    ground_truth["goodRatingOur"] = (ground_truth["ourSim"] >= 0.5).astype(int)
    ground_truth["goodRatingOur75"] = (ground_truth["ourSim"] >= 0.75).astype(int)
    ground_truth["goodRatingDiversityOur"] = (ground_truth["ourSim"] >= 0.5).astype(int)
    ground_truth["goodRatingDiversityOur75"] = (ground_truth["ourSim"] >= 0.75).astype(int)
    ground_truth["correct50"] = (ground_truth["goodRatingOur"] == ground_truth["GoodR+AF8-50"]).astype(int)
    ground_truth["correct75"] = (ground_truth["goodRatingOur75"] == ground_truth["GoodR+AF8-75"]).astype(int)
    ground_truth["correctDiv50"] = (ground_truth["goodRatingDiversityOur"] == ground_truth["diversity+AF8-50"]).astype(
        int)
    ground_truth["correctDiv75"] = (
                ground_truth["goodRatingDiversityOur75"] == ground_truth["diversity+AF8-75"]).astype(int)
    print("Accurate recommendation GR50: ", sum(ground_truth["correct50"]) / len(ground_truth["correct50"]))
    print("Accurate recommendation GR75: ", sum(ground_truth["correct75"]) / len(ground_truth["correct75"]))
    print("Accurate recommendation DR50: ", sum(ground_truth["correctDiv50"]) / len(ground_truth["correctDiv50"]))
    print("Accurate recommendation DR75: ", sum(ground_truth["correctDiv75"]) / len(ground_truth["correctDiv75"]))

    print("Pearson correlation: ", col1.corr(col2))

    print("Spearman correlation: ", col1.corr(col2, method="spearman"))
