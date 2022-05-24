import pickle



if __name__ == '__main__':
    all = []
    paths = ['pickles/sent_', 'pickles/sent_full_']
    for path in paths:
        with open(path + 'books.pickle', 'rb') as f:
            books = pickle.load(f)
        with open(path + 'corpus_eval.pickle', 'rb') as f:
            corpus_eval = pickle.load(f)
        all.append((corpus_eval, books, path.split('/')[1]))

    pass