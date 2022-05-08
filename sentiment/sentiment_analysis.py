import pandas as pd


class SentimentAnalysis:
    def __init__(self, lexicon):
        self.lexicon = lexicon

    def get_sentiment_word(self, word):
        wl = word.lower()
        if wl in self.lexicon:
            return self.lexicon[wl]
        else:
            return 0

    def get_sentiment_sentence(self, sentence, mask):
        sum_sen = 0
        counter = 0
        for i, word in enumerate(sentence):
            if i in mask:
                continue
            sum_sen += self.get_sentiment_word(word)
            counter += 1
        return sum_sen / float(counter)


def read_lexicon_sentiwordnet(path):
    lexicon = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            s = line.split('\t')
            if s[1] == '':
                continue
            objScore = 1.0 - (float(s[2]) + float(s[3]))
            words = s[4].split(' ')
            for word in words:
                word = word.split('#')[0]
                lexicon[word] = objScore
    return lexicon


def read_lexicon_afinn_en(path):
    lexicon = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            word, score = line.strip().split('\t')
            lexicon[word] = int(score)
    return lexicon


def read_lexicon_job(path):
    df = pd.read_csv(path, delimiter='\t')
    new = df.filter(['Word', 'AFINN'], axis=1)
    lexicon = dict(zip(new.Word, new.AFINN))
    return lexicon


def read_lexicon_kss(path):
    files = ['_words_Kadunc', '_words_lemmas', '_words_Slolex']
    xx = ['negative', 'positive']
    lexicons = {}
    for file in files:
        label = file.split('_')[-1]
        lexicons[label] = {}
        for pn in xx:
            with open(path + pn + file + '.txt', 'r') as f:
                lexicons[label][pn] = [x.strip() for x in f.readlines()]

    return lexicons


def convert_list_to_dict(list_lexicons):
    lexicon = {}
    for word in list_lexicons['positive']:
        lexicon[word] = 1
    for word in list_lexicons['negative']:
        lexicon[word] = -1
    return lexicon


if __name__ == "__main__":
    job = read_lexicon_job("lexicons/Slovene sentiment lexicon JOB 1.0/Slovene_sentiment_lexicon_JOB.txt")
    kss = convert_list_to_dict(read_lexicon_kss('lexicons/Slovene sentiment lexicon KSS 1.1/')['Slolex'])
    afinn_en = read_lexicon_afinn_en("lexicons/AFINN_en/AFINN-111.txt")
    sentiwordnet_en = read_lexicon_sentiwordnet("lexicons/SentiWordNet.txt")

    sa_kss = SentimentAnalysis(kss)
    sa_job = SentimentAnalysis(job)
    sa_afinn = SentimentAnalysis(afinn_en)
    sa_swn = SentimentAnalysis(sentiwordnet_en)

    sample_slo = "danes gremo v to gnilo hiso.".strip().split(' ')

    print(sa_kss.get_sentiment_word('gniloba'))
    print(sa_job.get_sentiment_word('gniloba'))
    print('\n')

    print(sa_kss.get_sentiment_sentence(sample_slo, []))
    print(sa_job.get_sentiment_sentence(sample_slo, []))
    print('\n')

    sample_eng = "I am very happy to see you.".strip().split(' ')

    print(sa_afinn.get_sentiment_word('horrible'))
    print(sa_swn.get_sentiment_word('horrible'))
    print('\n')

    print(sa_afinn.get_sentiment_sentence(sample_eng, []))
    print(sa_swn.get_sentiment_sentence(sample_eng, []))
    print('\n')
