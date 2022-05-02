import os
import re
import pickle
from preprocess_general import get_sem_eval

DIR = 'events/brat/'

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu)"
digits = "([0-9])"



def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    if "e.g." in text: text = text.replace("e.g.", "e<prd>g<prd>")
    if "i.e." in text: text = text.replace("i.e.", "i<prd>e<prd>")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("...", "<prd><prd><prd>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def get_story(filename):
    with open(DIR + filename + '.txt', 'r') as f:
        text_raw = f.read()
    with open(DIR + filename + '.ann', 'r') as f:
        ann_raw = f.read()

    ann = dict()
    for line in ann_raw.split('\n'):
        if line.startswith('T'):
            sep = line.split('\t')
            sep2 = sep[1].split(' ')
            if sep2[0] != 'EVENT':
                continue
            ann[sep[0]] = {'event': sep[2], 'start': int(sep2[1]), 'end': int(sep2[2])}

    return text_raw, ann

def get_combination_of_events(doc_name, sentence):
    events = []
    for i, event in enumerate(sentence['events']):
        for j, event in enumerate(sentence['events']):
            if i >= j:
                continue
            try:
                event_pair = [doc_name, sentence['text'], sentence['events'][i]['sentence_index'], sentence['events'][j]['sentence_index'], None]
                events.append(event_pair)
            except KeyError:
                continue
    return events

def get_combination_of_events_double(doc_name, sentence):
    events = []
    for i, event in enumerate(sentence['events']):
        for j, event in enumerate(sentence['events']):
            if i == j:
                continue
            try:
                event_pair = [doc_name, sentence['text'], sentence['events'][i]['sentence_index'],
                              sentence['events'][j]['sentence_index'], None]
                events.append(event_pair)
            except KeyError:
                continue
    return events


if __name__ == '__main__':
    files = set()
    [files.add(x) for x in [os.path.splitext(filename)[0] for filename in os.listdir(DIR)]]

    all_stories = {}
    for filename in files:
        text, ann = get_story(filename)
        all_stories[filename] = {'text': text, 'ann': ann}

    for story in all_stories:
        all_stories[story]['sentence_list'] = split_into_sentences(all_stories[story]['text'])
        all_stories[story]['sentences'] = {}
        c_index_start = 0
        for i, sentence in enumerate(all_stories[story]['sentence_list']):
            c_index_end = c_index_start + len(sentence)
            all_stories[story]['sentences'][i] = {'text': sentence, 'start': c_index_start, 'end': c_index_end, 'events': []}
            c_index_start = c_index_end + 1

        events = sorted(all_stories[story]['ann'].items(), key=lambda x: x[1]['start'])
        for event_id, event in events:
            e_start = event['start']
            e_end = event['end']
            for i, sentence in enumerate(all_stories[story]['sentence_list']):
                if e_start >= all_stories[story]['sentences'][i]['start'] and e_end < all_stories[story]['sentences'][i]['end']:
                    all_stories[story]['sentences'][i]['events'].append(event)
                    break

    for story in all_stories:
        for sentence in all_stories[story]['sentences']:
            text = all_stories[story]['sentences'][sentence]['text']
            for i, event in enumerate(all_stories[story]['sentences'][sentence]['events']):
                word = event['event']
                try:
                    index = text.split().index(word)
                except ValueError:
                    print(ValueError)
                    continue
                all_stories[story]['sentences'][sentence]['events'][i]['sentence_index'] = (index, index)


    all_stories_filtered = {}
    all_stories_filtered_min2 = {}
    for story in all_stories:
        all_stories_filtered[story] = {}
        all_stories_filtered_min2[story] = {}
        for sentence in all_stories[story]['sentences']:
            if len(all_stories[story]['sentences'][sentence]['events']) > 0:
                all_stories_filtered[story][sentence] = all_stories[story]['sentences'][sentence]
            if len(all_stories[story]['sentences'][sentence]['events']) > 1:
                all_stories_filtered_min2[story][sentence] = all_stories[story]['sentences'][sentence]

    stories_prepared_format = {}
    stories_prepared_format_double = {}

    for story in all_stories_filtered_min2:
        stories_prepared_format[story] = {}
        stories_prepared_format_double[story] = {}
        for sentence in all_stories_filtered_min2[story]:
            single = get_combination_of_events(story, all_stories_filtered_min2[story][sentence])
            double = get_combination_of_events_double(story, all_stories_filtered_min2[story][sentence])
            stories_prepared_format[story][sentence] = single
            stories_prepared_format_double[story][sentence] = double

    stories_prepared_format2 = {}
    stories_prepared_format_double2 = {}

    for story in stories_prepared_format:
        stories_prepared_format2[story] = []
        for sentence in stories_prepared_format[story]:
            stories_prepared_format2[story] += stories_prepared_format[story][sentence]

    for story in stories_prepared_format_double:
        stories_prepared_format_double2[story] = []
        for sentence in stories_prepared_format_double[story]:
            stories_prepared_format_double2[story] += stories_prepared_format_double[story][sentence]



    with open('data/litbank_full.pickle', 'wb') as f:
        pickle.dump(all_stories, f, pickle.HIGHEST_PROTOCOL)

    with open('data/litbank_prepared.pickle', 'wb') as f:
        pickle.dump(stories_prepared_format2, f, pickle.HIGHEST_PROTOCOL)

    with open('data/litbank_prepared_double.pickle', 'wb') as f:
        pickle.dump(stories_prepared_format_double2, f, pickle.HIGHEST_PROTOCOL)







