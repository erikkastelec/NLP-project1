import os

DIR = 'events/brat/'



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



if __name__ == '__main__':
    files = set()
    [files.add(x) for x in [os.path.splitext(filename)[0] for filename in os.listdir(DIR)]]

    all_stories = {}
    for filename in files:
        text, ann = get_story(filename)
        all_stories[filename] = {'text': text, 'ann': ann}

    print("x")






