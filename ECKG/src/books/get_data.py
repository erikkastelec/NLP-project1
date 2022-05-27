FILE_PATH = 'corpus.tsv'


def get_person_id(person, people):
    for p in people:
        if p.name == person:
            return p.id

    for p in people:
        if p.name in p.sinonims:
            return p.id


def get_person(id, people):
    for p in people:
        if p.id == id:
            return p


def get_relation_by_ids(relation, people):
    s = relation.split(' ')
    if '-' in s:
        sentiment = 'negative'
        s = relation.split(' - ')
    elif '+' in s:
        sentiment = 'positive'
        s = relation.split(' + ')
    elif '/' in s:
        sentiment = 'neutral'
        s = relation.split(' / ')

    id1 = get_person_id(s[0], people)
    id2 = get_person_id(s[1], people)

    return (id1, id2, sentiment)


class Person:
    def __init__(self, names, idp):
        self.name = names[0]
        self.sinonims = names
        self.id = idp


class Book:
    def __init__(self, filename, language, title, author, year, num_words, characters, protagonists, protagonists_id,
                 protagonists_names, antagonists, antagonists_id, antagonists_names, relations, text):
        self.file_name = filename
        self.language = language
        self.title = title
        self.author = author
        self.year = year
        self.num_words = num_words
        self.text = text
        self.relations = relations
        self.protagonists = protagonists
        self.antagonists = antagonists
        self.protagonists_names = protagonists_names
        self.protagonists_id = protagonists_id
        self.antagonists_id = antagonists_id
        self.antagonists_names = antagonists_names
        self.characters = characters


def get_data(file_path, get_text=False):
    with open(file_path, 'r') as f:
        books = []
        for i, line in enumerate(f):
            if i == 0:
                continue

            split = line.split('\t')
            if '' == split[0]:
                continue

            text = None
            if get_text:
                if len(file_path.split("/")) == 1:
                    text_path = split[0]
                else:
                    text_path = "/".join(file_path.split("/")[:-1]) + "/" + split[0]
                try:
                    with open(text_path, 'r') as f2:
                        text = f2.read()
                except:
                    text = None

            people = split[8].split(', ')
            if len(people) == 1 and people[0] == '':
                continue
            characters = []
            for pi, p in enumerate(people):
                new_person = Person(p.split('/'), pi)
                characters.append(new_person)

            protagonists_names = split[7].split(', ')
            antagonists_names = split[6].split(', ')

            if len(protagonists_names) == 1 and protagonists_names[0] == '' or protagonists_names[0] == '?':
                protagonists_names = []
                protagonists_id = []
                protagonists = []
            else:
                protagonists_id = [get_person_id(p, characters) for p in protagonists_names]
                protagonists = [get_person(p, characters) for p in protagonists_id]

            if len(antagonists_names) == 1 and antagonists_names[0] == '' or antagonists_names[0] == '?':
                antagonists_names = []
                antagonists_id = []
                antagonists = []
            else:
                antagonists_id = [get_person_id(p, characters) for p in antagonists_names]
                antagonists = [get_person(p, characters) for p in antagonists_id]

            relations = split[9].split(', ')
            if len(relations) == 1 and relations[0] == '':
                relations_ids = []
            else:
                relations_ids = [get_relation_by_ids(r, characters) for r in relations]

            new_book = Book(split[0], split[1], split[2], split[3], int(split[4]), int(split[5]), characters,
                            protagonists, protagonists_id, protagonists_names, antagonists, antagonists_id,
                            antagonists_names, relations_ids, text)

            books.append(new_book)
    return books


if __name__ == '__main__':
    data = get_data(FILE_PATH, get_text=True)

    slo_books = [d for d in data if d.language == 'slovenian']
    eng_books = [d for d in data if d.language == 'english']

    all_len = [d.num_words for d in data]
    slo_len = [d.num_words for d in slo_books]
    eng_len = [d.num_words for d in eng_books]

    print("Slo books:", min(slo_len), max(slo_len), sum(slo_len) / len(slo_len))
    print("Eng books:", min(eng_len), max(eng_len), sum(eng_len) / len(eng_len))
    print("All books:", min(all_len), max(all_len), sum(all_len) / len(all_len))
