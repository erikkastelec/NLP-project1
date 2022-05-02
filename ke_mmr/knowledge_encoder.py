import requests
from transformers import BertTokenizer
from string import punctuation



class KnowledgeEncoder:
    def __init__(self, tokenizer, limit=5):
        self.max_knowledge = limit
        self.limit = 1 + 2 * self.max_knowledge
        self.tokenizer = tokenizer


    def crawl_concept_net(self, event):
        obj = requests.get('http://api.conceptnet.io/c/en/' + event).json()
        relations = ['CapableOf', 'IsA', 'HasProperty', 'Causes', 'MannerOf', 'CausesDesire', 'UsedFor', 'HasSubevent',
                     'HasPrerequisite', 'NotDesires', 'PartOf', 'HasA', 'Entails', 'ReceivesAction', 'UsedFor',
                     'CreatedBy', 'MadeOf', 'Desires']
        res = []
        for e in obj['edges']:
            if e['rel']['label'] in relations:
                res.append(' '.join([e['rel']['label'], e['end']['label']]))
        return list(dict.fromkeys(res))

    def encode_knowledge(self, sentence, event1_offset, event2_offset):
        sentence = sentence.strip(punctuation)
        if event1_offset[0] > event2_offset[0]:
            event1_offset, event2_offset = event2_offset, event1_offset


        split = sentence.split(' ')
        part1 = split[:event1_offset[0]]
        part2 = split[event1_offset[0]:event1_offset[1]+1]
        part3 = split[event1_offset[1]+1:event2_offset[0]]
        part4 = split[event2_offset[0]:event2_offset[1]+1]
        part5 = split[event2_offset[1]+1:]



        event1_knowledge = self.crawl_concept_net((' ').join(part2))[:self.limit]
        event2_knowledge = self.crawl_concept_net((' ').join(part4))[:self.limit]
        event1_knowledge = [x.split() for x in event1_knowledge]
        event2_knowledge = [x.split() for x in event2_knowledge]
        event1_knowledge = [x for sublist in event1_knowledge for x in sublist]
        event2_knowledge = [x for sublist in event2_knowledge for x in sublist]

        part2 += event1_knowledge
        part4 += event2_knowledge

        part1_tokens = self.tokenizer.tokenize(((' ').join(part1)))
        part2_tokens = self.tokenizer.tokenize(((' ').join(part2)))
        part3_tokens = self.tokenizer.tokenize(((' ').join(part3)))
        part4_tokens = self.tokenizer.tokenize(((' ').join(part4)))
        part5_tokens = self.tokenizer.tokenize(((' ').join(part5)))

        part1_tokens = ['[CLS]'] + part1_tokens
        part5_tokens = part5_tokens + ['[SEP]']

        part1_id = self.tokenizer.convert_tokens_to_ids(part1_tokens)
        part2_id = self.tokenizer.convert_tokens_to_ids(part2_tokens)
        part3_id = self.tokenizer.convert_tokens_to_ids(part3_tokens)
        part4_id = self.tokenizer.convert_tokens_to_ids(part4_tokens)
        part5_id = self.tokenizer.convert_tokens_to_ids(part5_tokens)

        encoded = part1_id + part2_id + part3_id + part4_id + part5_id

        xx = event1_offset[0] + 1
        xx_len = len(part2_tokens)
        event1_indexes = list(range(xx, xx + xx_len))

        yy = len(part1_id) + len(part2_id) + len(part3_id)
        yy_len = len(part4_id)

        event2_indexes = list(range(yy, yy + yy_len))

        return encoded, event1_indexes, event2_indexes