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
        return res

    def encode_knowledge(self, sentence, event1_offset, event2_offset):
        sentence = sentence.strip(punctuation)
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        event1_offset = (event1_offset[0] + 1, event1_offset[1] + 1)
        event2_offset = (event2_offset[0] + 1, event2_offset[1] + 1)

        event1_tokens = tokens[event1_offset[0]:event1_offset[1] + 1]
        event1_words = ' '.join(event1_tokens)

        event2_tokens = tokens[event2_offset[0]:event2_offset[1] + 1]
        event2_words = ' '.join(event2_tokens)

        event1_knowledge = ' '.join(self.crawl_concept_net(event1_words)[:self.limit])
        event2_knowledge = ' '.join(self.crawl_concept_net(event2_words)[:self.limit])

        xx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(event1_knowledge))
        xx_len = len(xx)
        yy = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(event2_knowledge))
        yy_len = len(yy)

        e1_start, e1_end = event1_offset[0], event1_offset[1]
        e2_start, e2_end = event2_offset[0], event2_offset[1]

        enc_sentence = self.tokenizer.convert_tokens_to_ids(tokens)

        if e1_start < e2_start:
            temp1 = enc_sentence[:e1_end + 1] + xx + enc_sentence[e1_end + 1:]  # add e1 knowledge
            e1_end += xx_len

            e2_start += xx_len
            e2_end += xx_len

            temp2 = temp1[:e2_end + 1] + yy + temp1[e2_end + 1:]  # add e2 knowledge
            e2_end += yy_len
        else:

            temp1 = enc_sentence[:e1_end + 1] + xx + enc_sentence[e1_end + 1:]  # add e1 knowledge
            e1_end += xx_len

            temp2 = temp1[:e2_end + 1] + yy + temp1[e2_end + 1:]  # add e2 knowledge
            e2_end += yy_len

            e1_start += yy_len
            e1_end += yy_len

        return temp2, list(range(e1_start, e1_end+1)), list(range(e2_start, e2_end+1))



