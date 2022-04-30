from transformers import BertTokenizer
from string import punctuation
class MentionMaskingReasoner:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


    def mask_sentence(self, sentence, event1_offset, event2_offset):
        sentence = sentence.strip(punctuation)
        if event1_offset[0] > event2_offset[0]:
            event1_offset, event2_offset = event2_offset, event1_offset
        split = sentence.split(' ')
        part1 = split[:event1_offset[0]]
        part2 = split[event1_offset[0]:event1_offset[1]+1]
        part3 = split[event1_offset[1]+1:event2_offset[0]]
        part4 = split[event2_offset[0]:event2_offset[1]+1]
        part5 = split[event2_offset[1]+1:]




        part1_tokens = self.tokenizer.tokenize(((' ').join(part1)))
        part2_tokens = self.tokenizer.tokenize(((' ').join(part2)))
        part3_tokens = self.tokenizer.tokenize(((' ').join(part3)))
        part4_tokens = self.tokenizer.tokenize(((' ').join(part4)))
        part5_tokens = self.tokenizer.tokenize(((' ').join(part5)))

        part1_tokens = ['[CLS]'] + part1_tokens
        part2_tokens = ['[MASK]' for x in part2_tokens]
        part4_tokens = ['[MASK]' for x in part4_tokens]
        part5_tokens = part5_tokens + ['[SEP]']



        part1_id = self.tokenizer.convert_tokens_to_ids(part1_tokens)
        part2_id = self.tokenizer.convert_tokens_to_ids(part2_tokens)
        part3_id = self.tokenizer.convert_tokens_to_ids(part3_tokens)
        part4_id = self.tokenizer.convert_tokens_to_ids(part4_tokens)
        part5_id = self.tokenizer.convert_tokens_to_ids(part5_tokens)

        tokens = self.tokenizer.tokenize(sentence)

        encoded = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
        masked_encoded = part1_id + part2_id + part3_id + part4_id + part5_id

        xx = event1_offset[0] + 1
        xx_len = len(part2_tokens)
        event1_indexes = list(range(xx, xx+xx_len))

        yy = len(part1_id) + len(part2_id) + len(part3_id)
        yy_len = len(part4_id)

        event2_indexes = list(range(yy, yy+yy_len))

        return masked_encoded, encoded, event1_indexes, event2_indexes

