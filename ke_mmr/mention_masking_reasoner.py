from transformers import BertTokenizer

class MentionMaskingReasoner:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


    def mask_sentence(self, sentence, event1_offset, event2_offset):
        if event1_offset[0] > event2_offset[0]:
            event1_offset, event2_offset = event2_offset, event1_offset
        split = sentence.split(' ')
        tokens = self.tokenizer.tokenize(sentence)


        sentence1 = (' ').join(split[:event1_offset[0]] + ['[MASK]'] + split[event1_offset[1]+1:])
        sentence2 = (' ').join(split[:event2_offset[0]] + ['[MASK]'] + split[event2_offset[1]+1:])
        masked1_encoded = self.tokenizer.encode(sentence1)
        masked2_encoded = self.tokenizer.encode(sentence2)

        index1 = masked1_encoded.index(103)
        index2 = masked2_encoded.index(103)
        return masked1_encoded, masked2_encoded, [index1, index1], [index2, index2]


    def mask_sentence_old(self, sentence, event1_offset, event2_offset):
        if event1_offset[0] > event2_offset[0]:
            event1_offset, event2_offset = event2_offset, event1_offset

        split = sentence.split(' ')
        sentence1 = (' ').join(split[:event1_offset[0]] + ['[MASK]'] + split[event1_offset[1]+1:])
        sentence2 = (' ').join(split[:event2_offset[0]] + ['[MASK]'] + split[event2_offset[1]+1:])
        masked1_encoded = self.tokenizer.encode(sentence1)
        masked2_encoded = self.tokenizer.encode(sentence2)

        index1 = masked1_encoded.index(103)
        index2 = masked2_encoded.index(103)
        return masked1_encoded, masked2_encoded, [index1, index1], [index2, index2]

