from transformers import BertTokenizer


class MentionMaskingReasoner:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    def mask_sentence(self, sentence, event1_offset, event2_offset):
        """
        Masking sentence with event1 and event2 offsets.
        :param sentence: Sentence to mask.
        :param event1_offset: (starting_token, last_token) tuple of token indexes ref. to event1.
        :param event2_offset: (starting_token, last_token) tuple of token indexes ref. to event2.

        :return: Encoded representation of masked sentence.
        """

        tokens = self.tokenizer.tokenize(sentence)

        if event1_offset[0] > event2_offset[0]:
            event1_offset, event2_offset = event2_offset, event1_offset

        masked = ['[CLS]'] + tokens[:event1_offset[0]] + ['[MASK]'] + tokens[event1_offset[1]+1:event2_offset[0]] + ['[MASK]'] + tokens[event2_offset[1]+1:] + ['[SEP]']

        ids = [event1_offset[0]+1, event2_offset[0] + 1 - (event1_offset[1] - event1_offset[0])]

        masked_encoded = self.tokenizer.convert_tokens_to_ids(masked)
        return masked_encoded, (ids[0], ids[0]), (ids[1], ids[1])

    def mask_sentence2(self, sentence, event1_offset, event2_offset):
        """
        Masking sentence with event1 and event2 offsets.
        :param sentence: Sentence to mask.
        :param event1_offset: (starting_token, last_token) tuple of token indexes ref. to event1.
        :param event2_offset: (starting_token, last_token) tuple of token indexes ref. to event2.
        """

        tokens = self.tokenizer.tokenize(sentence)

        if event1_offset[0] > event2_offset[0]:
            event1_offset, event2_offset = event2_offset, event1_offset

        for i in range(event1_offset[0], event1_offset[1]+1):
            tokens[i] = '[MASK]'

        for i in range(event2_offset[0], event2_offset[1]+1):
            tokens[i] = '[MASK]'

        masked = ['[CLS]'] + tokens + ['[SEP]']

        masked_encoded = self.tokenizer.convert_tokens_to_ids(masked)

        return masked_encoded, (event1_offset[0]+1, event1_offset[1]+1), (event2_offset[0]+1, event2_offset[1]+1)
