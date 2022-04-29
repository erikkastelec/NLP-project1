import torch
import pickle
import random
import numpy as np
from transformers import BertTokenizer

from tqdm import tqdm
from dataset import Dataset
from model import BertCausalModel
from mention_masking_reasoner import MentionMaskingReasoner
from knowledge_encoder import KnowledgeEncoder


MODEL_PATH = 'models/bert_causal_model.pt'
MODEL_MASK_PATH = 'models/bert_causal_model_mask.pt'

class KEMMGPredictor:
    def __init__(self, model_path, model_mask_path, device='cuda', knowledge_limit=5):
        self.model = BertCausalModel(3)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)

        self.model_mask = BertCausalModel(3)
        self.model_mask.load_state_dict(torch.load(model_mask_path))
        self.model_mask.to(device)

        self.device = torch.device(device)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.knowledge_encoder = KnowledgeEncoder(self.tokenizer, knowledge_limit)
        self.masker = MentionMaskingReasoner(self.tokenizer)

    def transform(self, sentence, event1_index, event2_index, doc_name=None, label=None):
        emb_sen, e1i, e2i = self.knowledge_encoder.encode_knowledge(sentence, event1_index, event2_index)
        emb_kg = [doc_name, emb_sen, emb_sen, e1i, e2i, label]

        emb_sen1_mask, emb_sen2_mask, e1i, e2i = self.masker.mask_sentence(sentence, event1_index, event2_index)
        emb_mask = [doc_name, emb_sen1_mask, emb_sen2_mask, e1i, e2i, label]

        return emb_kg, emb_mask

    def transform_all(self, data):
        knowledge_list = []
        mask_list = []
        for doc_name, sentence, event1_index, event2_index, label in tqdm(data):
            emb_kg, emb_mask = self.transform(sentence, event1_index, event2_index, doc_name=doc_name, label=label)
            knowledge_list.append(emb_kg)
            mask_list.append(emb_mask)

        return knowledge_list, mask_list


    def predict(self, q, q_mask):
        single_ds = Dataset(1, [q])
        single_mask_ds = Dataset(1, [q_mask])
        single = [batch for batch in single_ds.reader(self.device, False)][0]
        single_mask = [batch for batch in single_mask_ds.reader(self.device, False)][0]

        with torch.no_grad():
            sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y, _ = single
            sentences_s_mask = single_mask[0]

            opt = self.model.forward_logits(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask)
            opt_mask = self.model_mask.forward_logits(sentences_s_mask, mask_s, sentences_t, mask_t, event1, event1_mask,
                                                 event2, event2_mask)

            opt_mix = torch.cat([opt, opt_mask], dim=-1)
            logits = self.model.additional_fc(opt_mix)

            predicted = torch.argmax(logits, -1)
            predicted = list(predicted.cpu().numpy())[0]

        return predicted

    def predict_all(self, test_set, test_set_mask, batch_size=7):
        dataset = Dataset(7, test_set)
        dataset_mask = Dataset(7, test_set_mask)

        dataset_batch = [batch for batch in dataset.reader(self.device, False)]
        dataset_mask_batch = [batch for batch in dataset_mask.reader(self.device, False)]

        dataset_mix = list(zip(dataset_batch, dataset_mask_batch))

        with torch.no_grad():
            predicted_all = []
            for batch, batch_mask in tqdm(dataset_mix, desc='Predicting'):
                sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y, _ = batch
                sentences_s_mask = batch_mask[0]

                opt = self.model.forward_logits(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2,
                                           event2_mask)
                opt_mask = self.model_mask.forward_logits(sentences_s_mask, mask_s, sentences_t, mask_t, event1,
                                                     event1_mask,
                                                     event2, event2_mask)

                opt_mix = torch.cat([opt, opt_mask], dim=-1)
                logits = self.model.additional_fc(opt_mix)

                predicted = torch.argmax(logits, -1)
                predicted = list(predicted.cpu().numpy())
                predicted_all += predicted

        return predicted_all

if __name__ == '__main__':
    predictor = KEMMGPredictor(MODEL_PATH, MODEL_MASK_PATH)




    print('Loading data...')
