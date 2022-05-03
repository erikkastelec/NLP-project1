import torch
import pickle
import random
import numpy as np
from transformers import BertTokenizer
import sys

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
        self.model.eval()
        self.model.to(device)

        self.model_mask = BertCausalModel(3)
        self.model_mask.load_state_dict(torch.load(model_mask_path))
        self.model_mask.eval()
        self.model_mask.to(device)

        self.device = torch.device(device)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.knowledge_encoder = KnowledgeEncoder(self.tokenizer, knowledge_limit)
        self.masker = MentionMaskingReasoner(self.tokenizer)

    def compute_f1(self, gold, predicted):
        c_predict = 0
        c_correct = 0
        c_gold = 0

        for g, p in zip(gold, predicted):
            if g != 0:
                c_gold += 1
            if p != 0:
                c_predict += 1
            if g != 0 and p != 0:
                c_correct += 1

        p = c_correct / (c_predict + 1e-100)
        r = c_correct / c_gold
        f = 2 * p * r / (p + r + 1e-100)

        print('correct', c_correct, 'predicted', c_predict, 'golden', c_gold)
        return p, r, f

    def transform(self, sentence, event1_index, event2_index, doc_name=None, label=None):
        emb_sen, e1i, e2i = self.knowledge_encoder.encode_knowledge(sentence, event1_index, event2_index)
        emb_kg = [doc_name, emb_sen, emb_sen, e1i, e2i, label]

        emb_sen1_mask, emb_sen2_mask, e1i, e2i = self.masker.mask_sentence(sentence, event1_index, event2_index)
        emb_mask = [doc_name, emb_sen1_mask, emb_sen2_mask, e1i, e2i, label]

        return emb_kg, emb_mask

    def transform_all(self, data):
        knowledge_list = []
        mask_list = []
        for doc_name, sentence, event1_index, event2_index, label in tqdm(data, total=len(data), file=sys.stdout):
            emb_kg, emb_mask = self.transform(sentence, event1_index, event2_index, doc_name=doc_name, label=label)
            knowledge_list.append(emb_kg)
            mask_list.append(emb_mask)

        return knowledge_list, mask_list

    def predict_inner(self, q, q_mask):
        single_ds = Dataset(1, [q])
        single_mask_ds = Dataset(1, [q_mask])
        single = [batch for batch in single_ds.reader(self.device, False)][0]
        single_mask = [batch for batch in single_mask_ds.reader(self.device, False)][0]

        with torch.no_grad():
            sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y, _ = single
            opt = self.model.forward_logits(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2,
                                       event2_mask)

            sentences_s_mask, mask_s_mask, sentences_t_mask, mask_t_mask, event1_mask, event1_mask_mask, event2_mask, event2_mask_mask, data_y_mask, _ = single_mask
            opt_mask = self.model_mask.forward_logits(sentences_s_mask, mask_s_mask, sentences_t_mask, mask_t_mask,
                                                      event1_mask, event1_mask_mask, event2_mask, event2_mask_mask)

            opt_mix = torch.cat([opt, opt_mask], dim=-1)
            logits = self.model.additional_fc(opt_mix)

            predicted = torch.argmax(logits, -1)
            predicted = list(predicted.cpu().numpy())[0]

        return predicted

    def predict(self, data, doc_name=None, label=None, need_embedding=True):
        if need_embedding:
            sentence, event1_index, event2_index = data
            emb_kg, emb_mask = self.transform(sentence, event1_index, event2_index, doc_name=doc_name, label=label)
        else:
            emb_kg, emb_mask = data
        return self.predict_inner(emb_kg, emb_mask)

    def predict_all_inner(self, test_set, test_set_mask, batch_size=7):
        dataset = Dataset(7, test_set)
        dataset_mask = Dataset(7, test_set_mask)

        dataset_batch = [batch for batch in dataset.reader(self.device, False)]
        dataset_mask_batch = [batch for batch in dataset_mask.reader(self.device, False)]

        dataset_mix = list(zip(dataset_batch, dataset_mask_batch))

        with torch.no_grad():
            predicted_all = []
            gold_all = []
            for batch, batch_mask in tqdm(dataset_mix, desc='Predicting', total=len(dataset_mix), file=sys.stdout):
                sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y, _ = batch
                opt = self.model.forward_logits(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2,
                                                event2_mask)

                sentences_s_mask, mask_s_mask, sentences_t_mask, mask_t_mask, event1_mask, event1_mask_mask, event2_mask, event2_mask_mask, data_y_mask, _ = batch_mask
                opt_mask = self.model_mask.forward_logits(sentences_s_mask, mask_s_mask, sentences_t_mask, mask_t_mask, event1_mask, event1_mask_mask, event2_mask, event2_mask_mask)

                opt_mix = torch.cat([opt, opt_mask], dim=-1)
                logits = self.model.additional_fc(opt_mix)

                predicted = torch.argmax(logits, -1)
                predicted = list(predicted.cpu().numpy())
                predicted_all += predicted

                gold = list(data_y.cpu().numpy())
                gold_all += gold

        return predicted_all, gold_all

    def predict_all(self, data, batch_size=7, need_embedding=True):
        if need_embedding:
            knowledge_list, mask_list = self.transform_all(data)
        else:
            knowledge_list, mask_list = data
        predicted, _ = self.predict_all_inner(knowledge_list, mask_list, batch_size=batch_size)
        return predicted

    def predict_all_eval(self, data, batch_size=7, need_embedding=True):
        if need_embedding:
            knowledge_list, mask_list = self.transform_all(data)
        else:
            knowledge_list, mask_list = data
        predicted, gold = self.predict_all_inner(knowledge_list, mask_list, batch_size=batch_size)
        p, r, f = self.compute_f1(predicted, gold)
        return predicted, gold, p, r, f


if __name__ == '__main__':
    predictor = KEMMGPredictor(MODEL_PATH, MODEL_MASK_PATH)

    sample1 = 'The patient was admitted to the hospital because of a heart attack.'  # e1 = 'admitted', e2 = 'heart attack'
    sample2 = 'The earthquake caused a tsunami.'  # e1 = 'earthquake', e2 = 'tsunami'
    sample3 = 'Both the earthquake and tsunami are natural disasters.'  # e1 = 'earthquake', e2 = 'tsunami'
    s1_e1_index = (1, 1)
    s1_e2_index = (11, 11)
    s2_e1_index = (1, 1)
    s2_e2_index = (4, 4)
    s3_e1_index = (2, 2)
    s3_e2_index = (4, 4)

    p1 = predictor.predict(sample1, s1_e1_index, s1_e2_index)
    p2 = predictor.predict(sample2, s2_e1_index, s2_e2_index)
    p3 = predictor.predict(sample3, s3_e1_index, s3_e2_index)

    print('Prediction for sample 1: ', p1)
    print('Prediction for sample 2: ', p2)
    print('Prediction for sample 3: ', p3)
