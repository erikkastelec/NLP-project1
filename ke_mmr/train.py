import torch
import pickle
import random
import numpy as np
import sys
import gc

from tqdm import tqdm
import variables as variables

from dataset import Dataset
from model import BertCausalModel
from pytorch_pretrained_bert import BertAdam

BATCH_SIZE = 3
EPOCHS = 100


class Trainer:
    def __init__(self, device='cuda', seed=1234):
        self.device = torch.device(device)
        self.model = None
        self.model_mask = None

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)


    def negative_sampling(self, data, ratio=0.7):
        result = []
        for d in data:
            if d[0][-1] == 'NULL':
                if random.random() < ratio:
                    continue
            result.append(d)
        return result


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

    def train(self, train_set, train_set_mask, test_set, test_set_mask, model_path=None, batch_size=BATCH_SIZE, epochs=EPOCHS):
        torch.cuda.empty_cache()
        gc.collect()
        train_pair = list(zip(train_set, train_set_mask))
        train_pair = self.negative_sampling(train_pair)
        train_set, train_set_mask = [d[0] for d in train_pair], [d[1] for d in train_pair]

        ###
        test_dataset = Dataset(batch_size, test_set)
        test_dataset_mask = Dataset(batch_size, test_set_mask)

        test_dataset_batch = [batch for batch in test_dataset.reader(self.device, False)]
        test_dataset_mask_batch = [batch for batch in test_dataset_mask.reader(self.device, False)]

        test_dataset_mix = list(zip(test_dataset_batch, test_dataset_mask_batch))

        ###
        train_dataset = Dataset(batch_size, train_set)
        train_dataset_mask = Dataset(batch_size, train_set_mask)

        train_dataset_batch = [batch for batch in train_dataset.reader(self.device, False)]
        train_dataset_mask_batch = [batch for batch in train_dataset_mask.reader(self.device, False)]

        train_dataset_mix = list(zip(train_dataset_batch, train_dataset_mask_batch))

        model = BertCausalModel(3).to(self.device)
        model_mask = BertCausalModel(3).to(self.device)

        learning_rate = 1e-5
        optimizer = BertAdam(model.parameters(), lr=learning_rate)
        optimizer_mask = BertAdam(model_mask.parameters(), lr=learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

        for epoch in tqdm(range(epochs), total=epochs, file=sys.stdout, desc='Epoch'):
            for batch, batch_mask in train_dataset_mix:
                model.train()
                model_mask.train()
                sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y, _ = batch
                opt = model.forward_logits(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2,
                                                event2_mask)

                sentences_s_mask, mask_s_mask, sentences_t_mask, mask_t_mask, event1_mask, event1_mask_mask, event2_mask, event2_mask_mask, data_y_mask, _ = batch_mask
                opt_mask = model_mask.forward_logits(sentences_s_mask, mask_s_mask, sentences_t_mask, mask_t_mask,
                                                          event1_mask, event1_mask_mask, event2_mask, event2_mask_mask)

                opt_mix = torch.cat([opt, opt_mask], dim=-1)
                logits = model.additional_fc(opt_mix)
                loss = loss_fn(logits, data_y)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(model_mask.parameters(), 1)

                optimizer.zero_grad()
                optimizer_mask.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_mask.step()
                #torch.cuda.empty_cache()

        model.eval()
        model_mask.eval()
        self.model = model
        self.model_mask = model_mask

        if model_path is not None:
            torch.save(model, model_path + '_full.pt')
            torch.save(model_mask, model_path + '_mask_full.pt')
            torch.save(model.state_dict(), model_path + '.pt')
            torch.save(model_mask.state_dict(), model_path + '_mask.pt')

        with torch.no_grad():
            predicted_all = []
            gold_all = []
            for batch, batch_mask in test_dataset_mix:
                sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y, _ = batch
                opt = model.forward_logits(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2,
                                           event2_mask)

                sentences_s_mask, mask_s_mask, sentences_t_mask, mask_t_mask, event1_mask, event1_mask_mask, event2_mask, event2_mask_mask, data_y_mask, _ = batch_mask
                opt_mask = model_mask.forward_logits(sentences_s_mask, mask_s_mask, sentences_t_mask, mask_t_mask,
                                                     event1_mask, event1_mask_mask, event2_mask, event2_mask_mask)

                opt_mix = torch.cat([opt, opt_mask], dim=-1)
                logits = model.additional_fc(opt_mix)

                predicted = torch.argmax(logits, -1)
                predicted = list(predicted.cpu().numpy())
                predicted_all += predicted

                gold = list(data_y.cpu().numpy())
                gold_all += gold
            precision, recall, f1 = self.compute_f1(gold_all, predicted_all)
        return (precision, recall, f1), model, model_mask



if __name__ == '__main__':
    with open('data/causaltb_train.pickle', 'rb') as f:
        train_set = pickle.load(f)

    with open('data/causaltb_test.pickle', 'rb') as f:
        test_set = pickle.load(f)

    with open('data/causaltb_train_mask.pickle', 'rb') as f:
        train_set_mask = pickle.load(f)

    with open('data/causaltb_test_mask.pickle', 'rb') as f:
        test_set_mask = pickle.load(f)


    print('Train', len(train_set), 'Test', len(test_set))

    tt = Trainer()
    (p, r, f1), _, _ = tt.train(train_set, train_set_mask, test_set, test_set_mask, model_path=None, batch_size=BATCH_SIZE, epochs=1)
    print('1 Epoch: ', p, r, f1)
    (p, r, f1), _, _ = tt.train(train_set, train_set_mask, test_set, test_set_mask, model_path=None,
                                batch_size=BATCH_SIZE, epochs=10)
    print('10 Epoch: ', p, r, f1)
    (p, r, f1), _, _ = tt.train(train_set, train_set_mask, test_set, test_set_mask, model_path=None,
                                batch_size=BATCH_SIZE, epochs=100)
    print('100 Epoch: ', p, r, f1)

