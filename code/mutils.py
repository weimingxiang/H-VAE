import datetime
import argparse
import logging
import torch
import xlrd
import os

import torch.nn as nn
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch.nn.functional as F

from sklearn.metrics import roc_curve, precision_recall_curve, auc


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


# -------------------------------------
# Common Used

def setup_seed(seed):
    pl.utilities.seed.seed_everything(2021)


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        now_time = str(datetime.datetime.now()).replace(" ", ":")
        log_file = os.path.join(
            args.save_path or args.init_checkpoint, now_time + 'train' + args.model + '.log')
    else:
        log_file = os.path.join(
            args.save_path or args.init_checkpoint, now_time + 'test' + args.model + '.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def batch2one(Z, y, z, class_num):
    for i in range(y.shape[0]):
        # Z[label][0] should be deleted..
        Z[y[i]] = torch.cat((Z[y[i]], z[i].cpu()), dim=0)
    return Z

# -------------------------------------
# Load funcs


def load_class(base_path):
    print("Load DeepCrispr dataset")
    InputFile = xlrd.open_workbook(os.path.join(
        base_path, 'offtarget_data/Classification/off_data_class1.xlsx'))
    sheet_all = InputFile.sheet_by_name("All_data")
    sgRNA_ids = sheet_all.col_values((0))
    sgRNA_list = sheet_all.col_values(1)
    DNA_list = sheet_all.col_values(2)
    labels_list = sheet_all.col_values(3)

    X_items = []
    for i in range(len(sgRNA_list)):
        X_items.append(np.array([sgRNA_list[i], DNA_list[i]]))

    return np.array(sgRNA_ids), np.array(X_items), np.array(labels_list, dtype=np.int32)


def load_datasetI1(base_path):
    print("Loading CIRCLE-seq dataset (dataset I/1)...")
    circle_data = pd.read_csv(os.path.join(base_path,
                                           "offtarget_data/CrisprNet_data/Dataset I (indel&mismatch)/dataset I-1/CIRCLE_seq_10gRNA_wholeDataset.csv"))
    sgRNA_ids = []
    on_seqs = []
    off_seqs = []
    sgrna_types = []
    label_seqs = []
    for idx, row in circle_data.iterrows():
        on_seq = row['sgRNA_seq']
        off_seq = row['off_seq']
        label = int(row['label'])
        read_val = row['Read']
        sgrna_type = row["sgRNA_type"]
        on_seqs.append(on_seq)
        off_seqs.append(off_seq)
        label_seqs.append(label)
        sgrna_types.append(sgrna_type)

    sgrna_set = []
    for idx, sgrna in enumerate(sgrna_types):
        if sgrna in sgrna_set:
            index = sgrna_set.index(sgrna)
            sgRNA_ids.append("sg" + str(index))
        else:
            sgrna_set.append(sgrna)
            index = sgrna_set.index(sgrna)
            sgRNA_ids.append("sg" + str(index))

    # 对sgRNA type进行编号

    X_items = []
    for i in range(len(sgRNA_ids)):
        X_items.append(np.array([on_seqs[i], off_seqs[i]]))
    print("Loading Complete")
    return np.array(sgRNA_ids), np.array(X_items), np.array(label_seqs), len(sgrna_set)


def load_datasetI2(base_path):
    print("Encoding GUIDE-seq dataset (dataset I/1)...")
    circle_data = pd.read_csv(os.path.join(base_path,
                                           "offtarget_data/CrisprNet_data/Dataset I (indel&mismatch)/dataset I-2/elevation_6gRNA_wholeDataset.csv"))
    on_seqs = []
    off_seqs = []
    label_seqs = []
    for idx, row in circle_data.iterrows():
        on_seq = row['crRNA']
        off_seq = row['DNA']
        read_val = row['read'] > 0
        on_seqs.append(on_seq)
        off_seqs.append(off_seq)
        if read_val:
            label_seqs.append(1)
        else:
            label_seqs.append(0)

    X_items = []
    for i in range(len(on_seqs)):
        X_items.append(np.array([on_seqs[i], off_seqs[i]]))
    print("Loading Complete")
    return np.array(X_items), np.array(label_seqs)

# -------------------------------------
# Split funcs


def Dataset_split_senerio1(data, test_ratio, seed=2020):
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices], data[test_indices]


def Dataset_split_senerio2(ids, items, labels, num):
    one = 'sg' + str(num)
    train_indices = []
    test_indices = []
    for i in range(len(ids)):
        if ids[i] == one:
            test_indices.append(i)
        else:
            train_indices.append(i)
    return items[train_indices], labels[train_indices], items[test_indices], labels[test_indices]


def Dataset_split_senerio3(ids, items, labels, seed):
    np.random.seed(seed)
    all_id = list(range(30))
    all_id.remove(25)
    all_id = np.array(all_id)
    three = np.random.choice(all_id, size=3, replace=False)
    selected_id = all_id[three].tolist()
    selected_id = ['sg' + str(num) for num in selected_id]
    train_indices = []
    test_indices = []
    for i in range(len(ids)):
        if ids[i] in selected_id:
            test_indices.append(i)
        else:
            train_indices.append(i)
    return items[train_indices], labels[train_indices], items[test_indices], labels[test_indices]

# -------------------------------------
# Datasets


class TrainDataset(Dataset):

    def __init__(self, items, labels):

        self.items, self.labels = items, labels
        self.labels = torch.from_numpy(self.labels)
        self.len = len(self.items)
        self.pos_index, self.neg_index = self._cal_nums()
        self.MATCH_ROW_NUMBER1 = {"AA": 1, "AC": 2, "AG": 3, "AT": 4, "CA": 5, "CC": 6, "CG": 7, "CT": 8, "GA": 9,
                                  "GC": 10, "GG": 11, "GT": 12, "TA": 13, "TC": 14, "TG": 15, "TT": 16}
        self.MATCH_ROW_NUMBER2 = {"A": 1, "G": 2, "C": 3, "T": 4}

    def __getitem__(self, index):

        return self._encode(self.items[index]), self.labels[index]

    def __len__(self):

        return self.len

    def _encode(self, arr):
        result_arr = []
        result_arr1 = []
        result_arr2 = []
        for i in range(len(arr[0])):
            on_t = arr[0][i]
            off_t = arr[1][i]
            if on_t == 'N' or on_t == 'n':
                on_t = off_t
            temp_str = on_t + off_t
            result_arr.append(self.MATCH_ROW_NUMBER1[temp_str] - 1)
            result_arr1.append(self.MATCH_ROW_NUMBER2[on_t] - 1)
            result_arr2.append(self.MATCH_ROW_NUMBER2[off_t] - 1)

        return [torch.LongTensor(result_arr), torch.LongTensor(result_arr1), torch.LongTensor(result_arr2)]

    def _cal_nums(self):
        pos_index = []
        neg_index = []
        for i, label in enumerate(self.labels):
            if label == 1:
                pos_index.append(i)
            else:
                neg_index.append(i)
        return pos_index, neg_index


class TrainDataset_local(Dataset):

    def __init__(self, items, labels):

        self.items, self.labels = items, labels
        self.len = len(self.items)
        self.pos_index, self.neg_index = self._cal_nums()
        self.MATCH_ROW_NUMBER1 = {"AA": 1, "AC": 2, "AG": 3, "AT": 4, "CA": 5, "CC": 6, "CG": 7, "CT": 8, "GA": 9,
                                  "GC": 10, "GG": 11, "GT": 12, "TA": 13, "TC": 14, "TG": 15, "TT": 16}
        self.MATCH_ROW_NUMBER2 = {"A": 1, "G": 2, "C": 3, "T": 4}

    def __getitem__(self, index):

        return self._encode(self.items[index]), self.labels[index]

    def __len__(self):

        return self.len

    def _encode(self, arr):
        result_arr = []
        result_arr1 = []
        result_arr2 = []
        for i in range(len(arr[0])):
            on_t = arr[0][i]
            off_t = arr[1][i]
            if on_t == 'N' or on_t == 'n':
                on_t = off_t
            temp_str = on_t + off_t
            result_arr.append(self.MATCH_ROW_NUMBER1[temp_str] - 1)
            result_arr1.append(self.MATCH_ROW_NUMBER2[on_t] - 1)
            result_arr2.append(self.MATCH_ROW_NUMBER2[off_t] - 1)

        return [torch.LongTensor(result_arr), torch.LongTensor(result_arr1), torch.LongTensor(result_arr2)]

    def _cal_nums(self):
        pos_index = []
        neg_index = []
        for i, label in enumerate(self.labels):
            if label == 1:
                pos_index.append(i)
            else:
                neg_index.append(i)
        return pos_index, neg_index


class TrainDataset_indels(Dataset):
    def __init__(self, items, labels):

        self.items, self.labels = items, labels
        self.labels = torch.from_numpy(self.labels)
        self.len = len(self.items)
        self.pos_index, self.neg_index = self._cal_nums()
        self.MATCH_ROW_NUMBER1 = {"AA": 1, "AC": 2, "AG": 3, "AT": 4, "CA": 5, "CC": 6, "CG": 7, "CT": 8, "GA": 9,
                                  "GC": 10, "GG": 11, "GT": 12, "TA": 13, "TC": 14, "TG": 15, "TT": 16, "--": 17,
                                  "A_": 18, "C_": 19, "G_": 20, "T_": 21, "_A": 22, "_C": 23, "_G": 24, "_T": 25}
        self.MATCH_ROW_NUMBER2 = {
            "-": 1, "A": 2, "G": 3, "C": 4, "T": 5, "_": 6}

    def __getitem__(self, index):

        return self._encode(self.items[index]), self.labels[index]

    def __len__(self):

        return self.len

    def _encode(self, arr):
        result_arr = []
        result_arr1 = []
        result_arr2 = []
        for i in range(len(arr[0])):
            on_t = arr[0][i]
            off_t = arr[1][i]
            if on_t == 'N' or on_t == 'n':
                on_t = off_t
            temp_str = on_t + off_t
            result_arr.append(self.MATCH_ROW_NUMBER1[temp_str] - 1)
            result_arr1.append(self.MATCH_ROW_NUMBER2[on_t] - 1)
            result_arr2.append(self.MATCH_ROW_NUMBER2[off_t] - 1)

        return [torch.LongTensor(result_arr), torch.LongTensor(result_arr1), torch.LongTensor(result_arr2)]

    def _cal_nums(self):
        pos_index = []
        neg_index = []
        for i, label in enumerate(self.labels):
            if label == 1:
                pos_index.append(i)
            else:
                neg_index.append(i)
        return pos_index, neg_index


class TestDataset(Dataset):

    def __init__(self, items, labels):

        self.items, self.labels = items, labels
        self.len = len(self.items)
        self.MATCH_ROW_NUMBER1 = {"AA": 1, "AC": 2, "AG": 3, "AT": 4, "CA": 5, "CC": 6, "CG": 7, "CT": 8, "GA": 9,
                                  "GC": 10, "GG": 11, "GT": 12, "TA": 13, "TC": 14, "TG": 15, "TT": 16}
        self.MATCH_ROW_NUMBER2 = {"A": 1, "G": 2, "C": 3, "T": 4}

    def __getitem__(self, index):

        return self._encode(self.items[index]), self.labels[index]

    def __len__(self):

        return self.len

    def _encode(self, arr):
        result_arr = []
        result_arr1 = []
        result_arr2 = []
        for i in range(len(arr[0])):
            on_t = arr[0][i]
            off_t = arr[1][i]
            if on_t == 'N' or on_t == 'n':
                on_t = off_t
            temp_str = on_t + off_t
            result_arr.append(self.MATCH_ROW_NUMBER1[temp_str] - 1)
            result_arr1.append(self.MATCH_ROW_NUMBER2[on_t] - 1)
            result_arr2.append(self.MATCH_ROW_NUMBER2[off_t] - 1)

        return [torch.LongTensor(result_arr), torch.LongTensor(result_arr1), torch.LongTensor(result_arr2)]


class TestDataset_indels(Dataset):
    def __init__(self, items, labels):

        self.items, self.labels = items, labels
        self.labels = torch.from_numpy(self.labels)
        self.len = len(self.items)
        #  self.pos_index, self.neg_index = self._cal_nums()
        self.MATCH_ROW_NUMBER1 = {"AA": 1, "AC": 2, "AG": 3, "AT": 4, "CA": 5, "CC": 6, "CG": 7, "CT": 8, "GA": 9,
                                  "GC": 10, "GG": 11, "GT": 12, "TA": 13, "TC": 14, "TG": 15, "TT": 16, "--": 17,
                                  "A_": 18, "C_": 19, "G_": 20, "T_": 21, "_A": 22, "_C": 23, "_G": 24, "_T": 25}
        self.MATCH_ROW_NUMBER2 = {
            "-": 1, "A": 2, "G": 3, "C": 4, "T": 5, "_": 6}

    def __getitem__(self, index):

        return self._encode(self.items[index]), self.labels[index]

    def __len__(self):

        return self.len

    def _encode(self, arr):
        result_arr = []
        result_arr1 = []
        result_arr2 = []
        for i in range(len(arr[0])):
            on_t = arr[0][i]
            off_t = arr[1][i]
            if on_t == 'N' or on_t == 'n':
                on_t = off_t
            temp_str = on_t + off_t
            result_arr.append(self.MATCH_ROW_NUMBER1[temp_str] - 1)
            result_arr1.append(self.MATCH_ROW_NUMBER2[on_t] - 1)
            result_arr2.append(self.MATCH_ROW_NUMBER2[off_t] - 1)

        return [torch.LongTensor(result_arr), torch.LongTensor(result_arr1), torch.LongTensor(result_arr2)]


# -------------------------------------
# Others
def GetKS(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_value = auc(fpr, tpr)
    ks = max(tpr - fpr)
    pre, recall, thresholds1 = precision_recall_curve(y_true, y_pred_prob)
    prc_value = auc(recall, pre)
    return roc_value, prc_value, ks


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' %
                     (mode, metric, step, metrics[metric]))

# -------------------------------------
# Train and test


def train_step(model, optimizer, train_iterator, args):
    '''
    A single train step. Apply back-propation and return the loss
    '''

    model.train()
    optimizer.zero_grad()

    x, y = next(train_iterator)

    if args.cuda:
        x = (x[0].cuda(), x[1].cuda(), x[2].cuda())
        y = y.cuda()

    loss = model.loss(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optimizer.step()

    log = {
        'loss': loss.item()
    }

    return log


def test_step(model, test_dataloader, args):
    '''
    Evaluate the model on test or valid datasets
    '''
    model.eval()
    # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
    # Prepare dataloader for evaluation

    logs = {}
    predict_all = np.array([], dtype=float)
    labels_all = np.array([], dtype=float)

    step = 0
    total_steps = len(test_dataloader)

    with torch.no_grad():
        for x, y in test_dataloader:
            if args.cuda:
                x = (x[0].cuda(), x[1].cuda(), x[2].cuda())
                y = y.cuda()

            preds = model(x)
            labels = y.data.cpu().numpy()
            predic = preds[:, 1].data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

            if step % args.test_log_steps == 0:
                logging.info('Evaluating the model... (%d/%d)' %
                             (step, total_steps))
            step += 1

        roc_value, prc_value, ks = GetKS(labels_all, predict_all)
        logs['ROC-AUC'] = roc_value
        logs['PR-AUC'] = prc_value

    return logs
