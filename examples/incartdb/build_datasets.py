import json
import numpy as np
import os
import random
import scipy.io as sio
import tqdm
import wfdb

STEP = 256
SLICE_LEN = 1807

def load_ecg_mat(ecg_file):
    return sio.loadmat(ecg_file)['val'].squeeze()

def load_all(data_path):
    rec_file = os.path.join(data_path, "RECORDS")
    with open(rec_file, 'r') as fid:
        records = [l.strip() for l in fid]

    dataset = []
    for record in tqdm.tqdm(records):
        ind = -1
        labels = []
        ecg_file = os.path.join(data_path, record + ".dat")
        ecg_file = os.path.abspath(ecg_file)
        ann = wfdb.rdann('data/' + record, 'atr')
        for ai in range(ann.ann_len):
            samp = ann.sample[ai]
            if samp < 0:
                continue
            label = ann.symbol[ai]
            li = samp / STEP
            while li > len(labels):
                labels.append('~')
            if li == len(labels):
                labels.append(label)
            elif li < len(labels) and label != 'N':
                #print(samp, li, len(labels))
                labels[-1] = label

        while len(labels) < SLICE_LEN:
            labels.append('~')

        dataset.append((ecg_file, labels))
    return dataset

def split(dataset, dev_frac):
    dev_cut = int(dev_frac * len(dataset))
    random.shuffle(dataset)
    dev = dataset[:dev_cut]
    train = dataset[dev_cut:]
    return train, dev

def make_json(save_path, dataset):
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {'ecg' : d[0],
                     'labels' : d[1]}
            json.dump(datum, fid)
            fid.write('\n')

if __name__ == "__main__":
    random.seed(2018)

    dev_frac = 0.1
    data_path = "data/"
    dataset = load_all(data_path)
    train, dev = split(dataset, dev_frac)
    make_json("train.json", train)
    make_json("dev.json", dev)

