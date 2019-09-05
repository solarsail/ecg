from __future__ import print_function

import argparse
import numpy as np
import keras
import os

import load

LABELS = [u'A', u'B', u'F', u'N', u'R', u'V', u'j', u'n', u'~']

def predict(data_file, model_path):
    ecg = load.load_ecg(data_file)
    preproc = load.Preproc(ecg, [])
    x = preproc.process_x([ecg])
    #print(x.shape)

    model = keras.models.load_model(model_path)
    probs = model.predict(x, verbose=1)
    #print(probs.shape)

    return probs[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="path to binary data file (in mV)")
    parser.add_argument("model_path", help="path to model")
    args = parser.parse_args()
    prob = predict(args.data_file, args.model_path)
    res = []
    np.savetxt('sc_predict.csv', prob, '%10.5f', ',')
    for j in range(len(prob)):
        mi = np.argmax(prob[j])
        res.append(LABELS[mi])
    np.savetxt('sc_res.csv', res, fmt='%s', delimiter=',')
