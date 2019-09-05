from __future__ import print_function

import argparse
import numpy as np
import keras
import os

import load
import util

def predict(data_json, model_path):
    preproc = util.load(os.path.dirname(model_path))
    dataset = load.load_dataset(data_json)
    x, y = preproc.process(*dataset)
    classes = preproc.classes

    model = keras.models.load_model(model_path)
    probs = model.predict(x, verbose=1)

    return probs, classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_json", help="path to data json")
    parser.add_argument("model_path", help="path to model")
    args = parser.parse_args()
    probs, classes = predict(args.data_json, args.model_path)
    for i in range(len(probs)):
        prob = probs[i]
        np.savetxt('predict_{}.csv'.format(i), prob, '%10.5f', ',')
        res = []
        for j in range(len(prob)):
            mi = np.argmax(prob[j])
            res.append(classes[mi])
        np.savetxt('res_{}.csv'.format(i), res, fmt='%s', delimiter=',')
    #print(probs)
