import argparse
from random import randint
import random
from sklearn.neighbors import NearestNeighbors

import cnn_pretrain
from debug_samples import get_text
import ntp
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu", default = 0, type = int, required = False)

    parser.add_argument(
        "--seed", default = 83432534, type = int, required = False)

    parser.add_argument(
        "--batch-size", default = 64, type = int, required = False)

    parser.add_argument(
        "--trained-lm", default = None, required = True, type = str)

    parser.add_argument(
        "--trained-clf", default = None, required = True, type = str)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu > -1:
        torch.cuda.manual_seed(args.seed)

    valid_dataset = cnn_pretrain.datasets.imdb.get_dataset(part = "valid")
    vocab = cnn_pretrain.datasets.imdb.get_vocab()

    valid_dataset.batch_size = args.batch_size
    valid_dataset.gpu = args.gpu

    print ('Loading trained language model {} . . . '.format(args.trained_lm))
    trained_lm = torch.load(args.trained_lm)

    if args.gpu > -1:
        trained_lm.cuda(args.gpu)

    print ('Loading trained classifier {} . . . '.format(args.trained_clf))
    trained_clf = torch.load(args.trained_clf)

    if args.gpu > -1:
        trained_clf.cuda(args.gpu)

    trained_lm.eval()
    trained_clf.eval()

    lm_input_module = trained_lm.input_embedding
    lm_encoder_module = trained_lm.cnn

    reps = []
    y_preds = []
    y_true = []
    texts = []
    for batch in valid_dataset.iter_batch():
        sequence, length = batch.inputs
        texts.extend(get_text(batch.inputs.sequence.data, vocab))
        encoder_input = lm_input_module(sequence)
        encoded_input = lm_encoder_module.encoder_state_output(
            encoder_input, length = length)
        preds = trained_clf(batch.inputs)
        y_preds.extend(preds.data.tolist())
        y_true.extend(batch.targets.data.tolist())
        reps.extend(encoded_input.data.tolist())

    # print ('len(reps): ', len(reps))
    reps = np.asarray(reps)
    y_true = np.asarray(y_true)
    y_preds = np.asarray(y_preds)

    incorrect_clfs = [i for i, v in enumerate(zip(y_true, y_preds)) if v[0] != np.argmax(v[1])]

    nbrs = NearestNeighbors(n_neighbors = 5, algorithm = 'ball_tree', metric = 'l2').fit(reps)
    distances, indices = nbrs.kneighbors(reps)

    cont = True
    while(cont):
        print('-' * 30)
        rv = incorrect_clfs[randint(0, len(incorrect_clfs))]
        print("Example:", rv)
        print("Text:", texts[rv])
        print ('y_true:', y_true[rv])
        print ('y_pred:', np.argmax(y_preds[rv]))
        print ('Nearest neighbors:', indices[rv])
        print ('Nearest neighbor distances:', distances[rv])
        for ind in indices[rv]:
                print('NN:', ind)
                print('Text:', texts[ind])
        nn_true = [y_true[ii] for ii in indices[rv]]
        print ('NN true labels:', nn_true)
        nn_preds = [np.argmax(y_preds[ii]) for ii in indices[rv]]
        print ('NN pred labels:', nn_preds)
        y_n = input("Continue: y or n ?")
        if y_n == 'y':
            cont = True
        else:
            cont = False

if __name__ == "__main__":
    main()
