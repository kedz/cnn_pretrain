import os
import argparse
import random

import torch
import ntp
import cnn_pretrain
import numpy as np
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu", default=-1, type=int, required=False)
    parser.add_argument(
        "--epochs", default=25, type=int, required=False)
    parser.add_argument(
        "--seed", default=83432534, type=int, required=False)
    parser.add_argument(
        "--positive-samples", type=int, default=10, required=False)
    parser.add_argument(
        "--noise-samples", type=int, default=10, required=False)

    parser.add_argument(
        "--lr", required=False, default=.0001, type=float)
    parser.add_argument(
        "--batch-size", default=8, type=int, required=False)
    parser.add_argument(
        "--embedding-size", type=int, required=False, default=300)

    ### Dropout settings ###
    parser.add_argument(
        "--word-dropout", required=False, default=.05, type=float)
    parser.add_argument(
        "--embedding-dropout", required=False, default=.1, type=float)
    parser.add_argument(
        "--conv-dropout", required=False, default=.25, type=float)

    ### Conv settings ###
    parser.add_argument(
        "--filter_widths", default=[3, 4, 5], type=int, nargs="+")
    parser.add_argument(
        "--num_filters", default=300, type=int, required=False)

    parser.add_argument(
        "--save-model", default=None, type=str)

    args = parser.parse_args()

    if args.save_model is None:
        ts = int((datetime.now() - datetime(1970,1,1)).total_seconds())
        args.save_model = os.path.join(
            cnn_pretrain.datasets.imdb.get_models_path(),
            "cnn_nce.{}.bin".format(ts))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu > -1:
        torch.cuda.manual_seed(args.seed)

    dataset = cnn_pretrain.datasets.imdb.get_dataset(part="unlabeled")
    dataset.batch_size = args.batch_size
    dataset.shuffle = True
    
    if args.gpu > -1:
        dataset.gpu = args.gpu

    freqs = cnn_pretrain.util.get_index_frequency(
        dataset.inputs.sequence, mask_value=0)
    freqs = freqs.float() ** .75
    vocab = cnn_pretrain.datasets.imdb.get_vocab()

    sampler = cnn_pretrain.dataio.ContextSampler(
        dataset, 
        freqs,
        num_positive=args.positive_samples,
        num_negative=args.noise_samples,
        pos_inv_freq=True)

    # TODO valid datset nans for some reason.
    valid_dataset = cnn_pretrain.datasets.imdb.get_dataset(part="valid")
    valid_dataset.batch_size = args.batch_size
    if args.gpu > -1:
        valid_dataset.gpu = args.gpu

    valid_sampler = cnn_pretrain.dataio.ContextSampler(
        valid_dataset, freqs, num_negative=args.noise_samples)
    
    input_embedding = ntp.modules.Embedding(
        vocab.size, args.embedding_size, 
        input_dropout=args.word_dropout, dropout=args.embedding_dropout,
        transpose=False)

    cnn = ntp.modules.EncoderCNN(
        args.embedding_size, args.filter_widths, args.num_filters, 
        dropout=args.conv_dropout,
        activation="relu")

    sample_embedding = ntp.modules.Embedding(
        vocab.size, cnn.output_size, 
        input_dropout=0, dropout=args.embedding_dropout,
        transpose=False)

    model = cnn_pretrain.models.CNNNCE(
        input_embedding, cnn, sample_embedding, args.noise_samples)
        
    if args.gpu > -1:
        model.cuda(args.gpu)

    crit = ntp.criterion.BinaryCrossEntropy(mode="logit", mask_value=-1)
    opt = ntp.optimizer.Adam(model.parameters(), lr=args.lr)

    ntp.trainer.optimize_criterion(crit, model, opt, sampler,
                                   max_epochs=args.epochs,
                                   validation_data=sampler,
                                   save_model=args.save_model)

    print("Best model saved to {}".format(args.save_model))

if __name__ == "__main__":
    main()
