import argparse
from datetime import datetime
import os
import random

import cnn_pretrain
import ntp
import torch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu", default = 0, type = int, required = False)
    parser.add_argument(
        "--epochs", default = 25, type = int, required = False)
    parser.add_argument(
        "--seed", default = 83432534, type = int, required = False)

    parser.add_argument(
        "--lr", required = False, default = .0001, type = float)
    parser.add_argument(
        "--batch-size", default = 64, type = int, required = False)
    parser.add_argument(
        "--embedding-size", type = int, required = False, default = 300)

    ### Dropout settings ###
    parser.add_argument(
        "--word-dropout", required = False, default = .05, type = float)
    parser.add_argument(
        "--embedding-dropout", required = False, default = .2, type = float)
    parser.add_argument(
        "--conv-dropout", required = False, default = .5, type = float)
    parser.add_argument(
        "--mlp-dropout", required = False, default = .5, type = float)

    ### Conv settings ###
    parser.add_argument(
        "--filter_widths", default = [1, 2, 3, 4, 5], type = int, nargs = "+")
    parser.add_argument(
        "--num_filters", default = 100, type = int, required = False)

    ### MLP settings ###
    parser.add_argument(
        "--hidden-layer-sizes", default = [100], type = int, nargs = "+")

    parser.add_argument(
        "--save-model", default = None, type = str)

    parser.add_argument(
        "--trained-model", default = None, type = str)

    args = parser.parse_args()

    if args.save_model is None:
        ts = int((datetime.now() - datetime(1970, 1, 1)).total_seconds())
        args.save_model = os.path.join(
            cnn_pretrain.datasets.imdb.get_models_path(),
            "cnn.{}.bin".format(ts))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu > -1:
        torch.cuda.manual_seed(args.seed)

    train_dataset = cnn_pretrain.datasets.imdb.get_dataset(part = "train")
    valid_dataset = cnn_pretrain.datasets.imdb.get_dataset(part = "valid")
    vocab = cnn_pretrain.datasets.imdb.get_vocab()

    train_dataset.batch_size = args.batch_size
    train_dataset.gpu = args.gpu
    train_dataset.shuffle = True
    valid_dataset.batch_size = args.batch_size
    valid_dataset.gpu = args.gpu

    trained_model = None
    if args.trained_model is not None:
        print ('Loading trained model {} . . . '.format(args.trained_model))
        trained_model = torch.load(args.trained_model)

    if trained_model is not None:
        input_module = trained_model.input_embedding
    else:
        input_module = ntp.modules.Embedding(
            vocab.size, args.embedding_size,
            input_dropout = args.word_dropout, dropout = args.embedding_dropout,
            transpose = False)
        print ('Loading word2vec  . . .')
        input_module.use_pretrained("word2vec", vocab)

    if trained_model is not None:
        encoder_module = trained_model.cnn
    else:
        encoder_module = ntp.modules.EncoderCNN(
            args.embedding_size, args.filter_widths, args.num_filters,
            dropout = args.conv_dropout,
            activation = "relu")

    predictor_module = ntp.modules.MultiLayerPerceptron(
        encoder_module.output_size, 2, output_activation = None,
        hidden_sizes = args.hidden_layer_sizes,
        hidden_layer_activations = "relu",
        hidden_layer_dropout = args.mlp_dropout)

    model = ntp.models.SequenceClassifier(
        input_module, encoder_module, predictor_module)

    if args.gpu > -1:
        model.cuda(args.gpu)

    crit = ntp.criterion.MultiClassCrossEntropy()
    crit.add_reporter(ntp.criterion.MultiClassAccuracyReporter())
    opt = ntp.optimizer.Adam(model.parameters(), lr = args.lr)

    ntp.trainer.optimize_criterion(crit, model, opt, train_dataset,
                                   validation_data = valid_dataset,
                                   max_epochs = args.epochs,
                                   save_model = args.save_model)

    print("Best model saved to {}".format(args.save_model))


if __name__ == "__main__":
    main()
