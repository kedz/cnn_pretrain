import os
import argparse
import random

import torch
import ntp
import cnn_pretrain
import numpy as np

import textwrap


def get_text(inputs, vocab):
    texts = []
    for i in range(inputs.size(0)):
        tokens = []
        for j in range(inputs.size(1)):
            if inputs[i, j] == 0:
                break
            tokens.append(vocab[inputs[i, j]])
        texts.append(" ".join(tokens))
    return texts

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu", default=-1, type=int, required=False)
    parser.add_argument(
        "--seed", default=83432534, type=int, required=False)

    parser.add_argument(
        "--model", default="data/models/cnn_nce.bin", type=str)

    args = parser.parse_args()

    model = torch.load(args.model, map_location=lambda storage, loc: storage)



#    random.seed(args.seed)
#    torch.manual_seed(args.seed)
#    if args.gpu > -1:
#        torch.cuda.manual_seed(args.seed)

    dataset = cnn_pretrain.datasets.imdb.get_dataset(part="unlabeled")
    dataset.batch_size = 1
    dataset.shuffle = True
    
    vocab = cnn_pretrain.datasets.imdb.get_vocab()


    for batch in dataset.iter_batch():
        input_texts = get_text(batch.inputs.sequence.data, vocab)
        knn, scores = model.nearest_neighbors(batch.inputs, 50)

        for i, text in enumerate(input_texts):
            print(textwrap.fill(input_texts[i]))
            print("")
            for idx, score in zip(knn.data[i], scores.data[i]):
                print(idx, score, vocab[idx])

        
            print("")
            print("")

        input()




#    if args.gpu > -1:
#        dataset.gpu = args.gpu

    freqs = cnn_pretrain.util.get_index_frequency(
        dataset.inputs.sequence, mask_value=0)
    freqs = freqs.float() ** .75
    vocab = cnn_pretrain.datasets.imdb.get_vocab()

    sampler = cnn_pretrain.dataio.ContextSampler(dataset, freqs,
        num_negative=args.noise_samples)

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
   # crit.add_reporter(ntp.criterion.MultiClassAccuracyReporter())
    opt = ntp.optimizer.Adam(model.parameters(), lr=args.lr)

    ntp.trainer.optimize_criterion(crit, model, opt, sampler,
                                   max_epochs=args.epochs,
                                   #validation_data=valid_sampler,
                                   save_model=args.save_model)

#    if args.save_model is not None:
#        print("Restoring model to best epoch...")
#        best_model = torch.load(args.save_model)
#
#        if args.save_predictor is not None:
#            pred_dir = os.path.dirname(args.save_predictor)
#            if pred_dir != "" and not os.path.exists(pred_dir):
#                os.makedirs(pred_dir)
#
#            data = {"model": best_model , "reader": reader}
#            print("Saving module and file reader...")
#            torch.save(data, args.save_predictor)

if __name__ == "__main__":
    main()
