import random
import torch
import cnn_pretrain
import pandas as pd
import argparse

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
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--pos-samples", type=int, default=10)
    parser.add_argument("--inv-freq", action="store_true", default=False)
    args = parser.parse_args()

    seed = 84872342
    random.seed(seed)
    torch.manual_seed(seed)


    num_pos = args.pos_samples
    num_neg = 1

    dataset = cnn_pretrain.datasets.imdb.get_dataset(part="unlabeled")
    dataset.batch_size = 5
    dataset.shuffle = True
    dataset.gpu = 0
    
    vocab_freqs = cnn_pretrain.util.get_index_frequency(
        dataset.inputs.sequence, mask_value=0)
    vocab_freqs_squashed = vocab_freqs.float() ** .75

    vocab = cnn_pretrain.datasets.imdb.get_vocab()

    sampler = cnn_pretrain.dataio.ContextSampler(
        dataset, vocab_freqs_squashed, num_positive=num_pos, 
        num_negative=num_neg, pos_inv_freq=args.inv_freq)

    sampled = 0
    for batch in sampler.iter_batch():
        texts = get_text(batch.inputs.sequence.data, vocab)
        batch_size = batch.inputs.sequence.size(0)
        for b, text in enumerate(texts):
            if sampled >= args.samples:
                return 
            sampled += 1
            print(" === Example {} ===".format(sampled))

            print(textwrap.fill(text))

            pos_tokens = [vocab[batch.inputs.samples.data[b, p, 0]] 
                          for p in range(num_pos)]
            neg_indices = batch.inputs.samples.data[b,:,1:].contiguous().view(
                -1)
            neg_tokens = [vocab[idx] for idx in neg_indices]
            pos_tokens += [""] * (len(neg_tokens) - len(pos_tokens))
            df = pd.DataFrame([pos_tokens, neg_tokens]).T
            df.columns = ["positive", "negative"]
            print("")
            print(df)
            print("")

if __name__ == "__main__":
    main()
