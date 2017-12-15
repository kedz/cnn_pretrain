import os
import argparse
import cnn_pretrain
import random
import torch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", default=83432534, type=int, required=False)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cnn_pretrain.datasets.imdb.create()

if __name__ == "__main__":
    main()
