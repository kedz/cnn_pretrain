import os

import torch
import ntp
from ntp.datasets import imdb

def get_data_path():
    return os.getenv(
        "CNN_PRETRAIN_DATA", 
        os.path.expanduser(os.path.join("~", "cnn_pretrain_data")))

def get_sample(size=25):
    train_tsv_path = imdb.get_imdb_data_path(split="train")
    with open("tmp.tsv", "w") as tmp_fp:
        with open(train_tsv_path, "r") as fp:
            tmp_fp.write(fp.readline())
            for i, line in enumerate(fp):
                if i == size:
                    break
                tmp_fp.write(line)
    
    tmp_name = "tmp.tsv"

    label_field = ntp.dataio.field_reader.Label(
        "label", vocabulary=["neg", "pos"])
    label_field.fit_parameters()

    input_field = ntp.dataio.field_reader.TokenSequence("text")

    labeled_reader = ntp.dataio.file_reader.tsv_reader(
        [input_field, label_field], skip_header=True, show_progress=True)

    labeled_reader.fit_parameters(tmp_name)
    (tr_inputs, tr_lengths), (tr_labels,) = labeled_reader.read(tmp_name)

    labeled_layout = [["inputs", [["sequence", "sequence"],
                          ["length", "length"]]],
                      ["targets", "targets"]]

    dataset = ntp.dataio.Dataset(
        (tr_inputs, tr_lengths, "sequence"),
        (tr_lengths, None, "length"),
        (tr_labels, None, "targets"),
        layout=labeled_layout,
        lengths=tr_lengths)
        
    return dataset, input_field.vocab

def get_dataset(part='train'):
    if part not in ["train", "valid", "test", "unlabeled"]:
        raise Exception(
            "part must be one of 'train', 'valid', 'test', 'unlabeled'.")
    path = os.path.join(
        get_data_path(), "imdb", "datasets", "{}.bin".format(part))
    if not os.path.exists(path):
        create()

    return torch.load(path)

def get_labeled_data_reader():
    path = os.path.join(get_data_path(), "imdb", "datasets", "imdb_reader.bin")
    if not os.path.exists(path):
        create()
    return torch.load(path)

def get_vocab():
    path = os.path.join(get_data_path(), "imdb", "datasets", "imdb_vocab.bin")
    if not os.path.exists(path):
        create()
    return torch.load(path)
    
def create(at_least=10, start_token=None, stop_token=None, replace_digit="#",
           lower=True, train_per=.85):

    imdb_data_path = os.path.join(get_data_path(), "imdb")
    dataset_path = os.path.join(imdb_data_path, "datasets")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    reader_path = os.path.join(dataset_path, "imdb_reader.bin")
    vocab_path = os.path.join(dataset_path, "imdb_vocab.bin")
    train_dataset_path = os.path.join(dataset_path, "train.bin")
    valid_dataset_path = os.path.join(dataset_path, "valid.bin")
    test_dataset_path = os.path.join(dataset_path, "test.bin")
    unlabeled_dataset_path = os.path.join(dataset_path, "unlabeled.bin")

    train_tsv_path = imdb.get_imdb_data_path(split="train")
    test_tsv_path = imdb.get_imdb_data_path(split="test")
    unsup_tsv_path = imdb.get_imdb_data_path(split="unsup")

    label_field = ntp.dataio.field_reader.Label(
        "label", vocabulary=["neg", "pos"])
    label_field.fit_parameters()

    input_field = ntp.dataio.field_reader.TokenSequence(
        "text", at_least=at_least, start_token=start_token,
        stop_token=stop_token, lower=lower, replace_digit=replace_digit)

    # Get vocab from union of training and unsupervised data.

    print("Fitting vocabulary parameters...")
    unlabeled_reader = ntp.dataio.file_reader.tsv_reader(
        [input_field], skip_header=True, show_progress=True)
    unlabeled_reader.fit_parameters(paths=[train_tsv_path, unsup_tsv_path])
    print("Saving vocabulary to {} ...".format(vocab_path))
    torch.save(input_field.vocab, vocab_path)
    
    labeled_reader = ntp.dataio.file_reader.tsv_reader(
        [input_field, label_field], skip_header=True, show_progress=True)

    print("Saving labeled data reader to {} ...".format(reader_path))
    torch.save(labeled_reader, reader_path)

    print("Reading original training partition...")
    (tr_inputs, tr_lengths), (tr_labels,) = labeled_reader.read(train_tsv_path)

    labeled_layout = [["inputs", [["sequence", "sequence"],
                          ["length", "length"]]],
                      ["targets", "targets"]]

    train_valid_dataset = ntp.dataio.Dataset(
        (tr_inputs, tr_lengths, "sequence"),
        (tr_lengths, None, "length"),
        (tr_labels, None, "targets"),
        layout=labeled_layout,
        lengths=tr_lengths)
        
    train_indices, valid_indices = ntp.trainer.stratified_generate_splits(
        train_valid_dataset.size, 
        train_valid_dataset.targets, 
        train_per=train_per, 
        valid_per=0)
    
    print("Writing labeled training partition to {} ...".format(
        train_dataset_path))
    train_dataset = train_valid_dataset.index_select(train_indices)
    torch.save(train_dataset, train_dataset_path)

    print("Writing labeled validation partition to {} ...".format(
        valid_dataset_path))
    valid_dataset = train_valid_dataset.index_select(valid_indices)
    torch.save(valid_dataset, valid_dataset_path)

    print("Reading unlabeled data...")
    (unsup_inputs, unsup_lengths), = unlabeled_reader.read(
        unsup_tsv_path)
    
    max_length = max(train_dataset.inputs.sequence.size(1), 
                     unsup_inputs.size(1))
    num_inputs = train_dataset.size + unsup_inputs.size(0)
    
    unlabeled_inputs = torch.LongTensor(num_inputs, max_length).fill_(0)

    train_size = train_dataset.size
    train_seq_size = train_dataset.inputs.sequence.size(1)

    unlabeled_inputs[:train_size,:train_seq_size].copy_(
        train_dataset.inputs.sequence)

    unsup_size = unsup_inputs.size(0)
    unsup_seq_size = unsup_inputs.size(1)

    unlabeled_inputs[train_size:,:unsup_seq_size].copy_(
        unsup_inputs)

    unlabeled_lengths = torch.cat(
        [train_dataset.inputs.length, unsup_lengths], 0)

    unlabeled_layout = [["inputs", [["sequence", "sequence"],
                                    ["length", "length"]]],]

    unlabeled_dataset = ntp.dataio.Dataset(
        (unlabeled_inputs, unlabeled_lengths, "sequence"),
        (unlabeled_lengths, None, "length"),
        layout=unlabeled_layout,
        lengths=unlabeled_lengths)
 
    print("Writing union of training and unlabeled partition to {} ...".format(
        unlabeled_dataset_path))
    torch.save(unlabeled_dataset, unlabeled_dataset_path)

    print("Reading original test partition...")
    (te_inputs, te_lengths), (te_labels,) = labeled_reader.read(test_tsv_path)

    test_dataset = ntp.dataio.Dataset(
        (te_inputs, te_lengths, "sequence"),
        (te_lengths, None, "length"),
        (te_labels, None, "targets"),
        layout=labeled_layout,
        lengths=te_lengths)
    
    print("Writing labeled testing partition to {} ...".format(
        test_dataset_path))
    torch.save(test_dataset, test_dataset_path)
