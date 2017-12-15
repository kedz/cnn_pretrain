import os


def get_data_path():
    return os.getenv(
        "CNN_PRETRAIN_DATA", 
        os.path.expanduser(os.path.join("~", "cnn_pretrain_data")))
