. `dirname $BASH_SOURCE`"/env/bin/activate"
RELPATH=`dirname $BASH_SOURCE`
ABSPATH=`realpath $RELPATH`
export PYTHONPATH="${ABSPATH}/ntg/python:${ABSPATH}"
export NT_DATA="${ABSPATH}/ntp_data"
export CNN_PRETRAIN_DATA="${ABSPATH}/cnn_pretrain_data"
