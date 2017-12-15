SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
echo $SCRIPTPATH
cd $SCRIPTPATH
git clone https://github.com/kedz/ntg.git
cd ntg
git checkout spen
git pull origin spen
cd ..
virtualenv --python=python3.6 env
source env/bin/activate

pip3 install numpy scipy scikit-learn pandas nltk
pip3 install http://download.pytorch.org/whl/cu75/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
