###############################################################################################
#
# Script for training good word and phrase vector model using public corpora, version 1.0.
# The training time will be from several hours to about a day.
#
# Downloads about 8 billion words, makes phrases using two runs of word2phrase, trains
# a 500-dimensional vector model and evaluates it on word and phrase analogy tasks.
#
###############################################################################################

# ./bin/word2phrase -train data.txt -output data-phrase.txt -threshold 200 -debug 2
# ./bin/word2phrase -train data-phrase.txt -output data-phrase2.txt -threshold 100 -debug 2

# 100b: -cbow 1 -size 300 -window 5 -negative 3 -hs 0 -sample 1e-5 -threads 12 -binary 1 -min-count 10
#PARAM="-cbow 1 -size 500 -window 10 -negative 10 -hs 0 -sample 1e-5 -threads 40 -binary 1 -iter 3 -min-count 10"
PARAM="-cbow 1 -size 100 -window 15 -negative 20 -hs 0 -sample 1e-5 -threads 56 -binary 1 -iter 50 -min-count 100"

BIN_PATH=./bin

#DATA_FPATH=../word2vec_data/data-phrase2.txt
DATA_PATH=../word2vec_data
DATA_FPATH=${DATA_PATH}/data_no_unk_tag.txt

WVEC_FPATH=vectors.bin
EVAL_FPATH=eval.txt

QWRD_FPATH=${DATA_PATH}/questions/questions-words.txt
QPHR_FPATH=${DATA_PATH}/questions/questions-phrases.txt

if [[ ${1} = "train" ]]; then
    $BIN_PATH/word2vec -train $DATA_FPATH -output $WVEC_FPATH $PARAM
    echo ""
    echo "Training done!"
fi

echo "Start evaluation: questions-words"
echo $PARAM                                                  >  $EVAL_FPATH
$BIN_PATH/compute-accuracy $WVEC_FPATH  400000 < $QWRD_FPATH >> $EVAL_FPATH  # should get to almost 78% accuracy on 99.7% of questions
echo "Start evaluation: questions-phrases"
echo "======================="                               >> $EVAL_FPATH
$BIN_PATH/compute-accuracy $WVEC_FPATH 1000000 < $QPHR_FPATH >> $EVAL_FPATH  # about 78% accuracy with 77% coverage
echo "Evaluation finished."
