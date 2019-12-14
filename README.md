# cmsc-473-673 

### Use yaml file if needed to set up conda env
#### conda env create --file 473NLPEmptyEnv.yaml
#### conda activate 473NLPEmptyEnv

### For question 7, run commands as in the following example:
##### Note, it is using a french and english training set

To train evaluate on dev, serialize, and set oov cutoff point to 1 and laplace constant to .1:

python "laplaceNoisyChannel.py" ./fr_gsd-ud-train.conllu ./en_ewt-ud-train.conllu ./fr_gsd-ud-test.conllu ./en_ewt-ud-test.conllu usedtobeforserializationbutcommentedout  1 .1

