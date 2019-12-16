# cmsc-473-673 

### Use yaml file if needed to set up conda env
#### conda env create --file 473NLPEmptyEnv.yaml
#### conda activate 473NLPEmptyEnv

## This is now using all 6 Languages in the following order: 
#### Lang1 is French, Lang2 is English, Lang3 is German
#### Lang 4 is Spanish, Lang 5 is Italian, Lang 6 is Dutch

### To train on train with training set and evaluate on dev with laplace constant of .1:

python "laplace6Lang.py" ./fr_gsd-ud-train.conllu ./en_ewt-ud-train.conllu ./de_gsd-ud-train.conllu ./es_gsd-ud-train.conllu ./it_isdt-ud-train.conllu ./nl_alpino-ud-train.conllu ./fr_gsd-ud-dev.conllu ./en_ewt-ud-dev.conllu ./de_gsd-ud-dev.conllu ./es_gsd-ud-dev.conllu ./it_isdt-ud-dev.conllu ./nl_alpino-ud-dev.conllu .1

### To output results to text file
python "laplace6Lang.py" ./fr_gsd-ud-train.conllu ./en_ewt-ud-train.conllu ./de_gsd-ud-train.conllu ./es_gsd-ud-train.conllu ./it_isdt-ud-train.conllu ./nl_alpino-ud-train.conllu ./fr_gsd-ud-dev.conllu ./en_ewt-ud-dev.conllu ./de_gsd-ud-dev.conllu ./es_gsd-ud-dev.conllu ./it_isdt-ud-dev.conllu ./nl_alpino-ud-dev.conllu .1 >outputlaplace.txt

### Later after tuning on dev set and finding best laplace constant, check results on test sets. Substitute .1 with best laplace constant
python "laplace6Lang.py" ./fr_gsd-ud-train.conllu ./en_ewt-ud-train.conllu ./de_gsd-ud-train.conllu ./es_gsd-ud-train.conllu ./it_isdt-ud-train.conllu ./nl_alpino-ud-train.conllu ./fr_gsd-ud-test.conllu ./en_ewt-ud-test.conllu ./de_gsd-ud-test.conllu ./es_gsd-ud-test.conllu ./it_isdt-ud-test.conllu ./nl_alpino-ud-test.conllu .1

### For backoff model run the following commands with optional commands for uniBackoff and biBackoff constants which default to .5 otherwise
python "backoff6lang.py" ./fr_gsd-ud-train.conllu ./en_ewt-ud-train.conllu ./de_gsd-ud-train.conllu ./es_gsd-ud-train.conllu ./it_isdt-ud-train.conllu ./nl_alpino-ud-train.conllu ./fr_gsd-ud-dev.conllu ./en_ewt-ud-dev.conllu ./de_gsd-ud-dev.conllu ./es_gsd-ud-dev.conllu ./it_isdt-ud-dev.conllu ./nl_alpino-ud-dev.conllu .5 .5
