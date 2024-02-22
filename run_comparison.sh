#!/bin/sh

##########################
##########################
### DOWNLOAD DWUG DATA ###
##########################
##########################
# Download SemEval-English
wget https://zenodo.org/records/7387261/files/dwug_en.zip?download=1
unzip dwug_en.zip?download=1
rm dwug_en.zip?download=1
mv dwug_en dwug_en12

# Download SemEval-Swedish
wget https://zenodo.org/records/7389506/files/dwug_sv.zip?download=1
unzip dwug_sv.zip?download=1
rm dwug_sv.zip?download=1
mv dwug_sv dwug_sv12

# Download SemEval-German
wget https://zenodo.org/records/7441645/files/dwug_de.zip?download=1
unzip dwug_de.zip?download=1
rm dwug_de.zip?download=1
mv dwug_de dwug_de12

# Download SemeEval-Latin
wget https://zenodo.org/records/5255228/files/dwug_la.zip?download=1
unzip dwug_la.zip?download=1
rm dwug_la.zip?download=1
mv dwug_la dwug_la12

# Download LSCDiscovery-Spanish
wget https://zenodo.org/records/6433667/files/dwug_es.zip?download=1
unzip dwug_es.zip?download=1
rm dwug_es.zip?download=1
mv dwug_es dwug_es12

# Download ChiWUG
wget https://zenodo.org/records/10023263/files/chiwug.zip?download=1
unzip chiwug.zip?download=1
rm chiwug.zip?download=1
mv chiwug dwug_zh12

# Download RuShiftEval
git clone https://github.com/akutuzov/rushifteval_public.git
mv rushifteval_public/durel/rushifteval1/ dwug_ru12
mv rushifteval_public/durel/rushifteval2/ dwug_ru23
mv rushifteval_public/durel/rushifteval3/ dwug_ru13
rm -rf rushifteval_public

# Download NorDiaChange
git clone https://github.com/ltgoslo/nor_dia_change.git
mv nor_dia_change/subset1 dwug_no12
mv nor_dia_change/subset2 dwug_no23
rm -rf nor_dia_change

##########################
##########################
### DOWNLOAD XL-Lexeme ###
##########################
##########################

# Download XL-Lexeme
git clone https://github.com/pierluigic/xl-lexeme.git
cd xl-lexeme
pip3 install .

####################################################
####################################################
### DOWNLOAD Correlation Clustering in Python ######
####################################################
####################################################

# Download Correlation Clustering
git clone https://github.com/Garrafao/correlation_clustering.git
mv correlation_clustering/src/correlation.py src/correlation_clustering.py
rm -rf correlation_clustering

##########################
##########################
## INSTALL REQUIREMENTS ##
##########################
##########################
pip install -r requirements.txt

##########################
##########################
## EMBEDDING EXTRACTION ##
##########################
##########################

# Embeddings: dwug_en12
python src/embs.py --benchmark dwug_en12 --model_name bert --pretrained_model bert-base-uncased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_en12 --model_name mbert --pretrained_model bert-base-multilingual-cased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_en12 --model_name xlm-r --pretrained_model xlm-roberta-base --device cuda --subword_prefix "_" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_en12 --model_name xl-lexeme --pretrained_model pierluigic/xl-lexeme --device cuda --batch_size 16 --max_length 512

# Embeddings: dwug_sv12
python src/embs.py --benchmark dwug_sv12 --model_name bert --pretrained_model af-ai-center/bert-base-swedish-uncased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_sv12 --model_name mbert --pretrained_model bert-base-multilingual-cased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_sv12 --model_name xlm-r --pretrained_model xlm-roberta-base --device cuda --subword_prefix "_" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_sv12 --model_name xl-lexeme --pretrained_model pierluigic/xl-lexeme --device cuda --batch_size 16 --max_length 512

# Embeddings: dwug_de12
python src/embs.py --benchmark dwug_de12 --model_name bert --pretrained_model bert-base-german-cased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_de12 --model_name mbert --pretrained_model bert-base-multilingual-cased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_de12 --model_name xlm-r --pretrained_model xlm-roberta-base --device cuda --subword_prefix "_" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_de12 --model_name xl-lexeme --pretrained_model pierluigic/xl-lexeme --device cuda --batch_size 16 --max_length 512

# Embeddings: dwug_la12
python src/embs.py --benchmark dwug_la12 --model_name mbert --pretrained_model bert-base-multilingual-cased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_la12 --model_name xlm-r --pretrained_model xlm-roberta-base --device cuda --subword_prefix "_" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_la12 --model_name xl-lexeme --pretrained_model pierluigic/xl-lexeme --device cuda --batch_size 16 --max_length 512

# Embeddings: dwug_es12
python src/embs.py --benchmark dwug_es12 --model_name bert --pretrained_model dccuchile/bert-base-spanish-wwm-uncased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_es12 --model_name mbert --pretrained_model bert-base-multilingual-cased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_es12 --model_name xlm-r --pretrained_model xlm-roberta-base --device cuda --subword_prefix "_" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_es12 --model_name xl-lexeme --pretrained_model pierluigic/xl-lexeme --device cuda --batch_size 16 --max_length 512

# Embeddings: dwug_ru12
python src/embs.py --benchmark dwug_ru12 --model_name bert --pretrained_model DeepPavlov/rubert-base-cased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_ru12 --model_name mbert --pretrained_model bert-base-multilingual-cased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_ru12 --model_name xlm-r --pretrained_model xlm-roberta-base --device cuda --subword_prefix "_" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_ru12 --model_name xl-lexeme --pretrained_model pierluigic/xl-lexeme --device cuda --batch_size 16 --max_length 512

# Embeddings: dwug_ru23
python src/embs.py --benchmark dwug_ru23 --model_name bert --pretrained_model DeepPavlov/rubert-base-cased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_ru23 --model_name mbert --pretrained_model bert-base-multilingual-cased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_ru23 --model_name xlm-r --pretrained_model xlm-roberta-base --device cuda --subword_prefix "_" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_ru23 --model_name xl-lexeme --pretrained_model pierluigic/xl-lexeme --device cuda --batch_size 16 --max_length 512

# Embeddings: dwug_ru13
python src/embs.py --benchmark dwug_ru13 --model_name bert --pretrained_model DeepPavlov/rubert-base-cased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_ru13 --model_name mbert --pretrained_model bert-base-multilingual-cased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_ru13 --model_name xlm-r --pretrained_model xlm-roberta-base --device cuda --subword_prefix "_" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_ru13 --model_name xl-lexeme --pretrained_model pierluigic/xl-lexeme --device cuda --batch_size 16 --max_length 512

# Embeddings: dwug_no12
python src/embs.py --benchmark dwug_no12 --model_name bert --pretrained_model NbAiLab/nb-bert-base --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_no12 --model_name mbert --pretrained_model bert-base-multilingual-cased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_no12 --model_name xlm-r --pretrained_model xlm-roberta-base --device cuda --subword_prefix "_" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_no12 --model_name xl-lexeme --pretrained_model pierluigic/xl-lexeme --device cuda --batch_size 16 --max_length 512

# Embeddings: dwug_no23
python src/embs.py --benchmark dwug_no23 --model_name bert --pretrained_model NbAiLab/nb-bert-base --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_no23 --model_name mbert --pretrained_model bert-base-multilingual-cased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_no23 --model_name xlm-r --pretrained_model xlm-roberta-base --device cuda --subword_prefix "_" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_no23 --model_name xl-lexeme --pretrained_model pierluigic/xl-lexeme --device cuda --batch_size 16 --max_length 512

# Embeddings: dwug_zh12
python src/embs.py --benchmark dwug_zh12 --model_name bert --pretrained_model bert-base-chinese --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_zh12 --model_name mbert --pretrained_model bert-base-multilingual-cased --device cuda --subword_prefix "##" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_zh12 --model_name xlm-r --pretrained_model xlm-roberta-base --device cuda --subword_prefix "_" --batch_size 16 --max_length 512
python src/embs.py --benchmark dwug_zh12 --model_name xl-lexeme --pretrained_model pierluigic/xl-lexeme --device cuda --batch_size 16 --max_length 512

##########################
##########################
## EMBEDDING EVALUATION ## (INDIVIDUAL LAYER) #
##########################
##########################

declare -a benchmarks=("dwug_en12" "dwug_sv12" "dwug_la12" "dwug_de12" "dwug_zh12" "dwug_es12" "dwug_ru12" "dwug_ru23" "dwug_ru13" "dwug_no12" "dwug_no23")
declare -a models=("bert" "mbert" "xlm-r")
declare -a layers=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")

for benchmark in "${benchmarks[@]}"
do
    echo "Eval: ${benchmark}"
    for model in "${models[@]}"
    do
	# skip bert for dwug_la12
	if [ "${model}" == "bert" ] && [ "${benchmark}" == "dwug_la12" ]; then
	    continue
	fi
	
	echo "- ${model}"
	for layer in "${layers[@]}"
	do
	    python src/stats.py --benchmark "${benchmark}" --model_name "${model}" --layer "${layer}"
	done
    done
    echo "- xl-lexeme"
    python src/stats.py --benchmark "${benchmark}" --model_name "xl-lexeme" --layer "tuned"
done


##########################
##########################
## EMBEDDING EVALUATION ## (COMBINATIONS) #
##########################
##########################

declare -a benchmarks=("dwug_en12" "dwug_sv12" "dwug_la12" "dwug_de12" "dwug_zh12" "dwug_es12" "dwug_ru12" "dwug_ru23" "dwug_ru13" "dwug_no12" "dwug_no23")
declare -a models=("bert" "mbert" "xlm-r")
layers=12
depth=4

for benchmark in "${benchmarks[@]}"
do
    echo "Eval: ${benchmark}"
    for model in "${models[@]}"
    do
	# skip bert for dwug_la12
	if [ "${model}" == "bert" ] && [ "${benchmark}" == "dwug_la12" ]; then
	    continue
	fi
	
	echo "- ${model}"
	python src/comb_stats.py --benchmark "${benchmark}" --model_name "${model}" --layers "${layers}" --depth "${depth}"
    done
done

##############################
##############################
## COMPUTATIONAL ANNOTATION ##
##############################
##############################

layer=12
batch_size=16

# Eval: dwug en12
benchmark="dwug_en12"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xl-lexeme" --batch_size "${batch_size}" --pretrained_model "pierluigic/xl-lexeme" --device "cuda"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "bert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "bert-base-uncased" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "mbert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "bert-base-multilingual-cased" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xlm-r" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "xlm-roberta-base" --device "cuda" --subword_prefix "_"

# Eval: dwug sv12
benchmark="dwug_sv12"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "bert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "af-ai-center/bert-base-swedish-uncased" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "mbert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "bert-base-multilingual-cased" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xlm-r" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "xlm-roberta-base" --device "cuda" --subword_prefix "_"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xl-lexeme" --batch_size "${batch_size}" --pretrained_model "pierluigic/xl-lexeme" --device "cuda"

# Eval: dwug la12
benchmark="dwug_la12"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "mbert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "bert-base-multilingual-cased" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xlm-r" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "xlm-roberta-base" --device "cuda" --subword_prefix "_"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xl-lexeme" --batch_size "${batch_size}" --pretrained_model "pierluigic/xl-lexeme" --device "cuda"

# Eval: dwug de12
benchmark="dwug_de12"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "bert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "bert-base-german-cased" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "mbert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "bert-base-multilingual-cased" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xlm-r" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "xlm-roberta-base" --device "cuda" --subword_prefix "_"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xl-lexeme" --batch_size "${batch_size}" --pretrained_model "pierluigic/xl-lexeme" --device "cuda"

# Eval: dwug es12
benchmark="dwug_es12"
#python src/computational_annotation.py --benchmark "${benchmark}" --model_name "bert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "dccuchile/bert-base-spanish-wwm-uncased" --device "cuda" --subword_prefix "##"
#python src/computational_annotation.py --benchmark "${benchmark}" --model_name "mbert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "bert-base-multilingual-cased" --device "cuda" --subword_prefix "##"
#python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xlm-r" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "xlm-roberta-base" --device "cuda" --subword_prefix "_"
#python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xl-lexeme" --batch_size "${batch_size}" --pretrained_model "pierluigic/xl-lexeme" --device "cuda"

# Eval: dwug ru12
benchmark="dwug_ru12"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "bert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "DeepPavlov/rubert-base-cased" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "mbert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "bert-base-multilingual-cased" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xlm-r" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "xlm-roberta-base" --device "cuda" --subword_prefix "_"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xl-lexeme" --batch_size "${batch_size}" --pretrained_model "pierluigic/xl-lexeme" --device "cuda"

# Eval: dwug ru23
benchmark="dwug_ru23"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "bert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "DeepPavlov/rubert-base-cased" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "mbert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "bert-base-multilingual-cased" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xlm-r" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "xlm-roberta-base" --device "cuda" --subword_prefix "_"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xl-lexeme" --batch_size "${batch_size}" --pretrained_model "pierluigic/xl-lexeme" --device "cuda"

# Eval: dwug ru13
benchmark="dwug_ru13"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "bert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "DeepPavlov/rubert-base-cased" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "mbert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "bert-base-multilingual-cased" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xlm-r" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "xlm-roberta-base" --device "cuda" --subword_prefix "_"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xl-lexeme" --batch_size "${batch_size}" --pretrained_model "pierluigic/xl-lexeme" --device "cuda"

# Eval: dwug no12
benchmark="dwug_no12"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "bert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "NbAiLab/nb-bert-base" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "mbert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "bert-base-multilingual-cased" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xlm-r" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "xlm-roberta-base" --device "cuda" --subword_prefix "_"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xl-lexeme" --batch_size "${batch_size}" --pretrained_model "pierluigic/xl-lexeme" --device "cuda"

# Eval: dwug no23
benchmark="dwug_no23"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "bert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "NbAiLab/nb-bert-base" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "mbert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "bert-base-multilingual-cased" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xlm-r" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "xlm-roberta-base" --device "cuda" --subword_prefix "_"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xl-lexeme" --batch_size "${batch_size}" --pretrained_model "pierluigic/xl-lexeme" --device "cuda"

# Eval: dwug zh12
benchmark="dwug_zh12"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "bert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "bert-base-chinese" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "mbert" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "bert-base-multilingual-cased" --device "cuda" --subword_prefix "##"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xlm-r" --layer "${layer}" --batch_size "${batch_size}" --pretrained_model "xlm-roberta-base" --device "cuda" --subword_prefix "_"
python src/computational_annotation.py --benchmark "${benchmark}" --model_name "xl-lexeme" --batch_size "${batch_size}" --pretrained_model "pierluigic/xl-lexeme" --device "cuda"


# GPT-4 EVALUATION (VERY EXPENSIVE, WE DID IT ONLY FOR ENGLISH)
declare -a benchmarks=("dwug_en12") # "dwug_sv12" "dwug_de12" "dwug_zh12" "dwug_es12" "dwug_ru12" "dwug_ru23" "dwug_ru13" "dwug_no12" "dwug_no23")

for benchmark in "${benchmarks[@]}"
do
    echo "Eval: ${benchmark}"
    python src/chatgpt.py --benchmark "${benchmark}"
done

#################################
#################################
## EVALUATION WITH WIC;WSI;GCD ##
#################################
#################################

declare -a benchmarks=("dwug_en12" "dwug_sv12" "dwug_de12" "dwug_es12" "dwug_ru12" "dwug_ru23" "dwug_ru13" "dwug_no12" "dwug_no23" "dwug_zh12")

for benchmark in "${benchmarks[@]}"
do
    wsi_lsc="--wsi_lsc"

    if [[ $benchmark == "dwug_ru12" || $benchmark == "dwug_ru23" || $benchmark == "dwug_ru13" ]]
    then
		wsi_lsc="compare"
    fi

    if [[ $benchmark == "dwug_en12" ]]
    then
		python src/model_evaluation.py --benchmark "dwug_en12" --model "chatgpt" --layer "chatgpt" --wsi_lsc
    fi
    

    if [[ "$wsi_lsc" == "--wsi_lsc" ]]; then
    	python src/model_evaluation.py --benchmark "${benchmark}" --model "xl-lexeme" --layer "tuned" --wsi_lsc
		python src/model_evaluation.py --benchmark "${benchmark}" --model "bert" --layer "12" --wsi_lsc
		python src/model_evaluation.py --benchmark "${benchmark}" --model "mbert" --layer "12" --wsi_lsc
		python src/model_evaluation.py --benchmark "${benchmark}" --model "xlm-r" --layer "12" --wsi_lsc
    else
    	python src/model_evaluation.py --benchmark "${benchmark}" --model "xl-lexeme" --layer "tuned"
		python src/model_evaluation.py --benchmark "${benchmark}" --model "bert" --layer "12"
		python src/model_evaluation.py --benchmark "${benchmark}" --model "mbert" --layer "12"
		python src/model_evaluation.py --benchmark "${benchmark}" --model "xlm-r" --layer "12"
    fi
done

#################################
#################################
#####  COMBINATIONS BOXPLOT  ####
#################################
#################################
python src/boxplot.py

#########
#THE END#
#########
