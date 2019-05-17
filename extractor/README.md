# The extraction pipeline in OpineDB

## Required packages
- Tensorflow 1.13
- Spacy
- NLTK
- jsonlines

## How to run?
#### 0. Download the bert model and unzip it

```
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip 
unzip uncased_L-12_H-768_A-12.zip
```

Download the spacy language model and nltk punkt

```
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
python -m nltk.downloader stopwords 
```

#### 1. Train the tagging model

```python code/train_tagging.py bert_path data_path model_path```

like

```
python code/train_tagging.py \
  ../uncased_L-12_H-768_A-12 \
  models/tagging/hotel_data \
  models/tagging/hotel
```

For the restaurant model, simply replace "hotel" with "restaurant".

#### 2. Train the classification model

```
python code/train_classifier.py bert_path data_path model_path
```

like

```
python code/train_classifier.py \
  ../uncased_L-12_H-768_A-12 \
  models/classification/hotel_data \
  models/classification/hotel
```

For the restaurant model, simply replace "hotel" with "restaurant".

#### 3. Train the pairing model

```
export DATA_PATH=models/pairing/data
export MODEL_PATH=models/pairing/model
export BERT_PATH=~/uncased_L-12_H-768_A-12

python code/bert/run_classifier.py \
   --task_name=mrpc \
   --do_train=True \
   --do_eval=True \
   --do_predict=False \
   --data_dir=$DATA_PATH \
   --vocab_file=$BERT_PATH/vocab.txt \
   --bert_config_file=$BERT_PATH/bert_config.json \
   --init_checkpoint=$BERT_PATH/bert_model.ckpt \
   --max_seq_length=128 \
   --train_batch_size=16 \
   --learning_rate=2e-5 \
   --num_train_epochs=10.0 \
   --output_dir=$MODEL_PATH
```

#### 4. Run the extractor

You can run the extractor with the Makefile. Simply do ``` make ```. The configuration is written in ```config.json``` which looks like:

```
{
  "input_dir_path" : "./data",
  "bert_path" : "../uncased_L-12_H-768_A-12",
  "tagging_path" : "models/tagging/hotel",
  "pairing_path" : "models/pairing/model",
  "classifier_path" : "models/classification/hotel",
  "output_path" : "results/amsterdam_reviews_with_extractions.json"
}
```
