INPUT_PATH := $(shell jq -r '.input_dir_path' config.json)
BERT_PATH := $(shell jq -r '.bert_path' config.json)
TAGGING_MODEL_PATH := $(shell jq -r '.tagging_path' config.json)
PAIRING_MODEL_PATH := $(shell jq -r '.pairing_path' config.json)
CLASSIFIER_MODEL_PATH := $(shell jq -r '.classifier_path' config.json)
OUTPUT := $(shell jq -r '.output_path' config.json)

# REVIEWS = data/restaurant_reviews_with_extractions.json
REVIEWS := $(INPUT_PATH)/raw_reviews.csv

SENTENCES := $(INPUT_PATH)/sentences_with_extractions.json

# PREPROCESSED_REVIEWS = data/preprocessed_restaurant_reviews_with_extractions.jsonl
PREPROCESSED_REVIEWS := $(INPUT_PATH)/reviews_with_sids.jsonl

# TAGGING_PATH = models/tagging/restaurant_data
TAGGING_PATH := $(INPUT_PATH)/tagging_data/

PAIRING_PATH := $(INPUT_PATH)/pairing_data/

# CLASSIFIER_INPUT = models/classification/restaurant_data/
CLASSIFIER_INPUT := $(INPUT_PATH)/classifier_data/

# LABELS_PATH = data/restaurant_attr_list.txt
LABELS_PATH := $(CLASSIFIER_MODEL_PATH)/label_mp.json

ALL: POSTPROCESS 

$(TAGGING_DATA)/test.txt $(PREPROCESSED_REVIEWS) : $(REVIEWS)
	python code/preprocess.py $(REVIEWS) $(TAGGING_PATH) $(PREPROCESSED_REVIEWS)

$(TAGGING_MODEL_PATH)/label_test.txt : $(TAGGING_PATH)/test.txt 
	python code/BERT-BiLSTM-CRF-NER/bert_lstm_ner.py \
      --task_name=NER  \
	  --do_train=False   \
	  --do_eval=False   \
	  --do_predict=True \
	  --data_dir=$(TAGGING_PATH)   \
	  --vocab_file=$(BERT_PATH)/vocab.txt \
	  --bert_config_file=$(BERT_PATH)/bert_config.json \
	  --init_checkpoint=$(BERT_PATH)/bert_model.ckpt \
	  --max_seq_length=128   \
	  --train_batch_size=16   \
	  --learning_rate=2e-5   \
	  --num_train_epochs=10.0   \
	  --output_dir=$(TAGGING_MODEL_PATH)/

$(SENTENCES) $(CLASSIFIER_INPUT)/test.tsv: $(TAGGING_MODEL_PATH)/label_test.txt
	python code/pairing.py $(TAGGING_MODEL_PATH)/label_test.txt $(SENTENCES) $(PAIRING_PATH) $(PAIRING_MODEL_PATH) $(CLASSIFIER_INPUT) $(BERT_PATH)

$(CLASSIFIER_MODEL_PATH)/test_results.tsv: $(CLASSIFIER_INPUT)/test.tsv
	python code/bert/run_classifier.py \
	  --task_name=cola \
	  --do_train=False \
	  --do_eval=False \
	  --do_predict=True \
	  --data_dir=$(CLASSIFIER_INPUT) \
      --vocab_file=$(BERT_PATH)/vocab.txt \
      --bert_config_file=$(BERT_PATH)/bert_config.json \
      --init_checkpoint=$(BERT_PATH)/bert_model.ckpt \
	  --max_seq_length=16 \
	  --train_batch_size=32 \
	  --learning_rate=2e-5 \
	  --num_train_epochs=3.0 \
	  --output_dir=$(CLASSIFIER_MODEL_PATH)/

POSTPROCESS: $(PREPROCESSED_REVIEWS) $(SENTENCES) $(CLASSIFIER_MODEL_PATH)/test_results.tsv $(LABELS_PATH)
	python code/postprocess.py $(PREPROCESSED_REVIEWS) $(SENTENCES) $(CLASSIFIER_MODEL_PATH)/test_results.tsv $(LABELS_PATH) $(OUTPUT)

clean:
	rm $(PREPROCESSED_REVIEWS)
	rm -rf $(TAGGING_PATH)
	rm -rf $(PAIRING_PATH)
	rm $(SENTENCES)
