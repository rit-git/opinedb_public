import os
import sys

def train_classifier(bert_path, data_path, model_path):
    # create the model path, if it doesn't exist
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # run the training command
    vocab_path = os.path.join(bert_path, 'vocab.txt')
    config_path = os.path.join(bert_path, 'bert_config.json')
    checkpoint_path = os.path.join(bert_path, 'bert_model.ckpt')

    cmd = """python code/BERT-BiLSTM-CRF-NER/bert_lstm_ner.py \
  --task_name=NER  \
  --do_train=True   \
  --do_eval=True   \
  --do_predict=False \
  --data_dir=%s   \
  --vocab_file=%s \
  --bert_config_file=%s \
  --init_checkpoint=%s \
  --max_seq_length=128   \
  --train_batch_size=16   \
  --learning_rate=2e-5   \
  --num_train_epochs=10.0   \
  --output_dir=%s""" % (data_path, vocab_path, config_path, checkpoint_path, model_path)
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python train_tagging.py bert_path data_path model_path")
        exit()
    bert_path = sys.argv[1]
    data_path = sys.argv[2]
    model_path = sys.argv[3]
    train_classifier(bert_path, data_path, model_path)
