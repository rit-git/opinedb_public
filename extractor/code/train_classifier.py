import json
import jsonlines
import os
import sys
import random

filenames = ['train.txt', 'dev.txt']
# filenames = ['train.txt', 'test.txt', 'dev.txt']

def collect_labels(data_path, label_mp):
    for fn in filenames:
        path = os.path.join(data_path, fn)
        if not os.path.exists(path):
            continue
        for line in open(path):
            items = line.strip().split('\t')
            label = items[-1]
            if label not in label_mp:
                label_mp[label] = len(label_mp)
    return label_mp


def convert_to_tsv(data_path, in_fn, out_fn, label_mp):
    in_fn = os.path.join(data_path, in_fn)
    out_fn = os.path.join(data_path, out_fn)
    with open(out_fn, 'w') as fout:
        if 'test.tsv' in out_fn:
            fout.write('index\tsentence\n')
        idx = 0
        output_lines = []
        for line in open(in_fn):
            items = line.strip().split('\t')
            label = label_mp[items[-1]]
            sentence = items[0].strip()
            if 'test.tsv' not in out_fn:
                output_lines.append('%d\t%d\t\t%s\n' % (idx, label, sentence))
            else:
                output_lines.append('%d\t%s\n' % (idx, sentence))
            idx += 1
        random.shuffle(output_lines)
        for line in output_lines:
            fout.write(line)


def train_classifier(bert_path, data_path, model_path):
    # list all the tsv files
    label_mp = {}
    collect_labels(data_path, label_mp)
    for fn in filenames:
        out_fn = fn[:-3] + 'tsv'
        convert_to_tsv(data_path, fn, out_fn, label_mp)

    # create the model path, if it doesn't exist
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    json.dump(label_mp, open(os.path.join(model_path, 'label_mp.json'), 'w'))

    # run the training command
    vocab_path = os.path.join(bert_path, 'vocab.txt')
    config_path = os.path.join(bert_path, 'bert_config.json')
    checkpoint_path = os.path.join(bert_path, 'bert_model.ckpt')

    cmd = """python code/bert/run_classifier.py \
  --task_name=cola \
  --do_train=True \
  --do_eval=True \
  --do_predict=False \
  --data_dir=%s \
  --vocab_file=%s \
  --bert_config_file=%s \
  --init_checkpoint=%s \
  --max_seq_length=16 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=%s""" % (data_path, vocab_path, config_path, checkpoint_path, model_path)
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python train_classifier.py bert_path data_path model_path")
        exit()
    bert_path = sys.argv[1]
    data_path = sys.argv[2]
    model_path = sys.argv[3]
    train_classifier(bert_path, data_path, model_path)
