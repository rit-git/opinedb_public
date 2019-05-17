import json
import jsonlines
import csv
import os
import sys
import spacy
import re

common_words = open('data/google-10000-english-no-swears.txt').read().splitlines()
common_words = set(common_words)
nlp = spacy.load('en_core_web_sm')

def handle_punct(text):
    # for the trustyou dataset
    text = text.replace("''", "'")
    new_text = ''
    i = 0
    N = len(text)
    while i < len(text):
        curr_chr = text[i]
        new_text += curr_chr
        if i > 0 and i < N - 1:
            next_chr = text[i + 1]
            prev_chr = text[i - 1]
            if next_chr.isalnum() and prev_chr.isalnum() and curr_chr in '.,?()!':
                new_text += ' '
        i += 1
    return new_text

def has_punct(text):
    if re.match("^[a-zA-Z0-9_ ]*$", text):
        return False
    else:
        return True

def sent_tokenizer(text):
    if text == '':
        return []
    # ori_sentences = nltk.sent_tokenize(text)
    punct_flag = has_punct(text)
    text = handle_punct(text)
    ori_sentences = []
    for sent in nlp(text, disable=['tagger', 'ner']).sents:
        text = sent.text.strip()
        if len(text) > 5 and '\n' not in text:
            ori_sentences.append(sent.text)

    if punct_flag:
        return ori_sentences

    # for the booking.com datasets
    result = []
    for ori_sentence in ori_sentences:
        sentences = [[]]
        for token in ori_sentence.split(' '):
            if len(token) > 0 and token[0].isupper() and token.lower() in common_words and (not len(sentences[-1]) <= 1):
                sentences.append([])
            sentences[-1].append(token)
        result += [' '.join(line) for line in sentences if len(line) > 0]
    return result


def mine_tips(review_path, bert_path, model_path, output_path):
    review_fn = os.path.join(review_path, 'raw_reviews.csv')
    reviews = []
    with open(review_fn, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            reviews.append(row)

    all_sentences = []
    for review in reviews:
        if 'text' in review:
            text = review['text']
        else:
            text = review['review']
        sentences = sent_tokenizer(text)
        sentence_ids = []
        for sent in sentences:
            sentence_ids.append(len(all_sentences))
            all_sentences.append(sent)
        review['sentence_ids'] = sentence_ids

    # write sentences
    data_path = os.path.join(review_path, 'sentences')
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    sentence_path = os.path.join(review_path, 'sentences/test.tsv')
    with open(sentence_path, 'w') as fout:
        fout.write('index\tsentence\n')
        for (sid, sent) in enumerate(all_sentences):
            fout.write('%d\t%s\n' % (sid, sent))

    # do prediction with model
    vocab_path = os.path.join(bert_path, 'vocab.txt')
    config_path = os.path.join(bert_path, 'bert_config.json')
    checkpoint_path = os.path.join(bert_path, 'bert_model.ckpt')
    cmd = """python code/bert/run_classifier.py \
  --task_name=cola \
  --do_train=False \
  --do_eval=False \
  --do_predict=True \
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

    # read results
    output_fn = os.path.join(model_path, 'test_results.tsv')
    tip_sid_set = set([])
    with open(output_fn) as fin:
        sid = 0
        for line in fin:
            p0, p1 = [float(x) for x in line.split('\t')]
            if p1 > p0:
                tip_sid_set.add(sid)
            sid += 1

    for review in reviews:
        review['tips'] = []
        for sid in review['sentence_ids']:
            if sid in tip_sid_set:
                review['tips'].append(all_sentences[sid])
        review.pop('sentence_ids')

    # output
    json.dump(reviews, open(output_path, 'w'))


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python mine_tips.py reviews_path bert_path model_path output_path")
        exit()
    review_path = sys.argv[1]
    bert_path = sys.argv[2]
    model_path = sys.argv[3]
    output_path = sys.argv[4]
    mine_tips(review_path, bert_path, model_path, output_path)
