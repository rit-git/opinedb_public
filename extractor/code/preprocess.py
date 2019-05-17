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
    text = text.replace("''", "'").replace("\n", ' ').replace("\\n", ' ')
    new_text = ''
    i = 0
    N = len(text)
    while i < len(text):
        curr_chr = text[i]
        new_text += curr_chr
        if i > 0 and i < N - 1:
            next_chr = text[i + 1]
            prev_chr = text[i - 1]
            if next_chr.isalnum() and prev_chr.isalnum() and curr_chr in '!?.,();:':
                new_text += ' '
        i += 1
    return new_text

def has_punct(text):
    if re.match("^[a-zA-Z0-9_ ]*$", text):
        return False
    else:
        return True

def sent_tokenizer(text):
    punct_flag = has_punct(text)
    text = handle_punct(text)
    ori_sentences = []
    for sent in nlp(text, disable=['tagger', 'ner']).sents:
        if len(sent) >= 3:
            ori_sentences.append(sent.text)

    if punct_flag:
        return ori_sentences
    else:
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

def preprocess_tagging(input_path, output_path, review_path):
    # reviews = json.load(open(input_path))
    reviews = []
    with open(input_path, newline='') as csvfile:
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
        if 'extractions' in review:
            review.pop('extractions')

    # convert sentences into tokens and labels
    tokens = []
    labels = []
    for sent in all_sentences:
        # token_list = nltk.word_tokenize(sent)
        token_list = []
        for token in nlp(sent, disable=['parser', 'ner', 'tagger']):
            token_list.append(token.text)
        tokens.append(token_list)
        labels.append(['O' for _ in token_list])

    # print to files
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, 'test.txt')

    with open(output_path, 'w') as f:
        for tlist, llist in zip(tokens, labels):
            for i in range(len(tlist)):
                f.write('%s %s\n' % (tlist[i], llist[i]))
            f.write('\n')

    # print reviews
    with jsonlines.open(review_path, mode='w') as writer:
        for obj in reviews:
            writer.write(obj)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python preprocess.py reviews_csv output_path output_reviews_jsonl")
        exit()
    #schema_json_file = sys.argv[1]
    #review_file = sys.argv[2]
    review_file = sys.argv[1]
    output_path = sys.argv[2]
    output_reviews_path = sys.argv[3]
    preprocess_tagging(review_file, output_path, output_reviews_path)
