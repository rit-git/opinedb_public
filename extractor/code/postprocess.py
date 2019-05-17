import json
import jsonlines
import os
import sys

def postprocess(review_path, sentence_path, classifier_path, labels_path, output_path):
    label_mp = json.load(open(labels_path))
    labels = [0] * len(label_mp)
    for key in label_mp:
        labels[label_mp[key]] = key

    classifier_results = []
    for lines in open(classifier_path):
        probs = lines.split('\t')
        probs = [float(x) for x in probs]
        label_id = 0
        for i in range(len(probs)):
            if probs[i] > probs[label_id]:
                label_id = i
        classifier_results.append(labels[label_id])

    sentences = json.load(open(sentence_path))
    for sentence in sentences:
        for ext in sentence['extractions']:
            eid = ext['eid']
            ext['negation'] = False
            ext['attribute'] = classifier_results[eid]
            ext.pop('eid')

    reviews = []
    with jsonlines.open(review_path) as reader:
        for review in reader:
            review['extractions'] = []
            for sid in review['sentence_ids']:
                review['extractions'] += sentences[sid]['extractions']
            review.pop('sentence_ids')
            reviews.append(review)

    json.dump(reviews, open(output_path, 'w'))


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python postprocess.py reviews_jsonl sentences_path classifier_output_path labels_path output_path")
        exit()
    #schema_json_file = sys.argv[1]
    #review_file = sys.argv[2]
    review_path = sys.argv[1]
    sentence_path = sys.argv[2]
    classifier_path = sys.argv[3]
    labels_path = sys.argv[4]
    output_path = sys.argv[5]
    postprocess(review_path, sentence_path, classifier_path, labels_path, output_path)
