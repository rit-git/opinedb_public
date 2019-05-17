import json
import jsonlines
import collections
import os
import sys

from nltk.corpus import stopwords

stopword_set =  set(stopwords.words('english'))

def is_stopword(token):
    return token in stopword_set or not token.isalpha()

def read_tagging_file(fn):
    tokens = [[]]
    labels = [[]]
    for line in open(fn):
        if len(line) < 3:
            tokens.append([])
            labels.append([])
        else:
            LL = line.strip().split(' ')
            token = LL[0]
            label = LL[-1]
            tokens[-1].append(token)
            labels[-1].append(label)
    return tokens, labels

def combine_aspect_opinion_terms(tokens, labels):
    if not 'B-AS' in labels or not 'B-OP' in labels:
        return []

    aspects = []
    opinions = []

    i = 0
    while i < len(tokens):
        if labels[i].startswith('B-'):
            start = i
            end = i
            expected = labels[i].replace('B-', 'I-')
            current_tokens = [tokens[i].lower()]

            while i + 1 < len(tokens) and labels[i + 1] == expected:
                end += 1
                i += 1
                current_tokens.append(tokens[i].lower())

            # skip the term if it is obviously wrong
            if not all([is_stopword(token) for token in current_tokens]):
                if 'AS' in expected:
                    aspects.append((start, end))
                else:
                    opinions.append((start, end))
        i += 1

    candidates = []
    used = set([])
    for (as_start, as_end) in aspects:
        for sign in [True, False]:
            op_start, op_end = None, None
            min_dist = 1e60
            for (start, end) in opinions:
                if (as_start < start) == sign and abs(as_start - start) < min_dist:
                    min_dist = abs(as_start - start)
                    op_start, op_end = start, end
            if op_start != None:
                phrase = ' '.join(tokens[op_start: op_end + 1] + tokens[as_start: as_end + 1]).lower()
                # remove duplicate
                if phrase not in used:
                    used.add(phrase)
                    candidates.append((' '.join(tokens[as_start: as_end + 1]),\
                        ' '.join(tokens[op_start: op_end + 1])))
    return candidates

def pairing(tagging_file, sentence_path, pairing_data_path, pairing_model_path, class_prediction_path, bert_path):
    ori_tokens, ori_labels = read_tagging_file(tagging_file)

    # create pairing data
    if not os.path.exists(pairing_data_path):
        os.makedirs(pairing_data_path)

    pairing_test_fn = os.path.join(pairing_data_path, 'test.tsv')
    with open(pairing_test_fn, 'w') as fout:
        fout.write('sid\taspect\topinion\tsentence\textraction\n')
        sid = 0
        for t, l in zip(ori_tokens, ori_labels):
            candidates = combine_aspect_opinion_terms(t, l)
            sentence = ' '.join(t)
            for aspect, opinion in candidates:
                fout.write('%d\t%s\t%s\t%s\t%s\n' % (sid, aspect, opinion, sentence, opinion + ' ' + aspect))
            sid += 1

    # run the pairing classifier
    # bert_path = '../../../uncased_L-12_H-768_A-12'
    cmd = """python code/bert/run_classifier.py \
        --task_name=mrpc \
        --do_train=False \
        --do_eval=False \
        --do_predict=True \
        --data_dir=%s \
        --vocab_file=%s/vocab.txt \
        --bert_config_file=%s/bert_config.json \
        --init_checkpoint=%s/bert_model.ckpt \
        --max_seq_length=128 \
        --train_batch_size=16 \
        --learning_rate=2e-5 \
        --num_train_epochs=10.0 \
        --output_dir=%s""" % (pairing_data_path, \
        bert_path, bert_path, bert_path, pairing_model_path)
    os.system(cmd)

    # read results
    sentence_extraction = []
    for (sid, tokens) in enumerate(ori_tokens):
        sentence_extraction.append({'sid': sid, \
            'sentence': ' '.join(tokens), \
            'extractions': []})

    pairing_results = open(os.path.join(pairing_model_path, 'test_results.tsv')).readlines()
    with open(pairing_test_fn) as fin:
        fin.readline()
        line_id = 0
        ext_id = 0
        for line in fin:
            sid, aspect, opinion, sentence, _ = line.strip().split('\t')
            sid = int(sid)
            p0, p1 = pairing_results[line_id].strip().split('\t')
            p0 = float(p0)
            p1 = float(p1)

            if p1 > p0:
                cand = {'eid' : ext_id, 'negation' : False,
                        'entity' : aspect, 'predicate' : opinion}
                sentence_extraction[sid]['extractions'].append(cand)
                ext_id += 1
            line_id += 1

    """
    # perform max bipartite matching (tried, but resulted in lower recall)
    import networkx as nx
    ext_id = 0
    for sent_ext in sentence_extraction:
        if len(sent_ext['extractions']) == 0:
            continue
        edges = []
        entities = set([])
        predicates = set([])
        for ext in sent_ext['extractions']:
            entities.add(ext['entity'])
            if ext['predicate'] not in entities:
                edges.append((ext['predicate'], ext['entity']))

        new_extractions = []
        if len(edges) > 0:
            G = nx.Graph()
            G.add_edges_from(edges)
            matching = nx.bipartite.maximum_matching(G, list(entities))
            for (opinion, aspect) in matching.items():
                if aspect in entities:
                    new_extractions.append({'eid': ext_id, 'negation': False,
                        'entity': aspect, 'predicate': opinion})
                    ext_id += 1
        sent_ext['extractions'] = new_extractions
    """
    json.dump(sentence_extraction, open(sentence_path, 'w'))

    # print files to run classification
    if not os.path.exists(class_prediction_path):
        os.makedirs(class_prediction_path)
    file_path = os.path.join(class_prediction_path, 'test.tsv')
    with open(file_path, 'w') as fout:
        fout.write('index\tsentence\n')
        for sent_ext in sentence_extraction:
            for cand in sent_ext['extractions']:
                ext_id = cand['eid']
                fout.write('%d\t%s\n' % (ext_id, cand['predicate'] + ' ' + cand['entity']))


if __name__ == '__main__':
    if len(sys.argv) < 7:
        print("Usage: python pairing.py tagging_file sentence_path pairing_data_path pairing_model_path class_prediction_path bert_path")
        exit()
    tagging_file = sys.argv[1]
    sentence_path = sys.argv[2]
    pairing_data_path = sys.argv[3]
    pairing_model_path = sys.argv[4]
    class_prediction_path = sys.argv[5]
    bert_path = sys.argv[6]
    pairing(tagging_file, sentence_path, pairing_data_path, pairing_model_path, class_prediction_path, bert_path)
