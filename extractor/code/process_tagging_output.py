import json
import jsonlines
import collections
import os
import sys
import spacy

from nltk.corpus import stopwords

nlp = spacy.load('en_core_web_sm')

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

    sentence = ' '.join(tokens)
    doc = nlp(sentence)
    if len(doc) != len(tokens): # this needs to be fixed
        print(doc)
        return []

    # print(sentence)
    graph = [[] for _ in tokens]
    for token in doc:
        u = token.i
        for child in token.children:
            v = child.i
            graph[u].append(v)
            graph[v].append(u)
    # print(graph)

    def get_dist(start, end):
        N = len(graph)
        dist = [-1] * N
        dist[start] = 0
        queue = collections.deque()
        queue.append(start)
        while len(queue) > 0:
            if dist[end] >= 0:
                break
            u = queue.popleft()
            for v in graph[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist[end]

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

    # get pair-wise distances
    distances = []
    for aid in range(len(aspects)):
        for oid in range(len(opinions)):
            dist = get_dist(aspects[aid][0], opinions[oid][0])
            if dist >= 0:
                distances.append((dist, aid, oid))
    distances.sort()
    used_aspects = set([])
    used_opinions = set([])
    results = []

    def add_to_results(results, aid, oid):
        start_as, end_as = aspects[aid]
        start_op, end_op = opinions[oid]
        opinion_term = doc[start_op : end_op + 1].text
        aspect_term = doc[start_as : end_as + 1].text
        results.append({'entity' : aspect_term, 'predicate' : opinion_term })

    # unique match first
    for (dist, aid, oid) in distances:
        if aid not in used_aspects and oid not in used_opinions:
            used_aspects.add(aid)
            used_opinions.add(oid)
            add_to_results(results, aid, oid)

    # handle the unmatched terms first
    for (dist, aid, oid) in distances:
        if aid not in used_aspects or oid not in used_opinions:
            used_aspects.add(aid)
            used_opinions.add(oid)
            add_to_results(results, aid, oid)
    return results


def process_tagging_output(tagging_file, sentence_path, class_prediction_path):
    ori_tokens, ori_labels = read_tagging_file(tagging_file)
    sentence_extraction = []

    if not os.path.exists(class_prediction_path):
        os.makedirs(class_prediction_path)
    file_path = os.path.join(class_prediction_path, 'test.tsv')
    with open(file_path, 'w') as fout:
        fout.write('index\tsentence\n')
        sid = 0
        ext_id = 0
        for t, l in zip(ori_tokens, ori_labels):
            candidates = combine_aspect_opinion_terms(t, l)
            sentence = ' '.join(t)
            sentence_extraction.append({'sid' : sid, 'sentence' : sentence, 'extractions' : []})

            # print(sentence)
            for cand in candidates:
                cand['eid'] = ext_id
                sentence_extraction[sid]['extractions'].append(cand)
                fout.write('%d\t%s\n' % (ext_id, cand['predicate'] + ' ' + cand['entity']))
                ext_id += 1
            sid += 1
    json.dump(sentence_extraction, open(sentence_path, 'w'))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python process_tagging_output.py tagging_file sentence_path class_prediction_path")
        exit()
    tagging_file = sys.argv[1]
    sentence_path = sys.argv[2]
    class_prediction_path = sys.argv[3]
    process_tagging_output(tagging_file, sentence_path, class_prediction_path)
