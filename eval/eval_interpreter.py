import sys
import json
import gensim
import importlib.util
import numpy as np
import scipy.stats
spec = importlib.util.spec_from_file_location("opine", "opine.py")
opinedb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opinedb)

from gensim.models import Word2Vec

def accuracy(list1, list2):
    N = len(list1)
    TP = sum([list1[i] == list2[i] for i in range(N)])
    return TP / N

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def read_results(fn):
    methods = {'w2v' : [], 'cooc' : [], 'combined' : []}
    for line in open(fn):
        if '  ' in line:
            score = float(line.split('  ')[1])
            for method in methods:
                if method in line:
                    methods[method].append(score)
    for m in methods:
        print(mean_confidence_interval(methods[m]))

# run w2v
def run_w2v():
    simple_opine.clear_cache()
    attributes = []
    for query in queries:
        # never fallback
        attr, phrase = simple_opine.interpret(query, fallback_threshold=-1.0)
        vec1 = simple_opine.phrase2vec(query)
        vec2 = simple_opine.phrase2vec(phrase)
        sim = simple_opine.cosine(vec1, vec2)
        # print('%s\t%s\t%s\t%f' % (query, attr, phrase, sim))
        attributes.append(attr)

    #for attr in attributes:
    #    print(attr)
    print('w2v acc : ', accuracy(attributes, query_groundtruth))

# run cooc
def run_cooc():
    simple_opine.clear_cache()
    attributes = []
    for query in queries:
        attr, phrase = simple_opine.cooc.interpret(query)
        if attr != None:
            vec1 = simple_opine.phrase2vec(query)
            vec2 = simple_opine.phrase2vec(phrase)
            sim = simple_opine.cosine(vec1, vec2)
            # print('%s\t%s\t%s\t%f' % (query, attr, phrase, sim))
        else:
            pass
            # print('%s\t%s' % (query, attr))
        attributes.append(attr)

    #for attr in attributes:
    #    print(attr)
    print('cooc acc : ', accuracy(attributes, query_groundtruth))

# run combined
def run_combined(threshold=0.5):
    simple_opine.clear_cache()
    attributes = []
    for query in queries:
        attr, phrase = simple_opine.interpret(query, fallback_threshold=threshold)
        vec1 = simple_opine.phrase2vec(query)
        vec2 = simple_opine.phrase2vec(phrase)
        sim = simple_opine.cosine(vec1, vec2)
        # print('%s\t%s\t%s\t%f' % (query, attr, phrase, sim))
        attributes.append(attr)

    #for attr in attributes:
    #    print(attr)

    print('combined acc : ', accuracy(attributes, query_groundtruth))

def run_hotel_examples():
    attr, phrase = simple_opine.cooc.interpret("for our anniversary")
    marker = simple_opine.get_marker(attr, phrase)
    print(attr, ':', marker)

    attr, phrase = simple_opine.cooc.interpret("multiple eating options")
    marker = simple_opine.get_marker(attr, phrase)
    print(attr, ':', marker)

    attr, phrase = simple_opine.cooc.interpret("kid friendly hotel")
    marker = simple_opine.get_marker(attr, phrase)
    print(attr, ':', marker)

def run_restaurant_examples():
    attr, phrase = simple_opine.cooc.interpret("dinner with kids")
    marker = simple_opine.get_marker(attr, phrase)
    print(attr, ':', marker)

    attr, phrase = simple_opine.cooc.interpret("close to public transportation")
    marker = simple_opine.get_marker(attr, phrase)
    print(attr, ':', marker)

    attr, phrase = simple_opine.cooc.interpret("private dinner")
    marker = simple_opine.get_marker(attr, phrase)
    print(attr, ':', marker)

def retrain_w2v(all_reviews_fn, word2vec_fn):
    sentences = json.load(open(all_reviews_fn))
    sentences = [gensim.utils.simple_preprocess(s) for s in sentences]
    model = Word2Vec(sentences, size=300)
    model.save(word2vec_fn)

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'read_result':
        fn = sys.argv[2]
        read_results(fn)
    else:
        if sys.argv[2] == 'hotel':
            path = 'data/amsterdam/'
            histogram_fn = path + 'entities_with_histograms.json'
            extraction_fn = path + 'amsterdam_reviews_with_extractions.json'
            sentiment_fn = path + 'sentiment.json'
            word2vec_fn = path + 'word2vec.model'
            all_reviews_fn = path + 'all_reviews.json'
            idf_fn = path + 'idf.json'
            query_label_fn = path + 'labels.json'
            selected_bids = None
            query_path = path + 'hotel_queries.txt'
            query_groundtruth_path = path + 'hotel_query_groundtruth.txt'
            threshold = 0.8
        else:
            path = 'data/toronto/'
            histogram_fn = path + 'jp_entities_with_histograms.json'
            extraction_fn = path + 'jp_restaurant_reviews_with_extractions.json'
            sentiment_fn = path + 'sentiment.json'
            word2vec_fn = path + 'word2vec.model'
            all_reviews_fn = path + 'all_reviews.json'
            idf_fn = path + 'idf.json'
            query_label_fn = path + 'labels.json'
            selected_bids = 'data/raw_jp_restaurants.json'
            query_path = path + 'restaurant_queries.txt'
            query_groundtruth_path = path + 'restaurant_query_groundtruth.txt'
            threshold = 0.8

        queries = open(query_path).read().splitlines()
        query_groundtruth = open(query_groundtruth_path).read().splitlines()
        if mode == 'retrain': # python eval_interpreter hotel retrain_w2v
            rep = 10
            for _ in range(rep):
                word2vec_fn = path + 'word2vec_new.model'
                retrain_w2v(all_reviews_fn, word2vec_fn)

                simple_opine = opinedb.SimpleOpine(histogram_fn, \
                        extraction_fn, sentiment_fn, \
                        word2vec_fn, idf_fn, \
                        query_label_fn, selected_bids)

                # run interpreters
                run_w2v()
                run_cooc()
                run_combined(threshold)
        else: # mode == 'single'
            simple_opine = opinedb.SimpleOpine(histogram_fn, \
                    extraction_fn, sentiment_fn, \
                    word2vec_fn, idf_fn, \
                    query_label_fn, selected_bids)

            # run interpreters
            run_w2v()
            run_cooc()
            run_combined(threshold)
            if sys.argv[1] == 'hotel':
                run_hotel_examples()
            else:
                run_restaurant_examples()
