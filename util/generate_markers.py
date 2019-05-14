import sys
import os
import json
import csv
import gensim
import spacy
import numpy as np
import re
import pickle
import math

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KDTree

from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# this can be replace with spacy's tokenizer, but much slower
from nltk import sent_tokenize

from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from scipy import spatial

SIA = SentimentIntensityAnalyzer()

def load_raw_reviews(raw_review_fn):
    raw_reviews = []
    with open(raw_review_fn, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if 'business_id' not in row:
                row['business_id'] = row['entity_id']
            if 'text' not in row:
                row['text'] = row['review']
                del row['review']
            raw_reviews.append(row)
    return raw_reviews

def load_extraction(extraction_fn):
    reviews = json.load(open(extraction_fn))
    for row in reviews:
        if 'text' not in row:
            row['text'] = row['review']
            del row['review']

    reviews = {review['review_id'] : review for review in reviews}
    return reviews

def extraction_to_phrase(extraction):
    attr = extraction['attribute']
    phrase = extraction['predicate'] + '-' + extraction['entity']
    phrase = phrase.replace(' ', '-')
    return attr.lower(), phrase.lower()

def generate_histograms(raw_reviews, reviews):
    """
    Convert the extraction results to a 'histogram' field in an entity object.
    It is written in this way simply because the business_id is not recorded
    in the extraction result.

    Args:
        raw_reviews (List): a list of raw reviews with the fields 'business_id' and 'review_id'
        reviews (Dict): mapping from review_id to extraction results
    Returns:
        Dict: a mapping from business_id to entity objects with the 'histogram',
              'name', and the 'reviews' field
    """
    entities = {}
    for review in raw_reviews:
        bid = review['business_id']
        rid = review['review_id']
        if bid not in entities:
            entities[bid] = {'histogram' : {}, 'name' : bid, 'reviews' : []}
        entities[bid]['reviews'].append(rid)
        extractions = reviews[rid]['extractions']
        for ext in extractions:
            attr, phrase = extraction_to_phrase(ext)
            if attr not in entities[bid]['histogram']:
                entities[bid]['histogram'][attr] = { phrase : 1}
            else:
                if phrase not in entities[bid]['histogram'][attr]:
                    entities[bid]['histogram'][attr][phrase] = 1
                else:
                    entities[bid]['histogram'][attr][phrase] += 1
    return entities


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

# sentiment analysis
def compute_phrase_sentiments(reviews):
    """
    Compute the phrase-level sentiment score with normalization.

    Args:
        reviews (Dict): a dictonary of reviews with the field 'extractions' and 'text'/'review'
    Returns:
        Dict: a mapping from phrases to sentiment scores
    """
    phrase_mp = {}
    sent_sum = 0.0
    sent_total = 0.0
    for rid in reviews:
        review = reviews[rid]

        if len(review['extractions']) == 0:
            continue
        if 'text' in review:
            text = handle_punct(review['text'])
        else:
            text = handle_punct(review['review'])

        sentences = []
        for sent in sent_tokenize(text):
            score = SIA.polarity_scores(sent)['compound']
            # score = TextBlob(sent).sentiment.polarity
            sentences.append((sent, score))
            sent_sum += score
            sent_total += 1

        # review['sentiment'] = review_total_score / review_sent_cnt
        for ext in review['extractions']:
            _, phrase = extraction_to_phrase(ext)
            entity = ext['entity']
            predicate = ext['predicate']
            found = False
            for (sid, (sent, score)) in enumerate(sentences):
                if entity in sent and predicate in sent:
                    # put the review sentence in the extraction
                    found = True
                    for close_sid in range(sid - 2, sid + 3):
                        # the sentiment score is computed with a window size of 5
                        if close_sid >= 0 and close_sid < len(sentences):
                            if phrase not in phrase_mp:
                                phrase_mp[phrase] = [0, 0]
                            _, score = sentences[close_sid]
                            phrase_mp[phrase][0] += score
                            phrase_mp[phrase][1] += 1
            if not found:
                if phrase not in phrase_mp:
                    phrase_mp[phrase] = [0, 0]
                for _, score in sentences:
                    phrase_mp[phrase][0] += score
                    phrase_mp[phrase][1] += 1


    sent_mean = sent_sum / sent_total
    for phrase in phrase_mp:
        phrase_mp[phrase] = phrase_mp[phrase][0] / phrase_mp[phrase][1] - sent_mean # normalized
    return phrase_mp


phrase2vec_cache = {}
def phrase2vec(phrase, positive_threshold=-0.1):
    """
    Computer the vector representation of a phrase. This version simply sums up all
    the word vector of the phrase.

    Args:
        phrase (str): a phrase seperated by space of dash
        positive_threshold (float): the sentiment threshold for positive phrase (dataset dependant)
    Returns:
        numpy.array: a 300d array of the vector representation
    """
    phrase = phrase.lower().replace('-', ' ')
    if phrase in phrase2vec_cache:
        return phrase2vec_cache[phrase]

    words = gensim.utils.simple_preprocess(phrase)
    res = np.zeros(300)
    # sum pooling
    for w in words:
        if w in model.wv:
            v = model.wv[w]
            res += v * idf[w]

    # separate positive and negative phrases
    # TODO: replace or remove it if there is a better way to handle
    # negation in word2vec
    if phrase in phrase_sentiments and phrase_sentiments[phrase] < positive_threshold:
        res = -res

    # normalize
    norm = np.linalg.norm(res)
    if norm > 0:
        res /= norm
    phrase2vec_cache[phrase] = res
    return res

def cosine(vec1, vec2):
    return 1.0 - spatial.distance.cosine(vec1, vec2)


stopword_set = set(stopwords.words('english'))

def clean_marker(phrase):
    phrase = phrase.lower().replace('-', ' ')
    terms = phrase.split(' ')
    terms = [t for t in terms if len(t) > 0]
    while len(terms) > 0 and terms[-1].lower() in stopword_set:
        terms.pop()
    return ' '.join(terms)

def verbalize(reviews):
    nlp = spacy.load('en_core_web_sm')
    phrase_set = set([])
    for _, review in reviews.items():
        for ext in review['extractions']:
            _, phrase = extraction_to_phrase(ext)
            phrase = clean_marker(phrase)
            phrase_set.add(phrase)
        counter = Counter()

    max_len = max([len(phrase.split(' ')) for phrase in phrase_set])
    for _, review in reviews.items():
        text = review['text']
        tokens = []
        for token in nlp(text, disable=['parser', 'tagger']):
            tokens.append(token.text.lower())

        for i in range(len(tokens)):
            for l in range(max_len):
                if i + l >= len(tokens):
                    break
                phrase = ' '.join(tokens[i:i+l])
                if phrase in phrase_set:
                    counter[phrase] += 1

    results = set([])
    for phrase, cnt in counter.most_common():
        if cnt >= 10 or (cnt >= 2 and len(results) <= 750): # the second part is for attractions
            results.add(phrase)
    return results


def construct_marker_summaries(entities, verbalize_set):

    class NN_index:
        def __init__(self, verbalize_set):
            self.all_phrases = list(verbalize_set)
            all_vectors = [phrase2vec(phrase) for phrase in self.all_phrases]
            self.kd_tree = KDTree(all_vectors, leaf_size=40)

        def query(self, phrase):
            vector = phrase2vec(phrase)
            phrase_id = self.kd_tree.query([vector], k=1)[1][0][0]
            return self.all_phrases[phrase_id]

    # build a nearest neighbor index for verbalization
    nn_index = NN_index(verbalize_set)

    """
    Construct the marker summaries by k-means clustering. The 'summary' field will be added
    to each entity. Each summary contains the center phrase, the size of the cluster,
    the sum of sentiment scores, and the center vector (average of the phrase vectors).

    Args:
        entities (Dict): a dictionary of entities (map business_ids to entities)
        verbalize_set (Set of str): a set of verbalized phrases
    Returns:
        None
    """
    def clustering(bid, histogram, num_bars=10):
        preds = list(histogram.keys())
        X = [phrase2vec(p) for p in preds]
        if len(preds) < num_bars:
            centers = [phrase2vec(phrase) for phrase in preds]
            labels = list(range(len(preds)))
        else:
            kmeans = KMeans(n_clusters=num_bars, random_state=0).fit(X)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
        # find nearest neighbor of each center
        markers = []
        for c in centers:
            markers.append({'size': 0,
                            'center': list(c),
                            'sum_senti': 0.0})

        for i in range(len(preds)):
            cid = labels[i]
            # verbalize
            phrase = preds[i]
            verbalized = clean_marker(phrase)
            is_new_verbalized = (verbalized in verbalize_set)

            flag = False
            if 'phrase' not in markers[cid]:
                flag = True
            else:
                is_old_verbalized = (markers[cid]['verbalized'] in verbalize_set)
                if is_new_verbalized != is_old_verbalized:
                    flag = is_new_verbalized
                else:
                    flag = (histogram[phrase] > histogram[markers[cid]['phrase']])

            if flag:
                markers[cid]['phrase'] = phrase
                if is_new_verbalized:
                    markers[cid]['verbalized'] = verbalized
                else:
                    markers[cid]['verbalized'] = nn_index.query(verbalized)

                markers[cid]['is_verbalized'] = is_new_verbalized
            markers[cid]['size'] += histogram[phrase]
            markers[cid]['sum_senti'] += phrase_sentiments[phrase] * histogram[phrase]

        # remove markers of size 0
        markers = [marker for marker in markers if 'phrase' in marker]
        return markers

    # perform the clustering
    for bid in entities:
        histograms = entities[bid]['histogram']
        entities[bid]['summaries'] = {}
        for attr in histograms:
            entities[bid]['summaries'][attr] = clustering(bid, histograms[attr], num_bars=10)
    return entities


def compute_marker_snippet(entities, reviews):
    """
    Select a review sentence for each marker in each entity.
    It fills in the 'snippet' field for each marker

    Args:
        entities (Dict): mapping from business_id's to entities
        reviews (Dict): reviews with extractions, a mapping from review_id's to reviews with the 'text' and 'extractions' field
    Returns:
        None
    """
    review_text_per_bid = {}
    for (_, review) in reviews.items():
        if 'business_id' not in review:
            review['business_id'] = review['entity_id']
        bid = review['business_id']
        if len(review['extractions']) == 0:
            continue
        sentences = sent_tokenize(handle_punct(review['text']))
        for ext in review['extractions']:
            _, phrase = extraction_to_phrase(ext)
            entity = ext['entity']
            predicate = ext['predicate']
            found = False
            for sent in sentences:
                if entity in sent and predicate in sent:
                    sent = sent.replace(entity, "<strong> %s </strong>" % entity)
                    sent = sent.replace(predicate, "<strong> %s </strong>" % predicate)
                    if (bid, phrase) not in review_text_per_bid:
                        review_text_per_bid[(bid, phrase)] = []
                    review_text_per_bid[(bid, phrase)].append(sent)
                    found = True
            if not found:
                if (bid, phrase) not in review_text_per_bid:
                    review_text_per_bid[(bid, phrase)] = []
                review_text_per_bid[(bid, phrase)].append(phrase)

    SIA = SentimentIntensityAnalyzer()
    for (bid, entity) in entities.items():
        summaries = entity['summaries']
        for (attr, markers) in summaries.items():
            for marker in markers:
                if 'phrase' not in marker:
                    print(marker)
                phrase = marker['phrase']
                sentences = review_text_per_bid[(bid, phrase)]
                best_sent = sentences[0]
                best_score = SIA.polarity_scores(best_sent)['compound']
                # best_score = TextBlob(best_sent).sentiment.polarity
                for sent in sentences:
                    new_score = SIA.polarity_scores(sent)['compound']
                    # new_score = TextBlob(sent).sentiment.polarity

                    if new_score > best_score:
                        best_score = new_score
                        best_sent = sent
                marker['snippet'] = best_sent


# auto labeling with rules
def positive_filter(histogram,
                    query,
                    positive_sentiment_threshold=-0.1,
                    similarity_threshold=0.6,
                    sim_count_threshold=15,
                    sim_fraction_threshold=0.1,
                    sentiment_count_threshold=5,
                    sentiment_frac_threshold=0.8):
    """
    Generate a pseudo-label for a histogram-query pair. This is used when human labels
    are too expensive to obtain. The labeling rule contains two part: (1) similarity and
    (2) sentiment. A pair is marked as positive if there are many similar phrases to the query,
    or there are enough similar phrases and the majority is positive.

    Args:
        histogram (Dict): the histogram to be labeled
        query (string): the query predicate
        positive_sentiment_threshold (float): the threshold for positive sentiment
        similarity_threshold (float): similarity threshold
        sim_count_threshold (float): the minimal count of similar phrases
        sim_fraction_threshold (float): the minimal fraction of similar phrases
        sentiment_count_threshold (float): the minimal count of similar phrases for the sentiment rule
        sentiment_frac_threshold (float): the minimal fraction of positive phrases
    Returns:
        Boolean: whether the histogram is a match with the query or not
        int: the number of matched phrases
    """
    # words = query.lower().split(' ')
    pos_cnt = 0
    total = 0
    match = 0
    query_vec = phrase2vec(query)
    for phrase in histogram:
        total += histogram[phrase]
        if phrase_sentiments[phrase] > positive_sentiment_threshold:
            pos_cnt += histogram[phrase]

        phrase_vec = phrase2vec(phrase)
        if cosine(query_vec, phrase_vec) >= similarity_threshold:
            match += histogram[phrase]
    # print(query, match, total, match / total)
    if (match >= sim_count_threshold or \
        match / total >= sim_fraction_threshold) or \
       (match >= sentiment_count_threshold and \
       pos_cnt / total >= sentiment_frac_threshold):
        return (True, match)
    return (False, match)


def generate_pseudo_labels(entities, queries_fn, output_fn):
    """
    Apply the labeling rules to generate pseudo labels.

    Args:
        entities (Dict): a dictionary of entities with the histogram field computed
        queries_fn (string): the path to the query strings
        output_fn (string): the output path
    Returns:
        None
    """
    queries = open(queries_fn).read().splitlines()
    # find all attributes
    ids = list(entities.keys())
    hist_data = []
    labels = []

    for (i, query) in enumerate(queries):
        # attr_name, ids = candidates[query]
        num_match = 0
        for bid in ids:
            found_match = False
            match_attr = ''
            max_matched_phrases = -1
            histogram = entities[bid]['histogram']
            for attr_name in histogram:
                filter_flag, matched_phrases = positive_filter(histogram[attr_name], query)
                if filter_flag:
                    found_match = True
                    if matched_phrases > max_matched_phrases:
                        matched_phrases = max_matched_phrases
                        match_attr = attr_name
            if found_match:
                hist_data.append((bid, match_attr, query))
                labels.append('yes')
                num_match += 1
            else:
                hist_data.append((bid, match_attr, query))
                labels.append('no')
        print(query, num_match, len(ids))

    # output the dataset
    dataset = []
    for (entity_id, attr_name, query), label in zip(hist_data, labels):
        dataset.append((entity_id, attr_name, query, label))
    json.dump(dataset, open(output_fn, 'w'))


def train_or_load_w2v_model(word2vec_fn, idf_fn, all_reviews_fn):
    """
    Train a word2vec model, or load it from a file if it already exists.

    Args:
        word2vec_fn (str): the path to the w2v model
        all_reviews_fn (str): the path to the json file of the list of reviews
    Returns:
        Word2Vec: a (gensim) word2vec model
    """
    if os.path.exists(word2vec_fn) and os.path.exists(idf_fn):
        model = Word2Vec.load(word2vec_fn)
        idf = json.load(open(idf_fn))
    else:
        import math
        sentences = json.load(open(all_reviews_fn))
        sentences = [gensim.utils.simple_preprocess(s) for s in sentences]
        # compute document frequency
        df = {}
        for sent in sentences:
            tokens = set(sent)
            for token in tokens:
                if token not in df:
                    df[token] = 1
                else:
                    df[token] += 1
        N = len(sentences)
        idf = dict([(token, math.log(N / df[token])) for token in df])
        model = Word2Vec(sentences, size=300)
        model.save(word2vec_fn)
        json.dump(idf, open(idf_fn, 'w'))
    return model, idf


def merge_original_entities(entities, ori_entity_fn):
    ori_entities = json.load(open(ori_entity_fn))
    for entity in ori_entities:
        if 'business_id' not in entity:
            entity['business_id'] = entity['entity_id']
        if 'gps' in entity and ('longitude' not in entity or 'latitude' not in entity):
            entity['latitude'], entity['longitude'] = entity['gps']

        bid = entity['business_id']
        if bid not in entities:
            print(bid)
        else:
            for key in entity:
                entities[bid][key] = entity[key]

if __name__ == "__main__":
    #if len(sys.argv) < 3:
    #    print("Usage: python calculate_histograms_and_sentiment.py entities_file reviews_with_extractions_file w2v_model_file photos_metadata_file\
    #                entities_target reviews_target senti_mp_phrase_target")
    #    exit()

    if len(sys.argv) < 11:
        print("Usage: python generate_markers.py entity_fn raw_review_fn extraction_fn all_reviews_fn word2vec_fn idf_fn sentiment_output_fn histogram_fn queries_fn labels_fn")
        data_path = 'test_pipeline/'
        entity_fn = data_path + 'google_sf_restaurant.json'
        raw_review_fn = data_path + 'raw_reviews.csv'
        extraction_fn = data_path + 'sf_restaurant_reviews_with_extractions.json'
        histogram_fn = data_path + 'sf_restaurants_with_histograms.json'
        sentiment_output_fn = data_path + 'sf_restaurant_sentiment.json'
        word2vec_fn = data_path + 'word2vec.model'
        idf_fn = data_path + 'idf.json'
        all_reviews_fn = data_path + 'all_reviews.json'
        labels_fn = data_path + 'restaurant_labels.json'
        queries_fn = data_path + 'restaurant_queries.txt'
    else:
        entity_fn = sys.argv[1]
        raw_review_fn = sys.argv[2]
        extraction_fn = sys.argv[3]
        all_reviews_fn = sys.argv[4]
        word2vec_fn = sys.argv[5]
        idf_fn = sys.argv[6]
        sentiment_output_fn = sys.argv[7]
        histogram_fn = sys.argv[8]
        queries_fn = sys.argv[9]
        labels_fn = sys.argv[10]

    # load the extraction results into an entity dictionary
    print('loading extraction results')
    raw_reviews = load_raw_reviews(raw_review_fn)
    reviews_with_extraction = load_extraction(extraction_fn)
    entities = generate_histograms(raw_reviews, reviews_with_extraction)

    # compute the phrase-level sentiment score
    print('computing phrase sentiment scores')
    if os.path.exists(sentiment_output_fn):
        phrase_sentiments = json.load(open(sentiment_output_fn))
    else:
        phrase_sentiments = compute_phrase_sentiments(reviews_with_extraction)
        json.dump(phrase_sentiments, open(sentiment_output_fn, 'w'))

    # train or load the w2v model
    print('loading/training word2vec model')
    model, idf = train_or_load_w2v_model(word2vec_fn, idf_fn, all_reviews_fn)

    # compute the verbalize set
    verbalized_set_fn = 'verbalized_set.json'
    if os.path.exists(verbalized_set_fn):
        verbalize_set = set(json.load(open(verbalized_set_fn)))
    else:
        print('computing verbalization mapping')
        verbalize_set = verbalize(reviews_with_extraction)
        json.dump(list(verbalize_set), open(verbalized_set_fn, 'w'))

    # compute the marker summaries by clustering
    print('computing marker summaries')
    entities = construct_marker_summaries(entities, verbalize_set)

    # compute the marker snippet from reviews and summaries
    print('computing marker snippets')
    compute_marker_snippet(entities, reviews_with_extraction)

    # merge with original entity attributes
    # print('merging with original entities')
    # merge_original_entities(entities, entity_fn)

    # output the entities index
    print('dummping results')
    json.dump(entities, open(histogram_fn, 'w'))

    # generate the pseudo labels using rules
    # TODO: since this step involves some parameter tuning,
    # I recommand running it separately.
    if not os.path.exists(labels_fn):
        print('generating pseudo labels')
        generate_pseudo_labels(entities, queries_fn, labels_fn)

