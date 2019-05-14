import sys
import os
import json
import gensim
import math
import random
import numpy as np

from gensim.models import Word2Vec
from gensim.summarization.bm25 import get_bm25_weights, BM25

from scipy import spatial
from sklearn.neighbors import KDTree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class CooccurInterpreter:
    def __init__(self, reviews):
        reviews = { review['review_id'] : review for review in reviews }
        self.reviews = reviews
        self.review_ids = []
        self.interpret_cache = {}
        self.position_index = {}
        self.idf = {}

        def build_index():
            # build bm25 index
            corpus = []
            total = 0.0
            SIA = SentimentIntensityAnalyzer()
            for rid in reviews:
                self.review_ids.append(rid)
                if 'text' in reviews[rid]:
                    sent = reviews[rid]['text']
                else:
                    sent = reviews[rid]['review']

                tokens = gensim.utils.simple_preprocess(sent.lower())
                reviews[rid]['sentiment'] = SIA.polarity_scores(sent)['compound']

                corpus.append(tokens)
                self.position_index[rid] = {}
                for (pos, token) in enumerate(tokens):
                    if token not in self.position_index[rid]:
                        self.position_index[rid][token] = [pos]
                    else:
                        self.position_index[rid][token].append(pos)

                for ext in reviews[rid]['extractions']:
                    attr = ext['attribute']
                    if attr not in self.idf:
                        self.idf[attr] = 1
                    else:
                        self.idf[attr] += 1
                    total += 1
            bm25 = BM25(corpus)
            for attr in self.idf:
                self.idf[attr] = math.log2(total / self.idf[attr])
            return bm25
        self.bm25 = build_index()


    def get_dist(self, position_index, phrase1, phrase2):
        tokens1 = gensim.utils.simple_preprocess(phrase1.lower())
        tokens2 = gensim.utils.simple_preprocess(phrase2.lower())

        positions1 = []
        for token1 in tokens1:
            if token1 in position_index:
                positions1 += position_index[token1]

        positions2 = []
        for token2 in tokens2:
            if token2 in position_index:
                positions2 += position_index[token2]

        result = -1
        for p1 in positions1:
            for p2 in positions2:
                if result < 0 or abs(p1 - p2) < result:
                    result = abs(p1 - p2)
        return result


    def interpret(self, qterm, debug=False):
        if qterm in self.interpret_cache:
            return self.interpret_cache[qterm]

        scores = self.bm25.get_scores(gensim.utils.simple_preprocess(qterm))
        # scores = self.bm25.get_scores(gensim.utils.simple_preprocess(qterm), 0.5)
        score_mp = {}
        for (i, rid) in enumerate(self.review_ids):
            if scores[i] > 0:
                score_mp[rid] = scores[i] * self.reviews[rid]['sentiment']
            else:
                score_mp[rid] = 0.0

        sorted_review_ids = sorted(self.review_ids, key=lambda x : -score_mp[x])
        attribute_scores = {}
        represented_phrases = {}

        for rid in sorted_review_ids[:10]:
            if score_mp[rid] <= 0:
                continue
            extractions = self.reviews[rid]['extractions']
            if debug:
                if 'text' in self.reviews[rid]:
                    print(self.reviews[rid]['text'])
                else:
                    print(self.reviews[rid]['review'])
            min_dist = -1
            min_dist_phrase = ''
            min_dist_attr = None
            for ext in extractions:
                phrase = ext['predicate'] + ' ' + ext['entity']
                dist = self.get_dist(self.position_index[rid], phrase, qterm)
                if dist >= 0 and (min_dist < 0 or dist < min_dist):
                    min_dist = dist
                    min_dist_phrase = phrase
                    min_dist_attr = ext['attribute']

            if debug:
                print(min_dist_attr, min_dist_phrase, min_dist)

            if min_dist_attr != None:
                if min_dist_attr not in attribute_scores:
                    represented_phrases[min_dist_attr] = min_dist_phrase
                    attribute_scores[min_dist_attr] = 1
                else:
                    attribute_scores[min_dist_attr] += 1

        best_attr_score = 0.0
        best_attr = None
        for attr in attribute_scores:
            attribute_scores[attr] *= self.idf[attr]
            if attribute_scores[attr] > best_attr_score:
                best_attr_score = attribute_scores[attr]
                best_attr = attr

        if best_attr == None:
            return None, None
        return best_attr, represented_phrases[best_attr]



class SimpleOpine:
    def __init__(self, histogram_fn, extraction_fn, phrase_sentiment_fn, word2vec_fn, idf_fn, query_label_fn, entity_fn=None):
        if entity_fn == None:
            self.entities = json.load(open(histogram_fn))
            self.reviews = json.load(open(extraction_fn))
        else:
            bids = set([])
            raw_entities = json.load(open(entity_fn))
            for row in raw_entities:
                bids.add(row['business_id'])
            # filtering
            self.entities = json.load(open(histogram_fn))
            self.entities = { bid : self.entities[bid] for bid in bids }
            self.reviews = json.load(open(extraction_fn))
            self.reviews = [review for review in self.reviews if review['business_id'] in bids]

        self.phrase_sentiments = json.load(open(phrase_sentiment_fn))
        self.model = Word2Vec.load(word2vec_fn)
        self.idf = json.load(open(idf_fn))
        self.phrase2vec_cache = {}
        self.phrase_mp = {}
        self.all_phrases = []
        self.all_vectors = []
        self.membership_cache = {}
        self.interpret_cache = {}

        # index for the w2v method for query interpretation
        def build_NN_index():
            for bid in self.entities:
                histogram = self.entities[bid]['histogram']
                for attr in histogram:
                    for phrase in histogram[attr]:
                        phrase = phrase.lower()
                        if phrase not in self.phrase_mp:
                            self.phrase_mp[phrase] = len(self.phrase_mp)
                            self.all_phrases.append((attr, phrase))
                            self.all_vectors.append(self.phrase2vec(phrase))

            return KDTree(self.all_vectors, leaf_size=40)

        self.kd_tree = build_NN_index()

        # index for the co-occurrence method
        self.cooc = CooccurInterpreter(self.reviews)

        def train_scorer(num_samples=1500):
            ground_truth = {}
            all_bids = set([])
            all_qterms = set([])
            for (bid, _, qterm, res) in json.load(open(query_label_fn)):
                if bid in self.entities:
                    ground_truth[(bid, qterm)] = 1.0 if res == 'yes' else 0.0
                    all_bids.add(bid)
                    all_qterms.add(qterm)
            all_bids = list(all_bids)
            all_qterms = list(all_qterms)
            X_phrases = []
            X_summary = []
            y_phrases = []
            y_summary = []
            while len(X_phrases) < num_samples:
                bid = random.choice(all_bids)
                qterm = random.choice(all_qterms)
                attr_name, _ = self.interpret(qterm)
                if attr_name in self.entities[bid]['summaries']:
                    # print(bid, attr_name, qterm)
                    X_phrases.append(self.get_features_phrases(self.entities[bid]['histogram'][attr_name], qterm))
                    X_summary.append(self.get_features_summary(self.entities[bid]['summaries'][attr_name], qterm))
                    if (bid, qterm) in ground_truth and ground_truth[(bid, qterm)] > 0:
                        y_phrases.append(1)
                        y_summary.append(1)
                    else:
                        y_phrases.append(0)
                        y_summary.append(0)

            X_summary, X_summary_test, y_summary, y_summary_test = \
                train_test_split(np.array(X_summary), y_summary, test_size=0.33)
            marker_model = LogisticRegression().fit(X_summary, y_summary)

            X_phrases, X_phrases_test, y_phrases, y_phrases_test = \
                train_test_split(np.array(X_phrases), y_phrases, test_size=0.33)
            phrase_model = LogisticRegression().fit(X_phrases, y_phrases)

            print('phrase model score = %f' % phrase_model.score(X_phrases_test, y_phrases_test))
            print('marker model score = %f' % marker_model.score(X_summary_test, y_summary_test))
            return phrase_model, marker_model

        self.phrase_model, self.marker_model = train_scorer()

    def clear_cache(self):
        self.phrase2vec_cache = {}
        self.membership_cache = {}
        self.interpret_cache = {}

    def phrase2vec(self, phrase):
        if phrase in self.phrase2vec_cache:
            return self.phrase2vec_cache[phrase]

        words = gensim.utils.simple_preprocess(phrase)
        res = np.zeros(300)
        for w in words:
            if w in self.model.wv:
                v = self.model.wv[w] * self.idf[w]
                res += v
        #if phrase in self.phrase_sentiments and self.phrase_sentiments[phrase] < 0:
        #    res = -res
        norm = np.linalg.norm(res)
        if norm > 0:
            res /= norm

        self.phrase2vec_cache[phrase] = res
        return res

    def cosine(self, vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 > 0 and norm2 > 0:
            return np.dot(vec1, vec2) / norm1 / norm2
        else:
            return 0.0
        # return 1.0 - spatial.distance.cosine(vec1, vec2)

    def get_marker(self, attr, phrase):
        # find the closest marker to the input phrase
        vec = self.phrase2vec(phrase)
        best_match = None
        best_sim = 0.0
        for entity in self.entities.values():
            if attr in entity['summaries']:
                for marker in entity['summaries'][attr]:
                    marker_vec = marker['center']
                    sim = self.cosine(vec, marker_vec)
                    if best_match == None or sim >= best_sim:
                        best_match = marker['verbalized']
                        best_sim = sim
        return best_match


    def get_features_phrases(self, histogram, qterm):
        qvec = self.phrase2vec(qterm)
        # count number of similar phrases
        sim_count = 1.0
        count2 = 1.0
        sent_sum = 0.0
        sent_sum2 = 0.0
        pos_count = 0.0
        neg_count = 0.0
        pos_match_count = 0.0
        neg_match_count = 0.0

        sum_phrases = np.zeros(300)
        for phrase in histogram:
            pvec = self.phrase2vec(phrase)
            sum_phrases += pvec * histogram[phrase]
            if self.cosine(qvec, pvec) > 0.8:
                sim_count += histogram[phrase]
                sent_sum += histogram[phrase] * self.phrase_sentiments[phrase]
                if self.phrase_sentiments[phrase] >= 0:
                    pos_match_count += histogram[phrase]
                else:
                    neg_match_count += histogram[phrase]

            sent_sum2 += histogram[phrase] * self.phrase_sentiments[phrase]
            count2 += histogram[phrase]
            if self.phrase_sentiments[phrase] >= 0:
                pos_count += histogram[phrase]
            else:
                neg_count += histogram[phrase]

        X = []
        X.append(sim_count)
        X.append(sent_sum / sim_count)
        X.append(sent_sum2 / count2)
        X.append(pos_count)
        X.append(neg_count)
        X.append(pos_match_count)
        X.append(neg_match_count)
        X.append(self.cosine(qvec, sum_phrases))
        return np.array(X)

    def get_features_summary(self, summary, qterm, num_markers=10):
        qvec = self.phrase2vec(qterm)
        num_marker = len(summary)
        summary.sort(key=lambda x : x['sum_senti'] / (x['size'] + 1))
        X = []
        for marker in summary:
            similarity = self.cosine(marker['center'], qvec)
            X.append(marker['sum_senti'])
            X.append(marker['size'])
            X.append(marker['sum_senti'] / (marker['size'] + 1))
            X.append(similarity)
            X.append(marker['sum_senti'] / (marker['size'] + 1) * similarity)
        for _ in range(num_markers - len(summary)):
            X += [0.0] * 5

        return X

    def interpret(self, query_term, fallback_threshold=0.4):
        if query_term in self.interpret_cache:
            return self.interpret_cache[query_term]
        query_len = len(gensim.utils.simple_preprocess(query_term))
        vector = self.phrase2vec(query_term)
        kd_tree_res = self.kd_tree.query([vector], k=1)
        phrase_id = kd_tree_res[1][0][0]
        res = self.all_phrases[phrase_id]

        # fall back if similarity is too low
        phrase = res[1]
        phrase_vec = self.phrase2vec(phrase)
        similarity = self.cosine(phrase_vec, vector)
        if similarity < fallback_threshold or query_len == 1: # 0.4
            cooc_res = self.cooc.interpret(query_term)
            if cooc_res[0] != None:
                res = cooc_res

        self.interpret_cache[query_term] = res
        return res

    def opine(self, query, bids=None, mode='marker'):

        def membership(bid, attr_name, qterm):
            if (bid, attr_name, qterm) in self.membership_cache:
                return self.membership_cache[(bid, attr_name, qterm)]

            if mode == 'marker':
                if 'summaries' in self.entities[bid] and attr_name in self.entities[bid]['summaries']:
                    summary = self.entities[bid]['summaries'][attr_name]
                    score = self.marker_model.predict_proba([self.get_features_summary(summary, qterm)])[0][1]
                else:
                    score = 1e-6
            else:
                if 'histogram' in self.entities[bid] and attr_name in self.entities[bid]['histogram']:
                    histogram = self.entities[bid]['histogram'][attr_name]
                    score = self.phrase_model.predict_proba([self.get_features_phrases(histogram, qterm)])[0][1]
                else:
                    score = 1e-6
            self.membership_cache[(bid, attr_name, qterm)] = score
            return score

        if bids == None:
            bids = list(self.entities.keys())
        scores = {bid : 1.0 for bid in bids}

        for qterm in query:
            qterm = qterm.lower()
            attr_name, _ = self.interpret(qterm)
            for bid in bids:
                scores[bid] *= membership(bid, attr_name, qterm)
        return sorted(bids, key=lambda x : -scores[x])


if __name__ == '__main__':
    if len(sys.argv) < 7:
        print("Usage: python opine.py histogram_fn extraction_fn sentiment_fn word2vec_fn idf_fn query_label_fn")
        exit()

    histogram_fn = sys.argv[1]
    extraction_fn = sys.argv[2]
    sentiment_fn = sys.argv[3]
    word2vec_fn = sys.argv[4]
    idf_fn = sys.argv[5]
    query_label_fn = sys.argv[6]

    opine = SimpleOpine(histogram_fn, extraction_fn, sentiment_fn, word2vec_fn, idf_fn, query_label_fn)
    print(opine.opine(['clean room', 'helpful staff']))
