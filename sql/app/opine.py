import sys
import os
import json
import gensim
import math
import random
import copy
import numpy as np

from gensim.models import Word2Vec
from scipy import spatial
from sklearn.neighbors import KDTree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from moz_sql_parser import parse
from moz_sql_parser import format


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class SimpleOpine(object):
    def __init__(self,
            histogram_fn="data/amsterdam_hotels_with_histograms.json",
            phrase_sentiment_fn="data/amsterdam_sentiment.json",
            word2vec_fn="data/word2vec.model",
            query_label_fn="data/amsterdam_labels.json"):
        self.entities = json.load(open(histogram_fn))
        self.phrase_sentiments = json.load(open(phrase_sentiment_fn))
        self.model = Word2Vec.load(word2vec_fn)
        self.phrase2vec_cache = {}
        self.phrase_mp = {}
        self.all_phrases = []
        self.all_vectors = []
        self.membership_cache = {}
        self.interpret_cache = {}

        def build_NN_index():
            for bid in self.entities:
                histogram = self.entities[bid]['histogram']
                for attr in histogram:
                    for phrase in histogram[attr]:
                        if phrase not in self.phrase_mp:
                            self.phrase_mp[phrase] = len(self.phrase_mp)
                            self.all_phrases.append((attr, phrase))
                            self.all_vectors.append(self.phrase2vec(phrase))
            return KDTree(np.array(self.all_vectors), leaf_size=40)

        self.kd_tree = build_NN_index()

        def train_scorer(num_samples=1500):
            ground_truth = {}
            all_bids = set([])
            all_qterms = set([])
            for (bid, _, qterm, res) in json.load(open(query_label_fn)):
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
            
            print('phrase model coef = ' + str(phrase_model.coef_)  )          
            print('phrase model intercept = ' + str(phrase_model.intercept_))
            
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
                v = self.model.wv[w]
                res += v
        if phrase in self.phrase_sentiments and self.phrase_sentiments[phrase] < 0:
            res = -res
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

    def interpret(self, query_term):
        if query_term in self.interpret_cache:
            return self.interpret_cache[query_term]
        vector = self.phrase2vec(query_term)
        phrase_id = self.kd_tree.query([vector], k=1)[1][0][0]
        res = self.all_phrases[phrase_id]
        self.interpret_cache[query_term] = res
        return res

    
    def opine(self, query, mode='marker'):

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

        scores = {bid : 1.0 for bid in self.entities}
        for qterm in query:
            qterm = qterm.lower()
            attr_name, _ = self.interpret(qterm)
            for bid in self.entities:
                scores[bid] *= membership(bid, attr_name, qterm)
        return sorted(list(self.entities.keys()), key=lambda x : -scores[x])

    
    def combine_histo_vector(self):
        for bid in self.entities:
            histogram = self.entities[bid]['histogram']
            for attr in histogram:
                for phrase in histogram[attr]:
                    new_phrase_dict = {}
                    
                    phrase_value = histogram[attr][phrase]
                    pvec = self.phrase2vec(phrase)
                    sentiment = self.phrase_sentiments[phrase]
                    
                    new_phrase_dict["value"] = phrase_value
                    new_phrase_dict["pvec"] = pvec.tolist() 
                    new_phrase_dict["sentiment"] = sentiment
                    
                    histogram[attr][phrase] = new_phrase_dict
                    
                    
        json.dump(self.entities, open('data/combine_histo.json', 'w',encoding='utf8'), indent=4, sort_keys=True,ensure_ascii=False)
        print("combine histo with sentiment and pvec done")

    def sqlparser(self, sql):
        parsed_sql = parse(sql)

        print("whole parse tree：", parsed_sql)

        objective_and_clause = []
        subjective_and_clause = []
        table_alias = {}

        for clause in parsed_sql:

            if clause is "select":
                print("select clause:",parsed_sql[clause])

            if clause is "from":
                if isinstance(parsed_sql[clause], dict): 
                    name = parsed_sql[clause]['value']
                    alias= parsed_sql[clause]['name']
                    table_alias[alias] = name
                        
                elif isinstance(parsed_sql[clause], str):
                        table_alias[parsed_sql[clause]] = parsed_sql[clause]
                        
                elif isinstance(parsed_sql[clause], list):
                    for tableclause in parsed_sql[clause]:
                        if isinstance(tableclause, dict): 
                            name = tableclause['value']
                            alias= tableclause['name']
                            table_alias[alias] = name

                        elif isinstance(tableclause, str):
                            table_alias[tableclause] = tableclause


            if clause is "where":
                for subclause in parsed_sql[clause]:
                    if subclause is "and":
                        for andclause in parsed_sql[clause][subclause]:
                            if "eq" in andclause :
                                leftclause = andclause['eq'][0]
                                rightclause = andclause['eq'][1]
                                splitkey = leftclause.split('.')
                                if len(splitkey) is 2 and splitkey[1].strip() == "opine":
                                    #translate table alias
                                    if splitkey[0] in table_alias:
                                        tablename = table_alias[splitkey[0]]+ "_histogram"
                                    else:
                                        tablename = splitkey[0]+ "_histogram"
                                    qterm = andclause['eq'][1]['literal'] 
                                    #TODO: change to simple tablename
                                    #subjective_and_clause.append({tablename:qterm})
                                    subjective_and_clause.append(qterm)
                                    
                                else:
                                    objective_and_clause.append(andclause)
                            else:
                                objective_and_clause.append(andclause)
                    else:
                        singleclause = parsed_sql[clause]
                        
                        if "eq" in singleclause :
                            leftclause = singleclause['eq'][0]
                            rightclause = singleclause['eq'][1]
                            splitkey = leftclause.split('.')
                            if len(splitkey) is 2 and splitkey[1].strip() == "opine":
                                #translate table alias
                                if splitkey[0] in table_alias:
                                    tablename = table_alias[splitkey[0]]+ "_histogram"
                                else:
                                    tablename = splitkey[0]+ "_histogram"
                                qterm = singleclause['eq'][1]['literal'] 
                                #TODO: change to simple tablename
                                #subjective_and_clause.append({tablename:qterm})
                                subjective_and_clause.append(qterm)

                            else:
                                objective_and_clause.append(singleclause)
                        else:
                            objective_and_clause.append(singleclause)
                            

        print("table alias",table_alias)                            
        print("subjective_and_clause",subjective_and_clause)
        print("objective_and_clause",objective_and_clause)      
        
        return parsed_sql, subjective_and_clause, objective_and_clause
        
    def translate(self, parsed_sql, qterm, attr, where_clause):
        qvec = self.phrase2vec(qterm)
        
        qvec_literal = 'ARRAY[' + ','.join([str(v) for v in qvec]) + ']::DOUBLE PRECISION[]'
        parsed_sql['select'] = [parsed_sql['select'],
                                {
                                    'value': 
                                         {'logistic': 
                                              [
                                                   {'get_feature_phrases': [qvec_literal, 'pvec', 'senti', 'count']},
                                                  'coef',
                                                  'intercept'
                                              ]
                                         }, 
                                     'name': 'score'
                                }
                               ]
        entity_table_clause = parsed_sql['from']
        if type(entity_table_clause) is dict:
            entity_table_name = entity_table_clause['value']
        else:
            entity_table_name = entity_table_clause
                    
        parsed_sql['from'] = [entity_table_clause,
                              {'value': entity_table_name + '_phrase_model', 'name': 'pm'},
                              {'value': entity_table_name + '_histogram', 'name': 'c'}]
        if type(where_clause) is not list:
            where_clause = [where_clause]
        parsed_sql['where'] = {'and': 
                                   where_clause + \
                                   [{'eq': ['c.attribute', {'literal': attr}]}, {'eq': ['h.name', 'c.hotel_name']}]
                              }
                                   
        parsed_sql['groupby'] = [{'value': 'h.name'}, {'value': 'coef'}, {'value': 'intercept'}]
        parsed_sql['orderby'] = {'value': 'score', 'sort': 'desc'}
        
        pgsql = format(parsed_sql)
        return pgsql
    
    def opine_sql(self, sqlquery, mode='histogram'):
        """
        One test case input (query):
            SELECT h.name
            FROM hotel_amsterdam AS h
            WHERE h.price <= 15
              AND h.opine = 'very clean room';
              
        expected output (pgsql):            
            SELECT
                h.name,
                madlib.logregr_predict_prob(
                generate_features_phrases( -- user-defined aggregate
                           p2v('very clean room'),
                                           -- numeric vectors are passed directly by the parser
                                           -- parser translates the query terms to vectors
                           pvec, senti, count),
                M.coef) AS score
            FROM
                hotel_amsterdam AS h
                hotel_amsterdam_phrase_model AS pm, -- single row logistic regression model
                hotel_amsterdam_histogram AS c
            WHERE h.price <= 15
              AND c.attribute = %s(nearest_attr)
              and h.name = c.hotel_name              
            GROUP BY h.name
            ORDER BY score
        """
        # TODOs
        # DONE 1. for each row in table hotel_amsterdam_histogram, add pvec, senti value to the table --
        # NEED CORRECTION!!! 2. finish self.translate
        # 3. write plpython aggregate for generate_feature
        # DONE 4. create phrase_model table and insert the coef into the table
        # 5. (after the deadline) add same for marker, allow multiple entity tables, 
        #    remove the need to load histogram and extractions in memory, etc.
        
        #parser part
        parsed_sql, qterms, where_clause = self.sqlparser(sqlquery)
        nearest_attr = []
        qvec_list = []
        result_row = []
        
        import psycopg2
        conn = psycopg2.connect("host=postgres dbname=postgres user=postgres port=5432")

        #no subjective query, only objective
        if len(qterms) == 0:
            pgsql = format(parsed_sql)
            cur = conn.cursor()
            cur.execute(pgsql)
            rows = cur.fetchall()
            #print(rows)
            result_row.append(rows)
            
            
        #computer each qterm column separately
        for qterm in qterms:
            qterm_sql = copy.deepcopy(parsed_sql)
            qterm_where_clause = copy.deepcopy(where_clause)
            
            qterm = qterm.lower()
            
            qvec = self.phrase2vec(qterm)
            qvec_list.append(qvec)
            
            attr_name, _ = self.interpret(qterm)
            nearest_attr.append(attr_name)
            
            print(qterm, attr_name)
            
            #translate
            pgsql = self.translate(qterm_sql, qterm, attr_name, qterm_where_clause)
            #print(pgsql)
            
            #plpython part
            cur = conn.cursor()
            cur.execute(pgsql)
            rows = cur.fetchall()
            
            result_row.append(rows)
        
        #aggregate logic
        result = {}
        for idx, row in enumerate(result_row):
            if row is None:
                continue
                
            if idx == 0:
                for bid in row:
                    if len(bid) > 1:
                        result[bid[0]] = bid[1]
                    else:
                        result[bid[0]] = 1
            else:
                for bid in row:
                    result[bid[0]] *= bid[1] #fuzzy logic now

        return sorted(result.items(), key=lambda x: -x[1])

    def opine_in_mem(self, query, bids=None, mode='histogram'):
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
                    features = self.get_features_phrases(histogram, qterm)
                    score = self.phrase_model.predict_proba([features])[0][1]
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
        sorted_bids = sorted(bids, key=lambda x : -scores[x])
        return [(bid, scores[bid]) for bid in sorted_bids]
    

if __name__ == '__main__':
    histogram_fn = sys.argv[1]
    sentiment_fn = sys.argv[2]
    word2vec_fn = sys.argv[3]
    query_label_fn = sys.argv[4]

    opine = SimpleOpine(histogram_fn, sentiment_fn, word2vec_fn, query_label_fn)


    #run simple opine original
    #print(opine.opine(['dirty room', 'helpful staff']))

    #combine pvec and sent to histo phrases
    #opine.combine_histo_vector()


    sql1 = """
            SELECT h.name
            FROM hotel_amsterdam AS h
            WHERE h.opine = 'helpful staff'
            """
    print(opine.opine_sql(sql1))

    sql2 = """
            SELECT h.name
            FROM hotel_amsterdam AS h
            WHERE h.opine = 'very clean room'
              AND h.price <= 15
              AND h.opine = 'helpful staff'
              AND h.opine = 'romantic'
            """
    print(opine.opine_sql(sql2))


    sql3 = """
            SELECT h.name
            FROM hotel_amsterdam AS h
            WHERE h.price <= 15
            """
    print(opine.opine_sql(sql3))

    sql4 = """
            SELECT h.name
            FROM hotel_amsterdam AS h
            WHERE h.opine = 'very clean room'
              AND h.price <= 15
              AND h.opine = 'helpful staff'
              AND h.price >= 15
              AND h.opine = 'romantic'
            """
    print(opine.opine_sql(sql4))

    sql5 = """
            SELECT h.name
            FROM hotel_amsterdam AS h
            WHERE h.opine = 'very clean room'
              AND h.price <= 500
              AND h.opine = 'helpful staff'
              AND h.price > 15
              AND h.opine = 'romantic'
            """
    print(opine.opine_sql(sql5))

    sql6 = """
            SELECT h.name
            FROM hotel_amsterdam AS h
            WHERE h.opine = 'very clean room'
              AND h.price = 15
              AND h.opine = 'helpful staff'
              AND h.opine = 'romantic'
            """
    print(opine.opine_sql(sql6))

    #print(opine.opine(['very clean room']))
    #print(opine.opine_in_mem('very clean room'))
