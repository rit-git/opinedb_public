'''
This file contains some simple heuristics to complete the extracted aspect 
and description pairs.

For each word, we will find its most frequent left and right neighbors.

@author: Xiaolan Wang

Created on 2019-03-28
Updated on 2019-03-28
'''  
import json
import operator
import nltk
import numpy
import pickle 
import re

from optparse import OptionParser
_max_step = 3
_max_entity_length = 4
_punct_matcher = re.compile(r'[?!\(\)\.,;&:\-$]+')
_number_matcher = re.compile(r'[0-9]+')
_buffer = 300
_penalize_ratio = 0.1
_stopword_penalize_ratio = 0.0001
_minimu_expand_freq = 0
_stop_words = set(nltk.corpus.stopwords.words('english')+['could', 'hotel', "with", 
					"next", "as", "on", "in", "at", "well", "lot", "next",
					"back", "right", "across", "door"]) 
_strict_stop_wors = set(['those', 'each', 'no', 'both', 'this', 'these', 'all',
					"aren't", 'couldn', "weren't", 's', 'mustn', 'didn', 
					'shouldn', 'ain', 'm', 'hasn', 'needn', "you'd", "shan't", 
					"wouldn't", 'wasn', "needn't", 'll', 'doesn', "don't", 
					"mightn't", 'don', "couldn't", 'haven', 'y', 'o', "doesn't", 
					"she's", "it's", "you'll", "should've", 're', 'herself', 
					"won't", "mustn't", 'd', "that'll", "didn't", "shouldn't", 
					'hadn', "hadn't", 'ma', "wasn't", 'shan', 'won', "you've", 
					"haven't", "isn't", 't', 'mightn', "hasn't", "you're", 'i', 
					'isn', 'wouldn', 've', 'that','during', 'if', 'because', 
					'than','while', 'they', 'she', 'himself', 'you', 'him', 
					'them', 'themselves', 'it', 'itself', 'yourself', 'he', 
					'myself', 'we', 'me', 'only', 'then', 'there', 'just',
					'aren', 'weren', 'hers', 'theirs', 'yours', 'ours', 
					'yourselves', 'ourselves', 'same', 'such', 'other', 'own',
					'will', 'should', 'can', 'her', 'its', 'my', 'your', 
					'his', 'their', 'our', 'whom', 'what', 'who', 'when', 
					'where', 'why', 'how', 'which','but', 'or', 'nor',
					"from", "many", "about", "us", "within", "although",
					"would", "runs", "etc", "told", "may", "and", "plus",
					"on", "in"])
_start_removal_words = set(['and', 'of', "\'s", "also", "could", "n't", "even",
					  "though", "went", "i.", "e.", ",", "to"])
_end_removal_words = set(['and', 'of', "also", "could", "n't", "even",
					  "though", "went", "i.", "e.", ",", "to", "from", 
					  "for", "on", "very"])

class Word:
	"""Each word has three properties: 
	1). its own frequence;
	2). most frequent left neighbor;
	3). most frequent right neighbor.
	"""
	def __init__(self, word):
		self.word = word
		self.freq = 0
		self.left = {}
		self.right = {}

	def update_neighbors(self, left, right):
		self.update(self.left, left)
		self.update(self.right, right)
		self.freq += 1

	def update(self, nmap, key):
		if not Util.validate_word(key, stopword=False):
			return
		if key not in nmap:
			nmap[key] = 0
		nmap[key] = nmap[key]+1

	def reduce_neighbors(self):
		self.left = self.reduce(self.left)
		self.right = self.reduce(self.right)

	def reduce(self, nmap):
		sorted_list = sorted(nmap.items(), key=operator.itemgetter(1))
		reduced_map = {}
		for item in sorted_list[-_buffer:]:
			reduced_map[item[0]] = item[1]
		return reduced_map

	def collect_topk(self, k, side='left'):
		nmap = self.left if side=='left' else self.right
		sorted_list = sorted(nmap.items(), key=operator.itemgetter(1))
		return [i for i in sorted_list[-k:]]

	def collect_topk_pairs(self, k, side='left', metric='count'):
		nmap = self.left if side=='left' else self.right
		sorted_list = sorted(nmap.items(), key=operator.itemgetter(1))
		pairs = []
		for i in sorted_list[-k:]:
			pair = " ".join([i[0],self.word]) if side=='left' \
											  else " ".join([self.word, i[0]])
			value = float(i[1])/float(self.freq) if metric=='ratio' else i[1]
			pairs.append((pair, value))
		return pairs

	def tostring(self):
		left = self.collect_topk(1, 'left')
		right = self.collect_topk(1, 'right')
		tempate = "(word:%s; freq:%d; left:%s; right:%s)"
		return tempate % (self.word, self.freq, left, right)

class FrequetWord:
	"""FrequentWord is an index of words appear in the review corpus.
	Key: word token
	Value: Word object
	"""
	def __init__(self, batch_size):
		self.word_list = {}
		self.batch_size = batch_size

	def find_self(self, word):
		if word not in self.word_list:
			return 0
		return self.word_list[word].freq

	def find(self, word, other, direction):
		if word not in self.word_list:
			return 0
		cur_map = self.word_list[word].left if direction == 'left' else \
													self.word_list[word].right
		if other not in cur_map:
			return 0
		return max(cur_map[other]-1, 0)

	def build(self, extractions):
		size = len(extractions)
		total = 0
		count = 0
		print("-----Start building-----")
		for extraction in extractions:
			self.update_single_extraction(extraction)
			count += 1
			total += 1
			if count >= self.batch_size:
				self.reduce()
				count = 0
			if total % max(self.batch_size, 1000) == 0:
				print("%d out of %d" % (total, size))
		self.reduce()
		print("-----Finish building-----")

	def update_single_extraction(self, extraction):
		context = nltk.word_tokenize(Util.proc_text(extraction['text']))
		for i in range(len(context)):
			if not Util.validate_word(context[i]):
				continue
			if context[i] not in self.word_list:
				self.word_list[context[i]] = Word(context[i])
			left = None if i == 0 else context[i-1]
			right = None if i == len(context)-1 else context[i+1]
			self.word_list[context[i]].update_neighbors(left, right)

	def reduce(self):
		updated = {}
		for key, value in self.word_list.items():
			if value.freq > 1:
				value.reduce_neighbors()
				updated[key] = value
		self.word_list = updated

	def tostring(self, k = 10, metric='count'):
		top_freq_pairs = {}
		for key, value in self.word_list.items():
			pair_list = value.collect_topk_pairs(k, 'left', metric) + \
						value.collect_topk_pairs(k, 'right', metric)
			for pair, value in pair_list:
				if pair not in top_freq_pairs:
					top_freq_pairs[pair] = 0
				top_freq_pairs[pair] = top_freq_pairs[pair] + value
		sorted_pairs = sorted(top_freq_pairs.items(), key=operator.itemgetter(1))
		print(sorted_pairs[-k:])

	def calculate_score(self, candidate, score_map = None):
		if len(candidate) == 0:
			return 0
		if len(candidate) == 1:
			return self.find_self(candidate[0]) * _penalize_ratio

		score = 0
		for i in range(1, len(candidate)):
			score += self.calculate_score_by_index(candidate[i], candidate[i-1],
												   score_map)
		updated_score = score / ((2**(max(0, len(candidate)-3)))*len(candidate))
		#print(candidate, updated_score, score)
		return updated_score

	def calculate_score_by_index(self, current, left, score_map):
		key = " ".join([current, left])
		if score_map and key in score_map:
			return score_map[key]
		score = self.find(left,current,'right')+self.find(current,left,'left')
		if current in _stop_words:
			score = score * _stopword_penalize_ratio
		score = score/2
		score_map[key] = score
		return score

class ExpandReviewExt:
	""" ExpandReviewExt cleans the extracted entity by a simple heuristic with
	the following steps:
	1. Construct a few candidate entities by explanding the original entity
	   by its neighbor tokens. 
	   E.g., Given a review "The hotel is close to union square.", and an
	   entity extraction "union", the candidate entities will be "to union"
	   "to union square", "union square". 
	2. Calculate the frequency score of all candidate entities, including the
	   original one.
	3. Updated the extraction if the highest score entity is not the original 
	   one. 

	To use:
	>>> expander = ExpandReviewExt(freq_word, review)
	>>> expander.process()
	"""
	def __init__(self, freq_word, review):
		self.freq_word = freq_word
		self.review = review
		self.source, self.info = self.initialize(review)

	def initialize(self, review):
		# Initialize
		source = nltk.word_tokenize(Util.proc_text(review['text']))
		
		pred_info = {}
		ent_info = {}
		pred_indices = set({})
		ent_indices = set({})

		# Load data
		for ext in review['extractions']:
			# find unique predicates and entities
			pred = Util.proc_text(ext['predicate'])
			if pred not in pred_info:
				pred_info[pred] = self.get_info(source, pred, pred_indices)
			ent = Util.proc_text(ext['entity'])
			if ent not in ent_info:
				ent_info[ent] = self.get_info(source, ent, ent_indices)
		info = {'pred':pred_info, 'ent':ent_info, 
				'pred_idx':pred_indices, 'end_idx':ent_indices}
		return source, info

	def process(self, option='ent', debug = False):
		opt_info = ['ent', 'pred_idx', 'entity', 'updated_entity']
		if option != 'ent':
			opt_info = ['pred', 'ent_idx', 'predicate', 'updated_predicate']

		updated_ent = {}
		for ent, info in self.info[opt_info[0]].items():
			rep, tokens = self.find_best_candidate(info, 
												   self.info[opt_info[1]],
												   debug=debug)
			if len(info['token']) != len(tokens):
				updated_ent[ent] = rep
				#print("Entity:", ent, "->", rep)

		# Update original review
		for ext in self.review['extractions']:
			ent = Util.proc_text(ext[opt_info[2]])
			if ent in updated_ent:
				ext[opt_info[3]] = updated_ent[ent]

	def find_best_candidate(self, info, conflicts, debug = False):
		tokenized, locations = info['token'], info['location']
		# Define cached results
		score_map = {}

		# Get candidates
		candidates = self.find_candidate(locations, conflicts)
		if debug:
			print(candidates)

		# Find best candidate
		max_score = self.freq_word.calculate_score(tokenized, score_map)
		best_candidate = tokenized
		if max_score < _minimu_expand_freq:
			return self.refine_candidate(tokenized)

		for candidate in candidates:
			score = self.freq_word.calculate_score(candidate, score_map)
			if debug:
				print(candidate, score, max_score)
			if score > max_score:
				best_candidate = candidate
				max_score = score
		return self.refine_candidate(best_candidate)

	def find_candidate(self, locations, conflicts):
		candidates = []
		for location in locations:
			candidates += self.expand_by_location(location, conflicts)
		return candidates

	def expand_by_location(self, location, conflicts):
		candidates = []
		cur_start, cur_end = location['start'], location['end']
		ext_start = cur_start
		while True:
			if self.expand_stop_check(ext_start, conflicts):
				ext_start += 1
				break
			if ext_start <= max(0, cur_start-_max_step):
				break
			ext_start -= 1
		ext_end = cur_end
		while True:
			if self.expand_stop_check(ext_end-1, conflicts, nostopword=False):
				ext_end -= 1
				break
			if ext_end >= min(len(self.source), cur_end+_max_step):
				break
			ext_end += 1
		for i in range(ext_start, cur_start+1):
			for j in range(cur_end, ext_end+1):
				if (i == cur_start and j == cur_end) or (j-i) > 4:
					continue
				candidates.append(self.source[i:j])
		return candidates

	def expand_stop_check(self, index, conflicts, nostopword=True):
		if bool(_punct_matcher.match(self.source[index])):
			return True
		if index in conflicts:
			return True
		pos = nltk.pos_tag([self.source[index]])
		if "VB" in pos[0][1]:
			return True
		if self.source[index] in _strict_stop_wors:
			return True
		if nostopword and self.source[index] in _stop_words:
			return True
		return False

	def refine_candidate(self, candidate):
		while len(candidate) > 0 and candidate[0] in _start_removal_words:
			candidate = candidate[1:]
		while len(candidate) > 0 and candidate[-1] in _end_removal_words:
			candidate = candidate[:-1]
		if len(candidate) < 1:
			return "", []
		return self.detokenize(candidate), candidate

	def get_info(self, source, target, update):
		tokens = nltk.word_tokenize(target)
		location = Util.find_locations(source, tokens)
		for index in self.ranges_to_indices(location):
			update.add(index)
		info = {'token':tokens, 'location':location}
		return info

	def ranges_to_indices(self, ranges):
		indices = set([])
		for r in ranges:
			for i in range(r['start'], r['end']):
				indices.add(i)
		return indices

	def detokenize(self, tokens):
		span = ""
		for i in range(len(tokens)):
			space = " " if i > 0 and "'" not in tokens[i] else ""
			span = span + space + tokens[i]
		return span

class Util:
	""" Utility functions.

	To use:
	Remove unwanted punctuations from text:
	>>> Util.proc_text(text)
	
	Valid a word by checking whether it is a punctuation or a stopword.
	>>> Util.validate_word(word_token)

	Read extractions from json file format.
	>>> Util.read_extractions(file_name)
	
	Find the start and end position(s) of a phrase in a sentence.
	>>> Util.find_locations(tokenized sentence, tokenized phrase)

	Read the existing FrequentWord index from disk or rebuild it from
	scratch.
	>>> Util.get_freq_words(path, batch_size, reviews, build = False)

	Clean extractions.
	>>> Util.process_all(reviews, freq_words, batch_size, limit)

	Save updated extractions.
	>>> Util.save_updated()
	"""
	@staticmethod
	def proc_text(text):
		text = text.replace(".", ". ").replace('\\n', " ").replace("´", "'")
		text = text.replace("''", "'").replace("‘", "'").replace("’", "'")
		text = text.replace('`', "'").replace(u'\xa0', u' ')
		text = text.replace("check - in", "check-in")
		return text.lower()

	@staticmethod
	def validate_word(word, stopword=True):
		if not word:
			return False
		if bool(_punct_matcher.match(word)):
			return False
		if stopword and word in _stop_words:
			return False
		return True

	@staticmethod
	def read_extractions(file_path):
		with open(file_path, "r") as file:
			extractions = json.load(file)
		return extractions

	@staticmethod
	def find_locations(source, tokenized):
		s_len = len(source)
		t_len = len(tokenized)
		match = [[False for i in range(t_len+1)] for j in range(s_len+1)]
		for i in range(0, s_len+1):
			match[i][0] = True
		for j in range(1, t_len+1):
			for i in range(1, s_len+1):
				match[i][j] = match[i-1][j-1] and source[i-1] == tokenized[j-1]
		locations = []
		for i in range(0, s_len+1):
			if match[i][t_len]:
				locations.append({'start':i-t_len, 'end':i})
		return locations

	@staticmethod
	def get_freq_words(path, batch_size, reviews, build = False):
		if not build:
			try:
				with open(path, "rb") as file:
					freq_words = pickle.load(file)
				return freq_words
			except:
				build = True
				pass
		assert reviews, "No data provided."
		freq_words = FrequetWord(batch_size)
		freq_words.build(reviews)
		if len(path) > 0:
			with open(path, 'wb') as file:
				pickle.dump(freq_words, file)
		return freq_words

	@staticmethod
	def process_all(reviews, freq_words, batch_size, limit):
		size = len(reviews)
		print("-----Start processing-----")
		for i in range(size):
			processor = ExpandReviewExt(freq_words, reviews[i])
			processor.process()
			#DataItem.process_one_extraction(reviews[i], freq_words)
			if i % max(batch_size, 1000) == 0:
				print("%d out of %d" % (i, size))
			if i > limit:
				break
		print("-----Finish processing-----")

	@staticmethod
	def save_updated(reviews, output_file, updated_only = False):
		updated_list = []
		if updated_only:
			for i in range(len(reviews)):
				for ext in reviews[i]['extractions']:
					if 'updated_entity' not in ext:
						continue
					updated_ext = {}
					updated_ext['id'] = i
					updated_ext['predicate'] = ext['predicate']
					updated_ext['entity'] = ext['updated_entity']
					updated_ext['original_entity'] = ext['entity']
					updated_ext['negation'] = ext['negation']
					updated_ext['attribute'] = ext['attribute']
					updated_list.append(updated_ext)
			with open(output_file, "w") as file:
				json.dump(updated_list, file, indent=4)
		else:
			with open(output_file, "w") as file:
				json.dump(reviews, file)

if __name__=='__main__':
	usage = "usage: %prog [options] input_file output_file ..."

	optParser = OptionParser(usage=usage, version="%prog ")
	optParser.add_option("-b", "--batchsize", dest="batch_size", 
    					 action = "store", 
    					 help="define batch size, default 1000", 
             			 default='1000')
	optParser.add_option("-l", "--limit", dest="limit",
    					 action = "store", 
    					 help="define maximum # of reviews, default unlimit", 
             			 default='unlimit')
	optParser.add_option("-f", "--frequentwords", dest="freq_words",
						 action = "store",
						 help="define path to store frequent words",
						 default='')
	optParser.add_option("-r", "--rebuild", dest="build",
						 action = "store_true",
						 help="rebuild the frequence word index",
						 default=False)
	optParser.add_option("-u", "--updatedonly", dest="updated_only",
						 action = "store_true",
						 help="only save updated reviews",
						 default=False)
	(options, args) = optParser.parse_args()
	
	if len(args) < 2:
		print(usage)
		sys.exit(1)
	else:
		input_file = args[0]
		output_file = args[1]

	reviews = Util.read_extractions(input_file)
	freq_words = Util.get_freq_words(options.freq_words, 
									int(options.batch_size), reviews, 
									build=options.build)
	if options.limit == 'unlimit':
		limit = len(reviews)+1
	else:
		limit = int(options.limit)
	Util.process_all(reviews, freq_words, int(options.batch_size), limit)

	Util.save_updated(reviews, output_file, updated_only=options.updated_only)
