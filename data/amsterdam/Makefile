S3_LINK := $(shell jq -r '.s3_link' config.json)
ZIP := $(shell jq -r '.zip_fn' config.json)
ENTITY := $(shell jq -r '.entity_fn' config.json)
RAW_REVIEWS := $(shell jq -r '.raw_review_fn' config.json)
EXTRACTIONS := $(shell jq -r '.extraction_fn' config.json)
ALL_REVIEWS := $(shell jq -r '.all_reviews_fn' config.json)
QUERIES := $(shell jq -r '.queries_fn' config.json)

ALL: $(ENTITY) $(RAW_REVIEWS) $(EXTRACTIONS)  $(ALL_REVIEWS) $(QUERIES)
	python ../../util/generate_markers.py $(ENTITY) $(RAW_REVIEWS) $(EXTRACTIONS) $(ALL_REVIEWS) word2vec.model idf.json sentiment.json entities_with_histograms.json $(QUERIES) labels.json

$(ENTITY) $(RAW_REVIEWS) $(EXTRACTIONS) $(ALL_REVIEWS) $(QUERIES): 
	wget -N $(S3_LINK)
	unzip -o $(ZIP)
