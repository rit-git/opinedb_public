# OpineDB

OpineDB is a subjective database engine for extracting, aggregating, and querying subjective data. See [our paper](https://arxiv.org/abs/1902.09661) for more details.

## Install required packages

All experimental codes are written in Python 3. To install the required packages:

```
pip install -r requirements.txt
```

## Datasets

To download the datasets, simply do

```
cd data/
make
```

The datasets consist of a hotel review dataset from [booking.com](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe) and a restaurant review dataset from [Yelp](https://www.yelp.com/dataset). More specifically, our experiment considers 4 subsets of the two datasets: hotels in Amsterdam, hotels in London < $300 per night, low price restaurants in Toronto, and japanese restaurants in Toronto.


### What's in each dataset?

The files within each dataset are listed in the configuration files ``data/$city/config.json`` for ``$city`` in amsterdam, london, and toronto. For example, in ``data/amsterdam/config.json``:

```
{
    "s3_link" : "https://s3.us-east-2.amazonaws.com/yelp-opine/new_extractor/simple_opine/amsterdam_hotels.zip",
    "zip_fn" : "amsterdam_hotels.zip",
    "entity_fn" : "raw_hotels.json",
    "raw_review_fn" : "raw_reviews.csv",
    "extraction_fn" : "amsterdam_reviews_with_extractions.json",
    "all_reviews_fn" : "all_reviews.json",
    "queries_fn" : "hotel_queries.txt", 
    "histogram_fn" : "entities_with_histograms.json",
    "sentiment_output_fn" : "sentiment.json",
    "word2vec_fn" : "word2vec.model",
    "idf_fn": "idf.json",
    "labels_fn" : "labels.json"
}
```

* the field ``s3_link`` is the download link of the dataset, 
* ``entity_fn`` is the JSON file containing the entity info,
* ``raw_review_fn`` is a csv file of the raw text reviews,
* ``extraction_fn`` is the file of reviews with the extracted opinions (see the extractor section below for how to run the extraction pipeline),
* ``all_reviews_fn`` is a (large enough) list of review text for training the word2vec model,
* ``queries_fn`` is a list of crowd-source subjective query predicates,
* ``histogram_fn`` is a JSON file containing the marker aggregates computed using ``util/generate_markers.py``,
* ``sentiment_output_fn`` contains the normalized average sentiment of each extracted phrase,
* ``word2vec_fn`` is the Word2Vec model trained from ``all_reviews_fn``,
* ``idf_fn`` is the IDF (inverse document frequency) of each token in the Word2Vec model, and
* ``labels_fn`` is the JSON file containing all the (entity, predicate) labels for evaluation.

The ``make`` command calls the script ``util/generate_markers.py`` for generating the files ``histogram_fn``, ``sentiment_output_fn``, ``word2vec_fn``, and ``idf_fn``.

## Run the experiment scripts

* To run one round of the query result quality experiment on one set of entities:

```
python eval/evaluate.py amsterdam
```

The keyword ``amsterdam`` can be replaced with ``london``, ``toronto_lp``, or ``toronto_jp``.

We also provide a python script ``eval/run_all.py`` to run all the experiments with 10 repetitions. Simply run:

```
python eval/run_all.py
python eval/read_results.py
```

The ``read_results.py`` script will print out the data for the two quality-related tables in the original paper.

* To run the query interpreter experiments, one can use the script ``eval/eval_interpreter.py``:

```
python eval/eval_interpreter.py retrain hotel
```

where the keyword ``retrain`` can be replaced with ``read_result`` to read the experimental results and the ``hotel`` keyword can be replaced with ``restaurant`` to produce the results on restaurants.


## How to run the extraction pipeline (on Google Colab):

See instructions in ``extractor/run_extractor.ipynb``. The pipeline was used for generating the ``*_reviews_with_extractions.json`` files in the downloaded datasets.

## How to run the extraction experiments (and the baseline):

See instructions in Section 2 of ``extractor/run_extractor.ipynb``.

## SQL support 

See instructions in ``sql/``.
