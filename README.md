# OpineDB: a subjective database engine for querying subjective data

## How to run:

### Install required packages

All experimental codes are written in Python 3. To install the required packages:

```
pip install -r requirements.txt
```

### Datasets

To download the datasets, simply do

```
cd data/
make
```

The datasets consist of a hotel review dataset from (booking.com)[https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe] and a restaurant review dataset from (Yelp)[https://www.yelp.com/dataset]. More specifically, our experiment considers 4 subsets of the two datasets: hotels in Amsterdam, hotels in London < $300 per night, low price restaurants in Toronto, and japanese restaurants in Toronto.

TODO: add a table of the dataset summaries.


### Run the experiment scripts

```
python eval/evaluate.py amsterdam
python eval/evaluate.py london
python eval/evaluate.py lp_toronto
python eval/evaluate.py jp_toronto
```

TODO: add the instructions for running the interpreter exp.

### How to run the extraction pipeline (on Google Colab):

See instructions in ``extractor/run_extractor.ipynb``. The pipeline was used for generating the ``*_reviews_with_extractions.json`` files in the downloaded datasets.

### How to run the extraction experiments (and the baseline):

See instructions in ``extractor/run_extractor_exp.ipynb``.
