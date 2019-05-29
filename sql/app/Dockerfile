FROM python:latest

RUN apt-get update
RUN pip install --upgrade pip

ADD requirements.txt .
RUN pip install -r ./requirements.txt

ADD opine.py .
ADD https://s3.us-east-2.amazonaws.com/yelp-opine/sql/data.tar.gz .
RUN tar -xzf data.tar.gz && rm data.tar.gz

CMD [ "python", "./opine.py", "./data/amsterdam_hotels_with_histograms.json", "./data/amsterdam_sentiment.json", "./data/word2vec.model", "./data/amsterdam_labels.json" ]
