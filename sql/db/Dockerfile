FROM postgres:11

RUN apt-get update
RUN apt-get -y install python3 postgresql-plpython3-11
RUN apt-get clean && \
     rm -rf /var/cache/apt/* /var/lib/apt/lists/*

ADD https://s3.us-east-2.amazonaws.com/yelp-opine/sql/opine.sql.gz ./
RUN chmod 666 ./opine.sql.gz
RUN mv ./opine.sql.gz /docker-entrypoint-initdb.d/

ENTRYPOINT ["/docker-entrypoint.sh"]
EXPOSE 5432
CMD ["postgres"]
