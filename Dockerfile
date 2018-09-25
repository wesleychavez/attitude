# Instructions copied from - https://hub.docker.com/_/python/
FROM gw000/keras:2.1.4-py3-tf-cpu

ADD flask-app/ /opt/flask-app
WORKDIR /opt/flask-app

RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y python3-pip
RUN pip3 --no-cache-dir install \
    pandas \
    scikit-learn \
    gensim \
    Flask==0.10.1

# tell the port number the container should expose
EXPOSE 5000

# run the command
CMD ["python3", "./app.py"]
