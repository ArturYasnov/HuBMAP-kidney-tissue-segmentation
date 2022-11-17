FROM python:3.8.1

COPY src /ml
COPY Data /ml/Data
COPY models /ml/models

ADD requirements.txt /ml/requirements.txt

WORKDIR /ml
RUN pip install -r /ml/requirements.txt
CMD python predict.py
