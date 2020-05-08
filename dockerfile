FROM frolvlad/alpine-python-machinelearning:latest

RUN apk --no-cache add gcc musl-dev python3-dev 

WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
EXPOSE 4000

ENTRYPOINT  ["python"]

CMD ["app.py"]
