FROM ubuntu:20.04

MAINTAINER Your Name "elya.poghosyan"

RUN apt-get update -y && \
    apt-get install -y python3-pip

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]



