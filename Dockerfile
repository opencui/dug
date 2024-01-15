FROM python:3.11.7-bullseye
WORKDIR /data
COPY . .
RUN pip install -r requirements.txt && python -m opencui && rm -rf *
