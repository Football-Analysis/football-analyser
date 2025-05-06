FROM python:3.13-slim

WORKDIR /football-predictor

COPY . /football-predictor

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-u", "main.py"]