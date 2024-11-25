FROM python:3.12-slim

WORKDIR /app

COPY ["requirements.txt", "./"]

RUN pip install -r requirements.txt

RUN mkdir -p src models

COPY ["./src/predict.py", "./src"]

COPY ["./models/dv_model.pkl", "./models"]

EXPOSE 9696

WORKDIR /app/src

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]