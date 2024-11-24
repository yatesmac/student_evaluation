FROM python:3.12-slim

WORKDIR /app

COPY ["requirements.txt", "./"]

RUN pip install -r requirements.txt

COPY ["./src/predict.py", "./"]

CMD ["mkdir", "models"]

COPY ["./models/dv_model.pkl", "./models"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]