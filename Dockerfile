FROM python:3.12-slim

WORKDIR /app

# ✅ Copy from correct src/ path
COPY ["src/predict.py", "model_C=1.0.bin", "./"]

RUN pip install --upgrade pip
RUN pip install flask scikit-learn numpy gunicorn

EXPOSE 5000

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:5000", "predict:app"]