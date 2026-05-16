FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install flask numpy gymnasium matplotlib mlflow

EXPOSE 5000

CMD ["python", "app.py"]