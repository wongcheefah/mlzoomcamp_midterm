FROM python:3.10.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "predict.py", "./"]

COPY ./model/best_model.pkl /app/model/

RUN pipenv install --system --deploy

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]