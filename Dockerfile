FROM python:3.10-slim

ENV  PYTHONUNBUFFERED=True

ENV APP_HOME=/APP_HOME

WORKDIR $APP_HOME

COPY . ./
RUN pip install -r requirements.txt
RUN pip install "fastapi[standard]"
CMD exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker temp:app