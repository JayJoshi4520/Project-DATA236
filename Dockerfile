FROM python:3.11.11-slim

# Environment variables
ENV PYTHONUNBUFFERED=True
ENV APP_HOME=/APP_HOME

# Set the working directory
WORKDIR $APP_HOME

# Install system dependencies and build TA-Lib from source
RUN apt-get update && apt-get install -y \
    gcc \
    make \
    wget \
    build-essential \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "fastapi[standard]"

# Start the application
CMD exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker temp:app
