FROM python:3.13-slim

WORKDIR /app

RUN useradd -m -s /bin/bash appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models /app/monitoring && \
    chown -R appuser:appuser /app/models /app/monitoring

COPY src/ /app/src/

COPY models /app/models

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]
