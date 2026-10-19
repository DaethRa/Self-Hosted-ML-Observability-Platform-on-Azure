FROM python:3.13-slim

WORKDIR /app

RUN useradd -m -s /bin/bash appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models && chown -R appuser:appuser /app/models

COPY src/ /app/src/

COPY models/metadata.json /app/models/metadata.json

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]
