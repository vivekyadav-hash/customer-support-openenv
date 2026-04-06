FROM python:3.11-slim

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

RUN chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app/src
ENV PORT=7860

CMD ["uvicorn", "envs.customer_support_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]