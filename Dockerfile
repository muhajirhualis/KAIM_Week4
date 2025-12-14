# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY notebooks/ ./notebooks/

# For now: just keep container alive (will be updated in Task 6)
CMD ["tail", "-f", "/dev/null"]