# Use a small, secure Python base
FROM python:3.11-slim

# Avoid interactive prompts & keep image small
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system deps if you need them (uncomment as needed)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential && \
#     rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python deps first (leverages Docker layer caching)
COPY requirements.txt ./ 
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your code
COPY . .

# Cloud Run sets $PORT; Streamlit must bind to 0.0.0.0 and that port
# (Cloud Run injects PORT; don’t hardcode it) :contentReference[oaicite:0]{index=0}
ENV PORT=8501
CMD ["sh", "-c", "streamlit run Job_Responsibility_Generator.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true"]
