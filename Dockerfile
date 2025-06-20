# Dockerfile for AutoML Streamlit App

# 1. Use an official Python image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# 4. Copy project files to container
COPY . /app

# 5. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Expose port Streamlit runs on
EXPOSE 8501

# 7. Set environment variables to suppress warnings
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 8. Run the Streamlit app
CMD ["streamlit", "run", "app.py"] docker file name