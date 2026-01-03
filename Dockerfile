FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy semua source code
COPY . .

# Install dependencies
RUN pip install -r "requirements.txt" --no-cache

# Expose port FastAPI
EXPOSE 8000

# Jalankan FastAPI dengan Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
