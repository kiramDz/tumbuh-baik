# Dockerfile untuk Flask
FROM python:3.10-slim

# Set working directory
WORKDIR /app


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh source ke container
COPY . .

# Expose port Flask
EXPOSE 5000

# Jalankan Flask
CMD ["python", "app.py"]
