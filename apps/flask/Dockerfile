FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Salin file requirements.txt terlebih dahulu
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh source code (termasuk app.py, dsb)
COPY . .

# Expose port Flask
EXPOSE 5000

# Jalankan Flask
CMD ["python", "-u", "app.py"]

