# Base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader vader_lexicon

# Expose port (useful for local testing, ignored by Cloud Run)
EXPOSE 7860

# Command to run the application
CMD ["python", "web-interface.py"]
