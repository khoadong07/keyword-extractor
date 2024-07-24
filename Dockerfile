FROM python:3.12-slim

# Set environment variable to ensure output is not buffered
ENV PYTHONUNBUFFERED=1

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy the requirements.txt file into the container
COPY requirements.txt /app/requirements.txt

# Install the dependencies from requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
