# Use Python 3.10 as the base image
FROM python:3.10

# Install Git
RUN apt-get update && apt-get install -y git

# Set up your working directory in the container
WORKDIR /app

# Copy your Python application files into the container
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Command to run your Python application
CMD ["python", "app.py"]

