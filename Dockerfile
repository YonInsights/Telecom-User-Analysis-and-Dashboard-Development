# Use the official Python image as the base
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code into the container
COPY . /app/

# Expose the port the application will run on
EXPOSE 5000

# Set the command to run the model serving script
CMD ["python", "app/main.py"]
