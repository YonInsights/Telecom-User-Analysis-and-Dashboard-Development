# Use the official Python image as the base
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . /app/

# Set the FLASK_APP environment variable to point to your Flask app
ENV FLASK_APP=app/main.py

# Expose the port the application will run on
EXPOSE 5000

# Set the command to run the Flask application using Flask's built-in server
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
