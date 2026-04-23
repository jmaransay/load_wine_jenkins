FROM python:3.12-slim
# Set the working directory in the container
WORKDIR /app
# Copy the requirements file into the container
COPY requirements.txt .
# Copy the "model.pkl" file
COPY model.pkl .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy your application code
COPY . .
# Expose the port for your application
EXPOSE 9000
# Run the application
CMD ["python", "container_serving.py"]
