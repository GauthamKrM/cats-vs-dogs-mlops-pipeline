# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy source code
COPY src/ src/
COPY models/ models/
# Note: In a real MLOps pipeline, models might be pulled from a registry or DVC remote
# rather than copied from build context. But for this assignment packaging, we include it.

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
