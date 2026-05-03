FROM python:3.10-slim

# Install system dependencies needed for OpenCV, Torch, etc.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user

# Set working directory and grant ownership to the non-root user BEFORE switching
WORKDIR /app
RUN chown -R user:user /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies globally as root
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code with correct ownership
COPY --chown=user:user . /app

# Switch to the non-root user
USER user

# Create uploads directory in /tmp (writable by the current user)
RUN mkdir -p /tmp/uploads

# Expose the default port for HF spaces (7860)
EXPOSE 7860

# Command to run the application using uvicorn, ensuring it uses the HF provided PORT (defaulting to 7860)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
