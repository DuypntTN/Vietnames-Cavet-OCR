FROM python:3.11.3

COPY requirements.txt .
# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt
# Install pytorch cuda version
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# Copy the source code
COPY . .
# Run the application
CMD ["python", "app.py"]