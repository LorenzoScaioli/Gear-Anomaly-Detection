# docker build -t ls_pytorch .
# docker run --shm-size=10gb -it -v ./:/workspace --gpus=all --name=ls_ingranaggi ls_pytorch

# Start from the official PyTorch image (CPU or CUDA version)
FROM pytorch/pytorch:latest

# Set a working directory inside the container
WORKDIR /workspace

# Copy local project files into the container
COPY . /workspace

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# (Optional) Install additional system packages
# RUN apt-get update && apt-get install -y libgl1-mesa-glx

# (Optional) Install project dependencies using Poetry
# RUN poetry install

# Set the default command
CMD ["bash"]
