FROM ubuntu:24.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libarmadillo-dev \
    libboost-all-dev \
    libglfw3-dev \
    libzmq3-dev \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment for Python
WORKDIR /app
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Conan inside the virtual environment
RUN pip install --upgrade pip && pip install conan

COPY . .

# Install Conan dependencies
RUN conan install . --build=missing --profile=conan/alpine

# Build the C++ project
WORKDIR /app/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release
RUN cmake --build .

# Run the application
CMD ["./bin/ExGrafTraining"]
