FROM ubuntu:latest AS builder
RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  git \
  python3 \
  python3-pip \
  m4 \
  python3-venv && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip && pip install conan
COPY . .
RUN conan install . --build=missing --profile:host=conan/alpine --profile:build=conan/alpine
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build

FROM ubuntu:latest
COPY --from=builder /app/build/Release/demo/ExGrafDemo /app/ExGrafDemo
WORKDIR /app
CMD ["./ExGrafDemo"]
