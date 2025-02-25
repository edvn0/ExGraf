FROM ubuntu:latest AS builder

RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  git \
  python3 \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV CONAN_USER_HOME="/app/.conan"
RUN pip3 install conan --break-system-packages && mkdir -p $CONAN_USER_HOME

COPY conan/ /app/conan/
COPY conanfile.py /app/

RUN conan install . --build=missing --profile:host=./conan/alpine --profile:build=./conan/alpine

COPY . .

RUN conan build . --build=missing -s=build_type=Release --profile:host=./conan/alpine --profile:build=./conan/alpine

FROM ubuntu:latest

WORKDIR /app
COPY --from=builder /app/build/Release/demo/ExGrafDemo /app/trainer
CMD ["./trainer"]
