FROM alpine:3.17

RUN apk update && apk add --no-cache \
    build-base \
    cmake \
    ninja \
		autoconf \
    python3 \
		linux-headers \
		perl \
    py3-pip && \
		pip install --upgrade pip && \
		pip install conan

WORKDIR /app
COPY conan/alpine /root/.conan2/profiles/default
COPY . .
RUN conan install . --build=missing -s build_type=Release
RUN cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release
RUN conan build . --build=missing -s=build_type=Release
RUN ctest --test-dir build/Release --output-on-failure
CMD ["./build/Release/ExGraf"]

