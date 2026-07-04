FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    cmake \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 保留本地获得的 2025.1.0 SDK 源码，并校验核心目录
COPY upmem-sdk-private /opt/upmem-sdk-private

RUN test -f /opt/upmem-sdk-private/src/dpu-rt/Toolchain-dpu.cmake \
    && test -f /opt/upmem-sdk-private/src/backends/python/dpu/compiler.py

# 构建与本地 SDK 版本匹配的公开 UPMEM LLVM DPU 编译器
RUN git clone --depth 1 --branch 2025.1.0 \
    https://github.com/upmem/llvm-project.git \
    /opt/upmem-llvm

RUN cmake -G Ninja \
    -S /opt/upmem-llvm/llvm \
    -B /opt/upmem-llvm/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=clang \
    -DLLVM_TARGETS_TO_BUILD=DPU \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_BUILD_TESTS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_LIBXML2=OFF \
    -DLLVM_ENABLE_LIBEDIT=OFF \
    -DLLVM_ENABLE_LIBPFM=OFF \
    -DLLVM_ENABLE_Z3_SOLVER=OFF

RUN cmake --build /opt/upmem-llvm/build \
    --target clang \
    --parallel 2

# 为 uPIMulator 保留其写死的 2023.2.0 路径，
# 实际包装 2025.1.0 的 DPU clang。
RUN install -d /root/upmem-2023.2.0-Linux-x86_64/bin \
    && printf '%s\n' \
        '#!/usr/bin/env bash' \
        'set -e' \
        '' \
        'if [ -z "${DPU_ARCH_VERSION:-}" ]; then' \
        '    export DPU_ARCH_VERSION="v1A"' \
        'fi' \
        '' \
        'exec /opt/upmem-llvm/build/bin/clang --target=dpu-upmem-dpurte -mcpu="${DPU_ARCH_VERSION}" -DNR_THREADS=24 -DNR_ATOMIC_BITS=256 "$@" -I/opt/upmem-sdk-private/src/dpu-rt/src/stdlib -I/opt/upmem-sdk-private/src/dpu-rt/src/syslib' \
        > /root/upmem-2023.2.0-Linux-x86_64/bin/dpu-upmem-dpurte-clang

RUN chmod 755 /root/upmem-2023.2.0-Linux-x86_64/bin/dpu-upmem-dpurte-clang \
    && printf 'int main(void) { return 0; }\n' > /tmp/dpu_probe.c \
    && /root/upmem-2023.2.0-Linux-x86_64/bin/dpu-upmem-dpurte-clang \
       -S /tmp/dpu_probe.c \
       -o /tmp/dpu_probe.s \
    && test -s /tmp/dpu_probe.s

RUN chmod 755 /root

WORKDIR /root/uPIMulator
