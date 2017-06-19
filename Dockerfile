FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

MAINTAINER Tzu Ming Hsu <stmharry@mit.edu>

RUN apt-get update && apt-get install -y --no-install-recommends \
        apache2 \
        apache2-dev \
        ca-certificates \
        curl \
        git \
        g++ \
        libapache2-mod-wsgi \
        libcupti-dev \
        python \
        python-dev \
        python-numpy \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py | python

# BAZEL
ENV BAZEL_VERSION=0.5.0

RUN curl -L https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh -o bazel-installer.sh && \
    chmod +x bazel-installer.sh && \
    ./bazel-installer.sh && \
    rm bazel-installer.sh

# TENSORFLOW
ENV TF_BRANCH=r1.2 \
    TF_ENABLE_XLA=1 \
    TF_NEED_CUDA=1 \
    TF_CUDA_COMPUTE_CAPABILITIES=3.0,3.5,5.2,6.0,6.1 \
    TF_CUDNN_VERSION=6 \
    CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu

RUN git clone https://github.com/tensorflow/tensorflow --branch $TF_BRANCH --single-branch /usr/src/tensorflow

WORKDIR /usr/src/tensorflow
RUN yes "" | ./configure && \
    bazel build -c opt --config=cuda tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
    pip install --no-cache-dir --upgrade /tmp/pip/tensorflow-*.whl && \
    rm -rf /usr/src/tensorflow && \
    rm -rf /tmp/pip && \
    rm -rf /root/.cache

# SLENDER
RUN git clone https://github.com/stmharry/slender.git /usr/src/slender && \
    rm -r /usr/src/slender/model
