FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

MAINTAINER Tzu Ming Hsu <stmharry@mit.edu>

ENV BAZEL_VERSION=0.5.0 \
    TF_BRANCH=r1.2 \
    FAISS_BRANCH=master \
    SLENDER_BRANCH=master

RUN apt-get update && apt-get install -y --no-install-recommends \
        apache2 \
        apache2-dev \
        ca-certificates \
        curl \
        g++ \
        libapache2-mod-wsgi \
        libcupti-dev \
        libopenblas-dev \
        python \
        python-dev \
        python-numpy \
        swig \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py | python

# BAZEL
RUN curl -L https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh -o bazel-installer.sh && \
    bash bazel-installer.sh && \
    rm bazel-installer.sh

# TENSORFLOW
ENV TF_ENABLE_XLA=1 \
    TF_NEED_CUDA=1 \
    TF_CUDA_COMPUTE_CAPABILITIES=3.0,3.5,5.2,6.0,6.1 \
    TF_CUDNN_VERSION=6 \
    CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu

RUN curl -L https://github.com/tensorflow/tensorflow/archive/$TF_BRANCH.tar.gz | tar -zx -C / && \
    mv /tensorflow-* /tensorflow

WORKDIR /tensorflow
RUN yes "" | ./configure && \
    bazel build -c opt --config=cuda tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
    pip install --no-cache-dir --upgrade /tmp/pip/tensorflow-*.whl && \
    rm -rf /tensorflow && \
    rm -rf /tmp/pip && \
    rm -rf /root/.cache

# FAISS
RUN curl -L https://github.com/facebookresearch/faiss/archive/master.tar.gz | tar -zx -C / && \
    mv /faiss-* /faiss

WORKDIR /faiss
RUN mv example_makefiles/makefile.inc.Linux makefile.inc && \
    echo PYTHONCFLAGS+=-I/usr/local/lib/python2.7/dist-packages/numpy/core/include >> makefile.inc && \
    echo BLASLDFLAGS=/usr/lib/libopenblas.so.0 >> makefile.inc && \
    make -j $(nproc) && \
    cd gpu && \
    make -j $(nproc) && \
    make py && \
    mv /faiss/faiss.py /faiss/swigfaiss_gpu.py /faiss/_swigfaiss_gpu.so /usr/local/lib/python2.7/dist-packages && \
    rm -rf /faiss

# SLENDER
RUN curl -L https://github.com/stmharry/slender/archive/$SLENDER_BRANCH.tar.gz | tar -zx -C /usr/src && \
    mv /usr/src/slender-* /usr/src/slender/ && \
    rm -r /usr/src/slender/model

WORKDIR /
