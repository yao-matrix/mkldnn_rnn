#!/bin/bash

bazel clean

# build all
bazel build --config=mkl --copt="-DEIGEN_USE_VML" --copt="-mfma" --copt="-mavx2" --copt="-march=broadwell" --copt="-O3" -s -c opt //tensorflow/tools/pip_package:build_pip_package

# generate wheel package
bazel-bin/tensorflow/tools/pip_package/build_pip_package ${PWD}

# install
pip install --upgrade ${PWD}/tensorflow-1.2.1-cp27-cp27mu-linux_x86_64.whl

# if you want to only build mkldnn_rnn
# bazel build --config=mkl -s -c opt //tensorflow/contrib/mkldnn_rnn:python/ops/_mkldnn_rnn_ops.so

# log level setting
# export TF_CPP_MIN_LOG_LEVEL=3

