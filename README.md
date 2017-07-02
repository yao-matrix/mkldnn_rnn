# How to build
1. put this git into ./tensorflow/contrib folder
2. $ patch -p1 < compile.patch
3. $ bazel build --config=mkl -c opt //tensorflow/contrib/mkldnn_rnn:python/ops/_mkldnn_rnn_ops.so
