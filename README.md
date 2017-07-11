patch -p1 < compile.patch
bazel build --config=mkl -c opt //tensorflow/contrib/mkldnn_rnn:python/ops/_mkldnn_rnn_ops.so  